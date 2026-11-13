"""Offer enumeration, roster settling, the deterministic screen and pruning (design section 7).

* ``roster_after`` applies a trade to one roster: players leave and arrive, an incoming player
  who is IL-eligible takes a free IL slot, and the active count is compared with the roster limit.
* ``Screen.settle`` finishes the roster: a long roster drops its least costly player (tried among
  the three lowest-value candidates), a short one adds the free agent that helps most (or nobody).
* ``screen_offers`` scores every legal offer of the configured shapes for both sides by the
  deterministic lineup value, prunes, and returns the shortlist for full simulation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Offer:
    partner: int
    give: tuple[str, ...]  # you give
    get: tuple[str, ...]  # you get

    @property
    def shape(self) -> str:
        return f"{len(self.give)}-{len(self.get)}"

    @property
    def id(self) -> str:
        return f"{self.partner}:{'+'.join(sorted(self.give))}>{'+'.join(sorted(self.get))}"


@dataclass
class Settled:
    """A roster after the trade and its forced moves."""

    pids: list[str]  # every player incl. IL
    adds: list[str] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)
    il: list[str] = field(default_factory=list)


def roster_after(ctx, rid: int, out: tuple[str, ...], inn: tuple[str, ...]) -> tuple[list[str], list[str], int]:
    """(players, IL players, open active spots) after ``out`` leave and ``inn`` arrive."""
    snap, rules = ctx.snap, ctx.snap.rules
    r = snap.rosters[rid]
    players = [p for p in r.players if p not in out] + [p for p in inn if p not in r.players]
    reserve = [p for p in r.reserve if p not in out]
    for p in inn:
        info = snap.players.get(p)
        if info and rules.il_eligible(info.injury_status) and len(reserve) < rules.reserve_slots:
            reserve.append(p)
    active = len(players) - len(reserve)
    return players, reserve, rules.roster_limit - active


def legal_after(ctx, rid: int, out, inn) -> bool:
    """Every player must have known eligibility, and IL occupants must be IL-eligible when added."""
    return all(p in ctx.elig and ctx.elig[p].any() for p in inn)


class Screen:
    """Deterministic roster value over the weeks a trade could affect."""

    def __init__(self, ctx, world, first_col: int):
        self.ctx, self.world = ctx, world
        self.v = world.valuer
        self.cols = [c for c in world.future_cols if c >= first_col]
        self.H = max(1, len(self.cols))
        self.cache: dict[frozenset, float] = {}
        self.esum = {p: float(self.v.E[self.v.i[p], self.cols].sum()) for p in self.v.pid}
        fa = [p for p in ctx.fa_ids if p in self.v.i]
        self.fa_all = sorted(fa, key=lambda p: -self.esum[p])
        picks = list(self.fa_all[:4])
        for slot_i in range(self.v.n_slots):  # best free agent for each slot type
            best = next((p for p in self.fa_all if ctx.elig[p][slot_i]), None)
            if best and best not in picks:
                picks.append(best)
        self.fa_screen = picks

    def value(self, pids) -> float:
        key = frozenset(pids)
        if key not in self.cache:
            self.cache[key] = self.v.lineup_value(list(key), self.cols)
        return self.cache[key]

    def lowest(self, pool: list[str], k: int) -> list[str]:
        return sorted(pool, key=lambda p: self.esum.get(p, 0.0))[:k]

    def settle(self, rid: int, out, inn, n_drop_cands: int = 3, fa: list[str] | None = None
               ) -> tuple[float, Settled] | None:
        """Best screened roster after the trade. None if the trade cannot be legal."""
        if not legal_after(self.ctx, rid, out, inn):
            return None
        players, reserve, open_ = roster_after(self.ctx, rid, out, inn)
        pids, adds, drops = list(players), [], []
        active = [p for p in players if p not in reserve]
        for _ in range(max(0, -open_)):  # required drops
            cands = self.lowest([p for p in active if p not in drops], n_drop_cands)
            d = max(cands, key=lambda c: self.value([p for p in pids if p != c]))
            drops.append(d)
            pids.remove(d)
        for _ in range(min(2, max(0, open_))):  # freed spots
            base = self.value(pids)
            best, best_v = None, base
            for f in (fa if fa is not None else self.fa_screen):
                if f in pids or f in adds:
                    continue
                v = self.value(pids + [f])
                if v > best_v + 1e-9:
                    best, best_v = f, v
            if best is None:
                break
            adds.append(best)
            pids.append(best)
        return self.value(pids), Settled(pids, adds, drops, list(reserve))

    def settle_options(self, rid: int, out, inn, k_fa: int, k_drop: int) -> list[Settled]:
        """Candidate completions to be judged by simulation (design: your three lowest-value drops,
        and the few free agents that help most, plus leaving the spot empty)."""
        players, reserve, open_ = roster_after(self.ctx, rid, out, inn)
        active = [p for p in players if p not in reserve]
        drop_sets = [[]]
        if open_ < 0:
            drop_sets = [[d] for d in self.lowest(active, k_drop)] if open_ == -1 else \
                [list(c) for c in itertools.combinations(self.lowest(active, k_drop + 1), -open_)]
        out_opts: list[Settled] = []
        for ds in drop_sets:
            base = [p for p in players if p not in ds]
            n_open = max(0, open_)
            adds_list: list[list[str]] = [[]]
            if n_open:
                bv = self.value(base)
                scored = sorted(((self.value(base + [f]) - bv, f) for f in self.fa_all[:30] + self.fa_screen
                                 if f not in base), reverse=True)
                top = list(dict.fromkeys(f for _, f in scored))[:k_fa]
                adds_list += [[f] for f in top]
            for ad in adds_list:
                out_opts.append(Settled(base + ad, ad, ds, list(reserve)))
        return out_opts


def subsets(pids: list[str], sizes: set[int]):
    for k in sorted(sizes):
        yield from itertools.combinations(sorted(pids), k)


@dataclass
class Row:
    offer: Offer
    my_gain: float  # screened FP per week
    their_gain: float | None = None
    my: Settled | None = None
    their: Settled | None = None
    market_change: float | None = None


def waiver_screen(ctx, screen: Screen, rid: int) -> float:
    """Best screened waiver-only move for ``rid``: add a free agent into an open spot, or drop and add."""
    base_p = [p for p in ctx.snap.rosters[rid].players if p in screen.v.i]
    base = screen.value(base_p)
    reserve = set(ctx.snap.rosters[rid].reserve)
    active = [p for p in base_p if p not in reserve]
    best = 0.0
    open_ = ctx.snap.rules.roster_limit - len(active)
    for f in screen.fa_screen:
        if open_ > 0:
            best = max(best, screen.value(base_p + [f]) - base)
        for d in screen.lowest(active, 3):
            best = max(best, screen.value([p for p in base_p if p != d] + [f]) - base)
    return best / screen.H


def screen_offers(ctx, screen: Screen, market, log=print) -> tuple[list[Row], dict]:
    """Enumerate, screen both sides, prune. Returns (shortlist rows, funnel statistics)."""
    from trade.followup import complete  # local: followup imports this module

    cfg = ctx.tcfg.search
    me = ctx.snap.my_roster_id
    sizes = {s for sh in cfg.shapes for s in map(int, sh.split("-"))}
    shapes = {tuple(map(int, sh.split("-"))) for sh in cfg.shapes}
    mine = [p for p in ctx.snap.rosters[me].players if p in ctx.elig and p in screen.v.i]
    base_me = screen.value([p for p in ctx.snap.rosters[me].players if p in screen.v.i])
    waiver_bar = waiver_screen(ctx, screen, me)
    stats = {"candidates": 0, "legal": 0, "my_gain_positive": 0, "beats_waiver": 0, "partner_ok": 0,
             "dominated": 0, "waiver_bar": waiver_bar}
    kept: dict[str, Row] = {}
    allvals: dict[tuple, tuple[float, float | None]] = {}
    tol = cfg.partner_loss_tolerance_fp
    for pr in ctx.fmt.roster_ids:
        if pr == me:
            continue
        theirs = [p for p in ctx.snap.rosters[pr].players if p in ctx.elig and p in screen.v.i]
        base_th = screen.value([p for p in ctx.snap.rosters[pr].players if p in screen.v.i])
        for a, b in sorted(shapes):
            for g in itertools.combinations(sorted(mine), a):
                for r in itertools.combinations(sorted(theirs), b):
                    stats["candidates"] += 1
                    s_me = screen.settle(me, g, r)
                    if s_me is None:
                        continue
                    stats["legal"] += 1
                    my_gain = (s_me[0] - base_me) / screen.H
                    allvals[(pr, frozenset(g), frozenset(r))] = (my_gain, None)
                    if my_gain <= 0:
                        continue
                    stats["my_gain_positive"] += 1
                    if my_gain <= waiver_bar:
                        continue
                    stats["beats_waiver"] += 1
                    s_th = complete(screen, pr, r, g, exclude=set(s_me[1].adds))  # his follow-up move included
                    if s_th is None:
                        continue
                    their = (s_th[0] - base_th) / screen.H
                    mc = market.change(g, r)  # partner's market-perceived change
                    allvals[(pr, frozenset(g), frozenset(r))] = (my_gain, their)
                    if their < -tol and mc < 0:
                        continue
                    stats["partner_ok"] += 1
                    o = Offer(pr, g, r)
                    kept[o.id] = Row(o, my_gain, their, s_me[1], s_th[1], mc)
    # dominance: drop a package if one of its smaller parts is at least as good for you and no worse for him
    out = []
    for row in kept.values():
        o, dom = row.offer, False
        parts = []
        if len(o.give) == 2:
            parts += [(set(o.give) - {x}, set(o.get)) for x in o.give]
        if len(o.get) == 2:
            parts += [(set(o.give), set(o.get) - {x}) for x in o.get]
        for g, r in parts:
            sub = allvals.get((o.partner, frozenset(g), frozenset(r)))
            if sub and sub[1] is not None and sub[0] >= row.my_gain and sub[1] >= row.their_gain:
                dom = True
                break
        if dom:
            stats["dominated"] += 1
        else:
            out.append(row)
    out.sort(key=lambda r: -r.my_gain)
    per: dict[int, int] = {}
    short: list[Row] = []
    for row in out:
        if per.get(row.offer.partner, 0) >= cfg.per_partner:
            continue
        per[row.offer.partner] = per.get(row.offer.partner, 0) + 1
        short.append(row)
        if len(short) >= cfg.shortlist_size:
            break
    stats["shortlist"] = len(short)
    return short, stats

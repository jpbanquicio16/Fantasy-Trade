"""The recommendation pipeline (design section 7): enumerate -> screen -> prune -> simulate the
shortlist -> re-evaluate finalists on fresh scenarios -> bar -> cards, or "no worthwhile trade".
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from trade import __version__
from trade.config import TradeConfig
from trade.context import Context, World, build_context
from trade.evaluate import OBJ_KEY, Choice, Est, Evaluator, Result, best_alt
from trade.followup import resolve_claims, try_follow_up
from trade.format import effective_date, first_new_week, processing_time
from trade.market import Market, build_market, tier
from trade.search import Offer, Screen, Settled, screen_offers


@dataclass
class Finalist:
    choice: Choice
    row: object
    market_change: float
    g_sel: float  # selection-stage objective gain (biased upward)
    result: Result | None = None  # fresh-scenario result
    g: Est | None = None
    me: dict[str, Est] = field(default_factory=dict)
    them: dict[str, Est] = field(default_factory=dict)
    alt_est: Est | None = None
    tier: str = ""
    qualifies: bool = False
    why_not: str = ""
    sens: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class Recommendation:
    ctx: Context
    objective: str
    finalists: list[Finalist]
    cards: list[Finalist]
    funnel: dict
    market: Market
    timing: dict
    n_sel: int
    n_fresh: int
    elapsed_s: float
    rec_id: str = ""
    no_trade: bool = False
    near_miss: Finalist | None = None


def alt_options(ev: Evaluator, screen: Screen, k_fa: int, k_drop: int) -> list[Settled]:
    """Waiver-only rosters for you: add into an open spot, or drop one of the lowest-value and add."""
    ctx, me = ev.ctx, ev.ctx.snap.my_roster_id
    r = ctx.snap.rosters[me]
    pids = [p for p in r.players if p in screen.v.i]
    active = [p for p in pids if p not in set(r.reserve)]
    open_ = ctx.snap.rules.roster_limit - len(active)
    base = screen.value(pids)
    out = []
    drops = [None] if open_ > 0 else []
    drops += screen.lowest(active, k_drop)
    cands = sorted(screen.fa_all[:30] + screen.fa_screen, key=lambda f: -(screen.value(pids + [f]) - base))
    for f in list(dict.fromkeys(cands))[:k_fa + 3]:
        for d in drops:
            base_p = [p for p in pids if p != d]
            out.append(Settled(base_p + [f], [f], [d] if d else [], list(r.reserve)))
    return out


def select_choice(ev: Evaluator, screen: Screen, row, objective: str, cfg: TradeConfig, alt: Settled | None
                  ) -> tuple[Choice, float]:
    """Pick your required drop / freed-spot add by simulation (not by the screen); the partner's
    completion is the screened one (the most likely human choice)."""
    me = ev.ctx.snap.my_roster_id
    o = row.offer
    opts = screen.settle_options(me, o.give, o.get, cfg.search.fa_candidates, cfg.search.drop_candidates)
    key = OBJ_KEY[objective]
    best, best_v = None, -1e18
    for opt in opts:
        d = ev.diff({me: opt.pids, o.partner: row.their.pids}, [me], partner=o.partner)[me]
        v = float(d[key].mean())
        if v > best_v:
            best, best_v = opt, v
    ch = Choice(o, best, row.their, alt)
    return ch, best_v


def passes_bar(g: Est, cfg: TradeConfig, objective: str) -> tuple[bool, str]:
    need = cfg.bar_for(objective)
    if g.mean <= need:
        return False, f"gain {g.mean:+.2f} is not above the {need:g} bar"
    if g.se >= g.mean * cfg.bar.se_fraction:
        return False, f"standard error {g.se:.2f} is not below {cfg.bar.se_fraction:.2f} of the gain"
    if cfg.bar.require_p10_positive and g.p10 <= 0:
        return False, f"10th percentile of the model band ({g.p10:+.2f}) is not above zero"
    return True, ""


def trade_timing(ctx: Context, tcfg: TradeConfig) -> tuple[dict, object]:
    """When a trade proposed today would process and the first week it counts (section 2)."""
    from waiver.config import ET, load_league

    snap, fmt = ctx.snap, ctx.fmt
    today = snap.taken_at.astimezone(ET).date()
    eff = effective_date(today, fmt.review_days, tcfg.rules.assume_accepted_in_days)
    week_dates = {w: snap.week_dates(w) for w in sorted({g.week for g in snap.schedule})}
    fw, frac = first_new_week(week_dates, eff, snap.week, tcfg.rules.traded_player_week_rule)
    return {"effective_date": eff, "first_week": fw, "frac": frac,
            "process_at": processing_time(eff, load_league().display_tz), "today": today,
            "deadline_week": fmt.trade_deadline_week, "review_days": fmt.review_days}, today


def recommend(tcfg: TradeConfig, objective: str | None = None, partner: int | None = None, refresh: bool = True,
              ctx: Context | None = None, log=print) -> Recommendation:
    t0 = time.monotonic()
    objective = objective or tcfg.objective
    ctx = ctx or build_context(tcfg, refresh, log)
    snap, fmt = ctx.snap, ctx.fmt
    timing, today = trade_timing(ctx, tcfg)
    fw, frac = timing["first_week"], timing["frac"]
    if fmt.trade_deadline_week and fw > fmt.trade_deadline_week:
        ctx.flags.append(f"past_trade_deadline:week_{fw}>{fmt.trade_deadline_week}")
        return Recommendation(ctx, objective, [], [], {"blocked": "trade deadline"}, Market(), timing, 0, 0,
                              time.monotonic() - t0, no_trade=True)

    log(f"simulating {tcfg.scenarios.shortlist} scenarios of the whole league through week {fmt.final_week} ...")
    world = World(ctx, tcfg.scenarios.shortlist, ctx.seed, log=log)
    ev = Evaluator(world, ctx, fw, frac)
    screen = Screen(ctx, world, ev.first_col)
    market = build_market(ctx, __import__("waiver.ingest.sleeper", fromlist=["x"]).SleeperClient())
    log(f"screening offers (first week that counts: {fw}, {len(screen.cols)} weeks) ...")
    short, funnel = screen_offers(ctx, screen, market, log)
    if partner is not None:
        short = [r for r in short if r.offer.partner == partner]
    log(f"funnel: {funnel}  [{time.monotonic() - t0:.0f}s]")

    alt_opts = alt_options(ev, screen, tcfg.search.fa_candidates, tcfg.search.drop_candidates)
    alt, alt_v = best_alt(ev, alt_opts, objective)
    funnel["best_waiver_only_sim"] = alt_v

    log(f"simulating {len(short)} shortlisted offers ... [{time.monotonic() - t0:.0f}s]")
    scored: list[Finalist] = []
    for row in short:
        ch, v = select_choice(ev, screen, row, objective, tcfg, alt)
        ch, r = try_follow_up(ev, screen, ch, v, objective, set(row.offer.get))
        ch, r = resolve_claims(ev, screen, ch, r, objective)
        scored.append(Finalist(ch, row, row.market_change or 0.0, float(r.g.mean())))
    scored.sort(key=lambda f: -f.g_sel)
    fin = scored[: tcfg.search.finalists]

    log(f"re-evaluating {len(fin)} finalists on {tcfg.scenarios.finalist} fresh scenarios ... [{time.monotonic() - t0:.0f}s]")
    world2 = World(ctx, tcfg.scenarios.finalist, ctx.seed + 104729, log=log)
    ev2 = Evaluator(world2, ctx, fw, frac)
    if tcfg.finalist_policy == "v2":
        attach_v2(ev2, world2)
    for f in fin:
        fill_finalist(f, ev2, world2, ctx, tcfg, objective, market, today)
    fin.sort(key=lambda f: -(f.g.mean if f.g else -1e9))

    tier_order = {t: i for i, t in enumerate(("Looks good for both", "Plausible on reputation",
                                              "Needs a case", "Unlikely"))}
    ok = [f for f in fin if f.qualifies and f.tier != "Unlikely"]
    ok.sort(key=lambda f: (tier_order[f.tier], -f.g.mean))
    cards, per = [], {}
    for f in ok:
        p = f.choice.offer.partner
        if per.get(p, 0) >= tcfg.cards.max_per_partner or len(cards) >= tcfg.cards.max_cards:
            continue
        per[p] = per.get(p, 0) + 1
        cards.append(f)
    near = fin[0] if fin and not cards else None
    rec = Recommendation(ctx, objective, fin, cards, funnel, market, timing, world.n, world2.n,
                         time.monotonic() - t0, no_trade=not cards, near_miss=near)
    stamp = f"{snap.snapshot_id}|{objective}|{ctx.seed}|{tcfg.model_dump_json()}"
    rec.rec_id = f"t_{snap.taken_at:%Y%m%dT%H%M%SZ}_{hashlib.sha256(stamp.encode()).hexdigest()[:6]}"
    return rec


def fill_finalist(f: Finalist, ev2: Evaluator, world2: World, ctx: Context, tcfg: TradeConfig, objective: str,
                  market: Market, today) -> None:
    """Fresh-scenario values, the bar, the tier and the sensitivity runs (section 10)."""
    me, o = ctx.snap.my_roster_id, f.choice.offer
    r = ev2.run(f.choice, objective)
    f.result = r
    b = world2.sc.batch
    f.me = {k: Est.of(v, b) for k, v in r.me.items()}
    f.them = {k: Est.of(v, b) for k, v in r.them.items()}
    f.g = Est.of(r.g, b)
    f.alt_est = Est.of(r.alt[OBJ_KEY[objective]], b) if r.alt is not None else None
    f.qualifies, f.why_not = passes_bar(f.g, tcfg, objective)
    f.tier = tier(f.them["pts"].mean, f.market_change)
    s = f.sens
    # the other reading of the open trade-week rule
    other = "immediate" if tcfg.rules.traded_player_week_rule == "next_week_if_played" else "next_week_if_played"
    wd = {w: ctx.snap.week_dates(w) for w in sorted({g.week for g in ctx.snap.schedule})}
    fw2, fr2 = first_new_week(wd, effective_date(today, ctx.fmt.review_days,
                                                 tcfg.rules.assume_accepted_in_days), ctx.snap.week, other)
    ev_alt = Evaluator(world2, ctx, fw2, fr2)
    s["rule_reading"] = {"rule": other, "g": float(ev_alt.run(f.choice, objective).g.mean())}
    # other ten teams manage worse
    s["weak_others"] = float(ev2.run(f.choice, objective, others="weak").g.mean())
    # the partner's roster held fixed isolates the rival effect
    held = ev2.run(f.choice, objective, hold_partner=True)
    f.extra["own_effect"] = {k: float(v.mean()) for k, v in held.me.items()}
    # lineup role of each incoming player
    info: dict = {}
    world2.valuer.team_weeks([p for p in f.choice.my.pids if p in world2.valuer.i], ev2.cols, info=info)
    f.extra["starts"] = info.get("starts", {})
    f.extra["cover_value"] = info.get("cover_value", 0.0)
    # the freed spot left empty: what if the free agent is claimed first
    if f.choice.my.adds:
        bare = Settled([p for p in f.choice.my.pids if p not in f.choice.my.adds], [], f.choice.my.drops,
                       f.choice.my.il)
        f.extra["if_add_lost"] = float(ev2.run(Choice(o, bare, f.choice.their, f.choice.alt), objective).g.mean())
    # signs of the three objectives
    f.extra["objectives_disagree"] = len({np.sign(round(f.me[k].mean, 3)) for k in ("pts", "top6", "title")}) > 1
    # reversal: how far the incoming players' output can fall before G reaches zero
    s["flip_pct"] = flip_point(f, ev2, objective)
    for flag, cond in (("sensitive_to_rule_reading", np.sign(s["rule_reading"]["g"]) != np.sign(f.g.mean)),
                       ("sensitive_to_how_others_manage", np.sign(s["weak_others"]) != np.sign(f.g.mean))):
        if cond:
            f.extra.setdefault("flags", []).append(flag)


def flip_point(f: Finalist, ev2: Evaluator, objective: str) -> float | None:
    """Smallest fall (in %) in every incoming player's output at which G(T) is no longer positive."""
    inc = f.choice.offer.get
    lo, hi = 0.0, 0.6
    def g(d: float) -> float:
        return float(ev2.run(f.choice, objective, scale={p: 1 - d for p in inc}).g.mean())
    if g(lo) <= 0:
        return 0.0
    if g(hi) > 0:
        return None
    for _ in range(7):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
    return round(100 * (lo + hi) / 2, 1)


def attach_v2(ev: Evaluator, world: World) -> None:
    """Re-play the two traded teams with the lineup design's lock-aware policy (finalist tier)."""
    from waiver.sim.policy import RosterPlan, simulate_week
    import numpy as _np

    ctx = world.ctx
    plan_sc = __import__("waiver.sim.scenarios", fromlist=["x"]).sample(
        ctx.snap, ctx.proj, ctx.params, ctx.lib, ctx.mcfg, ctx.scfg, ctx.players, ctx.horizon,
        ctx.tcfg.scenarios.planning, world.seed + 7919)
    S, sc = len(ctx.snap.rules.starting_slots), world.sc

    def v2(rid: int, pids: list[str], cols: list[int]) -> _np.ndarray:
        names = [p for p in pids if p in sc.index]
        plan = RosterPlan(_np.array([sc.index[p] for p in names], dtype=_np.int64),
                          _np.array([ctx.elig[p] for p in names]), _np.ones((len(names), sc.horizon.n_days), bool))
        out = _np.zeros((sc.n, len(cols)))
        for k, c in enumerate(cols):
            out[:, k] = simulate_week(sc, plan_sc, plan, world.weeks[c], None, S, world.rank()).total / ctx.res
        return out

    ev.v2 = v2

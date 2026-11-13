"""Trades for a player you name (design section 7, "Trades for a player you name").

You name one player. If another manager owns him the tool works in *acquire* mode (what would it take
to get him?); if you own him, in *move* mode (what can you get for him?); a free agent needs no trade.
It uses the same valuation, screen, simulator and bar as the whole-league search; only the list of offers
changes, so it can afford bigger packages (up to three players from you) and more simulations (up to 80).

Pipeline: find the player -> list every offer that includes him -> settle both rosters, each side's best
single follow-up drop-and-add included -> screen and prune -> keep the screened trade-off front plus the
best offers of each shape -> simulate -> re-evaluate the price-list candidates on fresh scenarios -> return
a price list (best qualifying offer per shape), the trade-off between your gain and the owner's, and cards.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import re
import time
import unicodedata
from dataclasses import dataclass, field

from trade.config import Paths, TradeConfig
from trade.context import Context, World, build_context
from trade.evaluate import OBJ_KEY, UNIT, Choice, Evaluator, best_alt
from trade.market import Market, build_market, tier
from trade.recommend import (
    Finalist,
    Recommendation,
    alt_options,
    attach_v2,
    fill_finalist,
    select_choice,
    trade_timing,
)
from trade.followup import complete, follow_up, resolve_claims, try_follow_up  # noqa: F401
from trade.search import Offer, Row, Screen, Settled, waiver_screen

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
SHAPES_ACQUIRE = ["1-1", "2-1", "3-1", "1-2", "2-2", "3-2"]  # you give - you get; he is always among what you get
SHAPES_MOVE = ["1-1", "2-1", "1-2", "2-2"]  # he is always among what you give


class AmbiguousPlayer(Exception):
    def __init__(self, query: str, matches: list[str]):
        super().__init__(f"{query!r} matches {len(matches)} players")
        self.query, self.matches = query, matches


class PlayerNotFound(Exception):
    pass


@dataclass
class TargetConstraints:
    protect: set[str] = field(default_factory=set)  # player ids you will not give up
    max_send: int = 3  # most players you send
    shapes: list[str] | None = None  # None = the mode's defaults
    teams: set[int] | None = None  # move mode: only these partners
    shortlist: int = 80
    per_shape: int = 8


# --- finding the player ----------------------------------------------------------------
def norm_tokens(s: str) -> list[str]:
    """Lower-case, accents and punctuation removed, generational suffixes dropped."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    return [t for t in toks if t not in SUFFIXES]


def find_players(names: dict[str, str], query: str) -> list[str]:
    """Player ids whose name matches ``query``: an id, then an exact normalised name, then every query
    token among the name's tokens, then a substring. Returns the first non-empty tier."""
    if query in names:
        return [query]
    q = norm_tokens(query)
    if not q:
        return []
    toks = {p: norm_tokens(n) for p, n in names.items()}
    tiers = (
        [p for p, t in toks.items() if t == q],
        [p for p, t in toks.items() if all(any(w == x for w in t) for x in q)],
        [p for p, t in toks.items() if " ".join(q) in " ".join(t)],
        [p for p, t in toks.items() if "".join(q) in "".join(t)],  # "PJ Washington" vs "P.J. Washington"
        [p for p, t in toks.items() if all(any(w.startswith(x) for w in t) for x in q)],
    )
    return next((sorted(t) for t in tiers if t), [])


def resolve_player(ctx: Context, query: str) -> str:
    """One player id, or AmbiguousPlayer / PlayerNotFound. Players without a team or a projection are only
    kept when they are the sole match (retired namesakes sit in Sleeper's player map)."""
    names = {p: i.name for p, i in ctx.snap.players.items()}
    hits = find_players(names, query)
    if not hits:
        raise PlayerNotFound(f"no player matches {query!r}")
    if len(hits) > 1:
        live = [p for p in hits if p in ctx.proj and ctx.snap.players[p].team]
        hits = live or hits
    if len(hits) > 1:
        raise AmbiguousPlayer(query, hits)
    return hits[0]


def owner_of(ctx: Context, pid: str) -> int | None:
    return next((rid for rid, r in ctx.snap.rosters.items() if pid in r.players), None)


# --- listing every offer that includes him ---------------------------------------------------
def parse_shapes(shapes: list[str]) -> set[tuple[int, int]]:
    return {tuple(map(int, s.split("-"))) for s in shapes}  # type: ignore[misc]


def enumerate_offers(mode: str, target: str, mine: list[str], theirs: dict[int, list[str]], cons: TargetConstraints):
    """Yield every Offer including ``target``.

    acquire: ``theirs`` holds the owner only; you give 1..max_send of your players, you get him alone or with
    one more of the owner's. move: you give him alone or with one more of yours; you get 1-2 players from any
    team in ``theirs``.
    """
    shapes = parse_shapes(cons.shapes or (SHAPES_ACQUIRE if mode == "acquire" else SHAPES_MOVE))
    tradable = [p for p in mine if p not in cons.protect and p != target]
    if mode == "acquire":
        (pr, roster), = theirs.items()
        gets = [(target,)] + [(target, x) for x in sorted(roster) if x != target]
        for a in range(1, cons.max_send + 1):
            for give in itertools.combinations(sorted(tradable), a):
                for get in gets:
                    if (a, len(get)) in shapes:
                        yield Offer(pr, give, get)
    else:
        gives = [(target,)] + [(target, y) for y in sorted(tradable)]
        gives = [g for g in gives if len(g) <= cons.max_send]
        for pr, roster in sorted(theirs.items()):
            if cons.teams and pr not in cons.teams:
                continue
            for give in gives:
                for b in (1, 2):
                    if (len(give), b) not in shapes:
                        continue
                    for get in itertools.combinations(sorted(roster), b):
                        yield Offer(pr, give, get)


# --- settling, follow-up moves ----------------------------------------------------------------
def screen_target(ctx: Context, screen: Screen, market: Market, offers, log=print) -> tuple[list[Row], dict]:
    cfg, me = ctx.tcfg.search, ctx.snap.my_roster_id
    pids = lambda rid: [p for p in ctx.snap.rosters[rid].players if p in screen.v.i]  # noqa: E731
    base_me = screen.value(pids(me))
    base_th = {rid: screen.value(pids(rid)) for rid in ctx.fmt.roster_ids if rid != me}
    waiver_bar = waiver_screen(ctx, screen, me)
    st = {"listed": 0, "legal": 0, "my_gain_positive": 0, "beats_waiver": 0, "partner_ok": 0, "waiver_bar": waiver_bar}
    rows: list[Row] = []
    for o in offers:
        st["listed"] += 1
        a = complete(screen, me, o.give, o.get)
        if a is None:
            continue
        st["legal"] += 1
        my_gain = (a[0] - base_me) / screen.H
        if my_gain <= 0:
            continue
        st["my_gain_positive"] += 1
        if my_gain <= waiver_bar:
            continue
        st["beats_waiver"] += 1
        b = complete(screen, o.partner, o.get, o.give, exclude=set(a[1].adds))
        if b is None:
            continue
        their = (b[0] - base_th[o.partner]) / screen.H
        mc = market.change(o.give, o.get)  # the partner's market-perceived change
        if their < -cfg.partner_loss_tolerance_fp and mc < 0:
            continue
        st["partner_ok"] += 1
        rows.append(Row(o, my_gain, their, a[1], b[1], mc))
    return rows, st


def pareto(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not beaten on both coordinates (larger is better for each)."""
    order = sorted(range(len(points)), key=lambda i: (-points[i][0], -points[i][1]))
    out, best = [], float("-inf")
    for i in order:
        if points[i][1] > best:
            out.append(i)
            best = points[i][1]
    return out


def pick_shortlist(rows: list[Row], limit: int, per_shape: int) -> list[Row]:
    """The screened trade-off front plus the best ``per_shape`` offers of each shape, up to ``limit``."""
    if not rows:
        return []
    chosen: dict[str, Row] = {rows[i].offer.id: rows[i] for i in pareto([(r.my_gain, r.their_gain) for r in rows])}
    by_shape: dict[str, list[Row]] = {}
    for r in sorted(rows, key=lambda r: -r.my_gain):
        by_shape.setdefault(r.offer.shape, []).append(r)
    for rs in by_shape.values():
        for r in rs[:per_shape]:
            chosen.setdefault(r.offer.id, r)
    out = sorted(chosen.values(), key=lambda r: -r.my_gain)
    return out[:limit]


# --- result ---------------------------------------------------------------------------------------
@dataclass
class PriceRow:
    label: str
    f: Finalist


@dataclass
class TargetedResult:
    ctx: Context
    status: str  # ok | free_agent | no_trade | blocked
    mode: str = ""
    target: str = ""
    message: str = ""
    objective: str = "points"
    rec: Recommendation | None = None
    price_list: list[PriceRow] = field(default_factory=list)
    front: list[dict] = field(default_factory=list)  # simulated offers on the trade-off front
    simulated: list[dict] = field(default_factory=list)
    funnel: dict = field(default_factory=dict)
    near_miss: Finalist | None = None
    elapsed_s: float = 0.0


def recommend_for_player(tcfg: TradeConfig, query: str, cons: TargetConstraints | None = None,
                         objective: str | None = None, refresh: bool = True, ctx: Context | None = None,
                         log=print) -> TargetedResult:
    t0 = time.monotonic()
    cons = cons or TargetConstraints()
    objective = objective or tcfg.objective
    ctx = ctx or build_context(tcfg, refresh, log)
    snap, fmt = ctx.snap, ctx.fmt
    me = snap.my_roster_id
    target = resolve_player(ctx, query)
    name = snap.players[target].name
    if target not in ctx.elig or target not in ctx.proj:
        return TargetedResult(ctx, "blocked", target=target, objective=objective,
                              message=f"{name} has no projection or known position eligibility, so he cannot be valued")
    owner = owner_of(ctx, target)
    timing, today = trade_timing(ctx, tcfg)
    fw, frac = timing["first_week"], timing["frac"]
    if fmt.trade_deadline_week and fw > fmt.trade_deadline_week:
        ctx.flags.append(f"past_trade_deadline:week_{fw}>{fmt.trade_deadline_week}")
        return TargetedResult(ctx, "blocked", target=target, objective=objective,
                              message=f"a trade proposed now would process in week {fw}, after the week "
                                      f"{fmt.trade_deadline_week} deadline")

    log(f"simulating {tcfg.scenarios.shortlist} scenarios of the whole league through week {fmt.final_week} ...")
    world = World(ctx, tcfg.scenarios.shortlist, ctx.seed, log=log)
    ev = Evaluator(world, ctx, fw, frac)
    screen = Screen(ctx, world, ev.first_col)
    market = build_market(ctx, __import__("waiver.ingest.sleeper", fromlist=["x"]).SleeperClient())

    if owner is None:
        return free_agent_result(ctx, screen, target, objective, t0)
    mode = "move" if owner == me else "acquire"
    elig = lambda rid: [p for p in snap.rosters[rid].players if p in ctx.elig and p in screen.v.i]  # noqa: E731
    mine = elig(me)
    theirs = {owner: elig(owner)} if mode == "acquire" else {r: elig(r) for r in fmt.roster_ids if r != me}
    offers = list(enumerate_offers(mode, target, mine, theirs, cons))
    log(f"{mode} {name}: {len(offers):,} offers include him; screening ... [{time.monotonic() - t0:.0f}s]")
    rows, funnel = screen_target(ctx, screen, market, offers, log)
    short = pick_shortlist(rows, cons.shortlist, cons.per_shape)
    funnel.update(mode=mode, shortlist=len(short))
    alt, alt_v = best_alt(ev, alt_options(ev, screen, tcfg.search.fa_candidates, tcfg.search.drop_candidates), objective)
    funnel["best_waiver_only_sim"] = alt_v

    log(f"simulating {len(short)} offers ... [{time.monotonic() - t0:.0f}s]")
    bar = tcfg.bar_for(objective)
    scored: list[Finalist] = []
    for row in short:
        ch, v = select_choice(ev, screen, row, objective, tcfg, alt)
        ch, r = try_follow_up(ev, screen, ch, v, objective, set(row.offer.get))
        ch, r = resolve_claims(ev, screen, ch, r, objective)
        f = Finalist(ch, row, row.market_change or 0.0, float(r.g.mean()))
        f.extra["them_sel"] = float(r.them["pts"].mean())
        f.extra["tier_sel"] = tier(f.extra["them_sel"], f.market_change)
        scored.append(f)

    fin = choose_finalists(scored, bar)
    log(f"re-evaluating {len(fin)} price-list candidates on {tcfg.scenarios.finalist} fresh scenarios ... "
        f"[{time.monotonic() - t0:.0f}s]")
    world2 = World(ctx, tcfg.scenarios.finalist, ctx.seed + 104729, log=log)
    ev2 = Evaluator(world2, ctx, fw, frac)
    if tcfg.finalist_policy == "v2":
        attach_v2(ev2, world2)
    for f in fin:
        fill_finalist(f, ev2, world2, ctx, tcfg, objective, market, today)
    fin.sort(key=lambda f: -(f.g.mean if f.g else -1e9))

    price = build_price_list(fin)
    live_fin = [f for f in fin if f.tier != "Unlikely"]
    near = (live_fin or fin)[0] if fin and not price else None
    cards = [p.f for p in price]
    rec = Recommendation(ctx, objective, fin, cards, funnel, market, timing, world.n, world2.n,
                         time.monotonic() - t0, no_trade=not cards, near_miss=near)
    stamp = f"{snap.snapshot_id}|target|{target}|{objective}|{ctx.seed}|{tcfg.model_dump_json()}"
    rec.rec_id = f"tg_{snap.taken_at:%Y%m%dT%H%M%SZ}_{hashlib.sha256(stamp.encode()).hexdigest()[:6]}"

    sim_rows = [{"shape": f.choice.offer.shape, "partner": f.choice.offer.partner, "offer_id": f.choice.offer.id,
                 "label": offer_label(ctx, f.choice.offer), "your_gain": f.g_sel, "their_pts": f.extra["them_sel"], "market_change": f.market_change,
                 "tier": f.extra["tier_sel"]} for f in scored]
    clearing = [s for s in sim_rows if s["your_gain"] > bar]
    shown = [s for s in clearing if s["tier"] != "Unlikely"]  # 'unlikely' offers are hidden, as on the cards
    front = [shown[i] for i in pareto([(s["your_gain"], s["their_pts"]) for s in shown])]
    funnel["clear_bar"] = len(clearing)
    funnel["clear_bar_not_unlikely"] = sum(s["tier"] != "Unlikely" for s in clearing)
    funnel["owner_gains"] = sum(s["their_pts"] > 0 for s in clearing if s["tier"] != "Unlikely")
    res = TargetedResult(ctx, "ok" if cards else "no_trade", mode, target, objective=objective, rec=rec,
                         price_list=price, front=front, simulated=sim_rows, funnel=funnel,
                         near_miss=rec.near_miss, elapsed_s=time.monotonic() - t0)
    write_csv(res, name)
    return res


def offer_label(ctx: Context, o: Offer) -> str:
    nm = lambda p: ctx.snap.players[p].name if p in ctx.snap.players else p  # noqa: E731
    return f"{', '.join(map(nm, o.give))} for {', '.join(map(nm, o.get))}"


def choose_finalists(scored: list[Finalist], bar: float, per_shape: int = 2, kind: int = 3, cap: int = 14):
    """Candidates for the price list (selection-stage values; the fresh stream decides): the best of each
    shape whose owner is outside the 'unlikely' tier (such offers are never shown), the kindest to the owner
    among those that clear the bar, and one best offer per shape regardless so a near miss can be explained."""
    live = [f for f in scored if f.extra["tier_sel"] != "Unlikely"]
    picked: dict[str, Finalist] = {}

    def best_by_shape(pool: list[Finalist], n: int) -> None:
        by_shape: dict[str, list[Finalist]] = {}
        for f in sorted(pool, key=lambda f: -f.g_sel):
            by_shape.setdefault(f.choice.offer.shape, []).append(f)
        for fs in by_shape.values():
            for f in fs[:n]:
                picked.setdefault(f.choice.offer.id, f)

    best_by_shape(live, per_shape)
    for f in sorted((f for f in live if f.g_sel > bar), key=lambda f: -f.extra["them_sel"])[:kind]:
        picked.setdefault(f.choice.offer.id, f)
    if len(picked) < cap // 2:  # little to show: add the best of every shape so a near miss exists
        best_by_shape(scored, 1)
    return list(picked.values())[:cap]


def build_price_list(fin: list[Finalist]) -> list[PriceRow]:
    """Best qualifying offer in each shape (owner not in the 'unlikely' tier), then the smallest owner loss."""
    ok = [f for f in fin if f.qualifies and f.tier != "Unlikely"]
    out: list[PriceRow] = []
    shapes = sorted({f.choice.offer.shape for f in ok}, key=lambda s: (sum(map(int, s.split("-"))), s))
    for sh in shapes:
        out.append(PriceRow(sh, max((f for f in ok if f.choice.offer.shape == sh), key=lambda f: f.g.mean)))
    if ok:
        kind = max(ok, key=lambda f: f.them["pts"].mean)
        if all(kind is not p.f for p in out):
            out.append(PriceRow("Smallest owner loss", kind))
    return out


def free_agent_result(ctx: Context, screen: Screen, target: str, objective: str, t0: float) -> TargetedResult:
    """He is on waivers: no trade is needed. Show the drop the waiver tool would pair with claiming him."""
    snap, me = ctx.snap, ctx.snap.my_roster_id
    r = snap.rosters[me]
    pids = [p for p in r.players if p in screen.v.i]
    active = [p for p in pids if p not in set(r.reserve)]
    base = screen.value(pids)
    name = snap.players[target].name
    if snap.rules.roster_limit - len(active) > 0:
        gain, drop = screen.value(pids + [target]) - base, None
    else:
        drop, gain = max(((d, screen.value([p for p in pids if p != d] + [target]) - base)
                          for d in screen.lowest(active, 3)), key=lambda x: x[1])
    msg = (f"{name} is a free agent, so no trade is needed. Claiming him "
           + ("fills an open spot" if drop is None else f"and dropping {snap.players[drop].name}")
           + f" is worth about {gain / screen.H:+.2f} FP a week by the screen (a rough guide; use the waiver tool "
             "for the simulated value).")
    return TargetedResult(ctx, "free_agent", "free_agent", target, msg, objective, elapsed_s=time.monotonic() - t0)


def write_csv(res: TargetedResult, name: str) -> None:
    if res.rec is None:
        return
    path = Paths().ensure().reports / f"target_{re.sub(r'[^a-z0-9]+', '_', name.lower())}_{res.rec.rec_id}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res.simulated[0]) if res.simulated else ["offer_id"])
        w.writeheader()
        w.writerows(res.simulated)
    res.funnel["csv"] = str(path)
    from trade.chart import tradeoff_svg

    bar = res.ctx.tcfg.bar_for(res.objective)
    svg = path.with_suffix(".svg")
    svg.write_text(tradeoff_svg(res.simulated, res.front, bar, f"Offers for {name}: your gain vs the owner's"))
    res.funnel["chart"] = str(svg)


# --- text output ---------------------------------------------------------------------------------
def render_targeted(res: TargetedResult) -> str:
    from trade.cards import OBJ_LABEL, _name, card, fallback  # noqa: F401

    ctx = res.ctx
    head = f"{res.rec.rec_id if res.rec else 'target'} · objective: {OBJ_LABEL[res.objective]} · {res.elapsed_s:.0f}s"
    if res.status in ("blocked", "free_agent"):
        return f"{head}\n\n{res.message}"
    rec, nm, fn = res.rec, _name(ctx, res.target), res.funnel
    owner = f"roster {owner_of(ctx, res.target)}" if res.mode == "acquire" else "you"
    out = [head, f"{res.mode.upper()} {nm} ({owner}) · {fn['listed']:,} offers include him -> {fn['legal']:,} legal -> "
           f"{fn['my_gain_positive']:,} you gain (screen) -> {fn['beats_waiver']:,} beat your best waiver move -> "
           f"{fn['partner_ok']:,} partner tolerates -> {fn['shortlist']} simulated", ""]
    if ctx.flags:
        out += ["flags: " + ", ".join(ctx.flags), ""]
    unit = UNIT[OBJ_KEY[res.objective]]
    if res.status == "no_trade":
        out.append(f"NO WORTHWHILE TRADE FOR {nm.upper()}. Nothing clears the bar (gain above the threshold, standard "
                   "error under a third of it, 10th percentile above zero, owner not in the 'unlikely' tier).")
        if res.near_miss:
            nm_ = res.near_miss
            out += ["", "Nearest miss:", card(rec, nm_, 1, 1), "", f"Why it misses: {nm_.why_not or 'owner in the unlikely tier'}."]
        out.append(f"Selection stage: {fn.get('clear_bar', 0)} of {fn['shortlist']} simulated offers cleared the bar, "
                   f"{fn.get('clear_bar_not_unlikely', 0)} with an owner outside the 'unlikely' tier. Common causes: he "
                   "duplicates positions you already fill, or every offer the owner might take costs you more than he adds.")
        return "\n".join(out)

    out.append(f"Clear your bar: {fn['clear_bar']} · owner not in the 'unlikely' tier: {fn['clear_bar_not_unlikely']} · "
               f"of which the owner gains by projection: {fn['owner_gains']}")
    out += ["", f"PRICE LIST (best qualifying offer per shape; gain in {unit}, fresh scenarios)", ""]
    hdr = f"{'Shape':<20}{'You give':<44}{'You get':<36}{'Your gain':>16}{'Owner':>9}  Tier"
    out.append(hdr)
    for p in res.price_list:
        o, f = p.f.choice.offer, p.f
        give = ", ".join(_name(ctx, x) for x in o.give)
        get = ", ".join(_name(ctx, x) for x in o.get)
        sh = p.label if p.label[0].isalpha() else f"{p.label} (roster {o.partner})"
        out.append(f"{sh:<20}{give:<44.43}{get:<36.35}{f.g.mean:>+9.2f} (SE {f.g.se:.2f}){f.them['pts'].mean:>+8.2f}  {f.tier}")
    if res.front:
        out += ["", "TRADE-OFF FRONT (selection-stage values; offers no other offer beats for both sides at once)", "",
                f"{'Offer':<74}{'Your gain':>12}{'Owner FP/wk':>13}  Tier"]
        for s in sorted(res.front, key=lambda s: -s["your_gain"]):
            label = s["label"]
            out.append(f"{label:<74.73}{s['your_gain']:>+12.2f}{s['their_pts']:>+13.2f}  {s['tier']}")
        if fn.get("csv"):
            out.append(f"\nevery simulated offer: {fn['csv']}")
        if fn.get("chart"):
            out.append(f"trade-off chart: {fn['chart']}")
    out += ["", "=" * 78, ""]
    for i, p in enumerate(res.price_list, 1):
        out += [card(rec, p.f, i, len(res.price_list)), "", "-" * 78, ""]
    return "\n".join(out)

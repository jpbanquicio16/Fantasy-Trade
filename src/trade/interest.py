"""When a manager asks for one of your players (design section 7, "When a manager asks for one of your players").

You name the manager, your player he wants and (if he made one) his offer. The tool

1. judges his offer with the same paired whole-league simulation as every other card,
2. builds counters that stay close to his offer (ask for one more, swap the piece, add one of yours for an
   upgrade) and ranks those that clear your bar by the smallest concession asked of him in market terms,
3. searches every deal with him that includes your player (one to three of his players, at most
   ``max_extra`` more of yours),
4. says "keep him" when nothing clears your bar.

His request is evidence: he asked first, so he probably values your player above the market's view. No
probability is shown; the card says so in words. The request is logged in ``proposals`` with his roster as the
sender, the first acceptance data the system gets.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import dataclass, field

from trade.config import TradeConfig
from trade.context import Context, World, build_context
from trade.evaluate import OBJ_KEY, UNIT, Evaluator, best_alt
from trade.followup import complete, resolve_claims, try_follow_up
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
from trade.search import Offer, Row, Screen
from trade.targeted import pick_shortlist, resolve_player, screen_target

HIS_OFFER = "His offer"
ASK_MORE = "Counter: ask for one more"
SWAP = "Counter: swap the piece"
UPGRADE = "Counter: add one of yours, upgrade"
FULL = "Full search with him"
COUNTER_KINDS = (ASK_MORE, SWAP, UPGRADE)


class InvalidRequest(Exception):
    """The request names a player who is not where it says (a stale or mistyped offer)."""


@dataclass
class InterestConstraints:
    protect: set[str] = field(default_factory=set)  # your players you will not add
    max_extra: int = 1  # most extra players of yours in a counter or a deal
    max_get: int = 3  # most players of his in a full-search deal
    counters_per_kind: int = 8  # screened counters simulated per kind
    full_shortlist: int = 40
    include_full: bool = True


@dataclass
class InboundResult:
    ctx: Context
    status: str  # ok | blocked
    message: str = ""
    objective: str = "points"
    partner: int = 0
    my_player: str = ""
    their_offer: tuple[str, ...] = ()
    rec: Recommendation | None = None
    his: Finalist | None = None
    counters: list[tuple[str, Finalist]] = field(default_factory=list)  # ranked: easiest to agree first
    best_deal: Finalist | None = None
    verdict: str = ""
    surplus: float | None = None  # his projected gain from his own offer (FP a week)
    funnel: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


# --- resolving the request -------------------------------------------------------------------
def resolve_partner(ctx: Context, who: str) -> int:
    """A roster id, or a unique owner-name match."""
    rosters = ctx.snap.rosters
    if who.strip().isdigit() and int(who) in rosters:
        return int(who)
    hits = [rid for rid, r in rosters.items() if r.owner_name and who.lower() in r.owner_name.lower()]
    if len(hits) != 1:
        raise InvalidRequest(f"{who!r} matches {len(hits)} managers; use a roster id")
    return hits[0]


def validate_request(ctx: Context, partner: int, my_player: str, their_offer: tuple[str, ...]) -> None:
    """Reject a request that names a player who is no longer where it says (section 10, inbound-mode test)."""
    snap, me = ctx.snap, ctx.snap.my_roster_id
    if partner == me or partner not in snap.rosters:
        raise InvalidRequest(f"roster {partner} is not another manager's")
    nm = lambda p: snap.players[p].name if p in snap.players else p  # noqa: E731
    if my_player not in snap.rosters[me].players:
        raise InvalidRequest(f"{nm(my_player)} is not on your roster")
    for p in their_offer:
        if p not in snap.rosters[partner].players:
            raise InvalidRequest(f"{nm(p)} is not on roster {partner}, so his offer is stale")
    for p in (my_player, *their_offer):
        if p not in ctx.elig or p not in ctx.proj:
            raise InvalidRequest(f"{nm(p)} has no projection or position eligibility, so he cannot be valued")


# --- listing offers (pure; tested without a league) ---------------------------------------------
def enumerate_counters(partner: int, my_player: str, their_offer: tuple[str, ...], mine: list[str],
                       theirs: list[str], cons: InterestConstraints) -> list[tuple[str, Offer]]:
    """Counters close to his offer, each tagged with its kind. His own offer is not among them."""
    base = tuple(their_offer)
    extras = [p for p in sorted(mine) if p != my_player and p not in cons.protect]
    others = [p for p in sorted(theirs) if p not in base]
    seen: dict[str, tuple[str, Offer]] = {}

    def add(kind: str, give: tuple[str, ...], get: tuple[str, ...]) -> None:
        o = Offer(partner, give, get)
        if o.id != Offer(partner, (my_player,), base).id:
            seen.setdefault(o.id, (kind, o))

    for x in others:
        add(ASK_MORE, (my_player,), base + (x,))
    for y in others:
        add(SWAP, (my_player,), (y,))
    for n in range(1, cons.max_extra + 1):
        for zs in itertools.combinations(extras, n):
            for y in sorted(theirs):
                add(UPGRADE, (my_player, *zs), (y,))
    return list(seen.values())


def enumerate_deals(partner: int, my_player: str, mine: list[str], theirs: list[str],
                    cons: InterestConstraints) -> list[Offer]:
    """Every deal with him that includes your player: one to ``max_get`` of his, up to ``max_extra`` more of yours."""
    extras = [p for p in sorted(mine) if p != my_player and p not in cons.protect]
    gives = [(my_player,)] + [(my_player, *zs) for n in range(1, cons.max_extra + 1)
                              for zs in itertools.combinations(extras, n)]
    return [Offer(partner, g, get) for g in gives
            for k in range(1, cons.max_get + 1) for get in itertools.combinations(sorted(theirs), k)]


def concession(his_market_change: float, f_market_change: float) -> float:
    """How much more (in market terms) a counter asks of him than his own offer does. Smaller is easier."""
    return his_market_change - f_market_change


def rank_counters(his_market_change: float, fins: list[tuple[str, Finalist]]) -> list[tuple[str, Finalist]]:
    """Counters that clear your bar and are not in the 'unlikely' tier, smallest concession first
    (ties to the larger gain for you)."""
    ok = [(k, f) for k, f in fins if k in COUNTER_KINDS and f.qualifies and f.tier != "Unlikely"]
    return sorted(ok, key=lambda kf: (concession(his_market_change, kf[1].market_change), -kf[1].g.mean))


def pick_counters(rows: list[tuple[str, Row]], per_kind: int) -> list[tuple[str, Row]]:
    """Simulate the screened-best ``per_kind`` counters of each kind (the screen shortlists, simulation decides)."""
    out: list[tuple[str, Row]] = []
    for kind in COUNTER_KINDS:
        rs = sorted((kr for kr in rows if kr[0] == kind), key=lambda kr: -kr[1].my_gain)
        out += rs[:per_kind]
    return out


def verdict_text(his: Finalist, ranked: list, best_deal: Finalist | None, nm: str, objective: str) -> str:
    unit = UNIT_NAME.get(objective, "")
    if his.qualifies and his.tier != "Unlikely":
        return (f"ACCEPTABLE AS OFFERED: it clears your bar ({his.g.mean:+.2f} {unit}). Counters below may get more.")
    why = his.why_not or "he is in the unlikely tier"
    if ranked:
        return f"DECLINE AS OFFERED ({why}); COUNTER with the first counter below, the smallest ask that clears your bar."
    if best_deal is not None and best_deal.qualifies:
        return f"DECLINE AS OFFERED ({why}); no counter near his offer clears your bar, but the best deal with him does."
    return f"KEEP {nm.upper()}: nothing with him clears your bar ({why})."


UNIT_NAME = {"points": "FP a week", "top6": "pts of P(top 6)", "title": "pts of P(title)"}


# --- the pipeline --------------------------------------------------------------------------------
def build_row(ctx: Context, screen: Screen, market: Market, o: Offer, base_me: float, base_th: dict) -> Row | None:
    """A screened row with no filters (his offer and counters are judged whatever the screen thinks)."""
    me = ctx.snap.my_roster_id
    a = complete(screen, me, o.give, o.get)
    if a is None:
        return None
    b = complete(screen, o.partner, o.get, o.give, exclude=set(a[1].adds))
    if b is None:
        return None
    return Row(o, (a[0] - base_me) / screen.H, (b[0] - base_th[o.partner]) / screen.H, a[1], b[1],
               market.change(o.give, o.get))


def respond_to_interest(tcfg: TradeConfig, partner: str | int, my_player: str, their_offer: list[str] | None = None,
                        cons: InterestConstraints | None = None, objective: str | None = None,
                        refresh: bool = True, ctx: Context | None = None, log=print) -> InboundResult:
    t0 = time.monotonic()
    cons = cons or InterestConstraints()
    objective = objective or tcfg.objective
    ctx = ctx or build_context(tcfg, refresh, log)
    snap, fmt = ctx.snap, ctx.fmt
    me = snap.my_roster_id
    pr = resolve_partner(ctx, str(partner))
    mine_p = resolve_player(ctx, my_player)
    offer_p = tuple(resolve_player(ctx, q) for q in their_offer or [])
    validate_request(ctx, pr, mine_p, offer_p)
    nm = lambda p: snap.players[p].name  # noqa: E731

    timing, today = trade_timing(ctx, tcfg)
    fw, frac = timing["first_week"], timing["frac"]
    if fmt.trade_deadline_week and fw > fmt.trade_deadline_week:
        ctx.flags.append(f"past_trade_deadline:week_{fw}>{fmt.trade_deadline_week}")
        return InboundResult(ctx, "blocked", objective=objective, partner=pr, my_player=mine_p,
                             message=f"a trade accepted now would process in week {fw}, after the week "
                                     f"{fmt.trade_deadline_week} deadline")

    log(f"simulating {tcfg.scenarios.shortlist} scenarios of the whole league through week {fmt.final_week} ...")
    world = World(ctx, tcfg.scenarios.shortlist, ctx.seed, log=log)
    ev = Evaluator(world, ctx, fw, frac)
    screen = Screen(ctx, world, ev.first_col)
    market = build_market(ctx, __import__("waiver.ingest.sleeper", fromlist=["x"]).SleeperClient())
    elig = lambda rid: [p for p in snap.rosters[rid].players if p in ctx.elig and p in screen.v.i]  # noqa: E731
    mine, theirs = elig(me), elig(pr)
    base_me = screen.value([p for p in snap.rosters[me].players if p in screen.v.i])
    base_th = {rid: screen.value([p for p in snap.rosters[rid].players if p in screen.v.i])
               for rid in fmt.roster_ids if rid != me}

    kinds: dict[str, str] = {}  # offer id -> kind
    rows: dict[str, Row] = {}
    funnel: dict = {}

    # 1. his offer (if any)
    his_offer = Offer(pr, (mine_p,), offer_p) if offer_p else None
    if his_offer is not None:
        r = build_row(ctx, screen, market, his_offer, base_me, base_th)
        if r is None:
            raise InvalidRequest("his offer cannot be made legal on both rosters (roster size, slots or eligibility)")
        kinds[his_offer.id], rows[his_offer.id] = HIS_OFFER, r
    his_mc = rows[his_offer.id].market_change if his_offer else 0.0

    # 2. counters near his offer
    if his_offer is not None:
        cands = enumerate_counters(pr, mine_p, offer_p, mine, theirs, cons)
        built = [(k, r) for k, o in cands if (r := build_row(ctx, screen, market, o, base_me, base_th)) is not None]
        funnel["counters_listed"] = len(cands)
        for k, r in pick_counters(built, cons.counters_per_kind):
            kinds[r.offer.id], rows[r.offer.id] = k, r

    # 3. every deal with him that includes your player
    if cons.include_full:
        deals = enumerate_deals(pr, mine_p, mine, theirs, cons)
        full_rows, st = screen_target(ctx, screen, market, deals, log)
        funnel.update({f"deals_{k}": v for k, v in st.items() if k != "waiver_bar"})
        for r in pick_shortlist(full_rows, cons.full_shortlist, 6):
            if r.offer.id not in rows:
                kinds[r.offer.id], rows[r.offer.id] = FULL, r
    funnel["simulated"] = len(rows)

    # 4. simulate everything on the selection stream
    alt, alt_v = best_alt(ev, alt_options(ev, screen, tcfg.search.fa_candidates, tcfg.search.drop_candidates), objective)
    funnel["best_waiver_only_sim"] = alt_v
    log(f"simulating {len(rows)} offers ... [{time.monotonic() - t0:.0f}s]")
    bar = tcfg.bar_for(objective)
    scored: list[tuple[str, Finalist]] = []
    for oid, row in rows.items():
        ch, v = select_choice(ev, screen, row, objective, tcfg, alt)
        ch, res = try_follow_up(ev, screen, ch, v, objective, set(row.offer.get))
        ch, res = resolve_claims(ev, screen, ch, res, objective)
        f = Finalist(ch, row, row.market_change or 0.0, float(res.g.mean()))
        f.extra["them_sel"] = float(res.them["pts"].mean())
        f.extra["tier_sel"] = tier(f.extra["them_sel"], f.market_change)
        scored.append((kinds[oid], f))

    # 5. fresh scenarios for what will be shown
    live = [(k, f) for k, f in scored if f.extra["tier_sel"] != "Unlikely" and f.g_sel > bar]
    chosen: dict[str, tuple[str, Finalist]] = {}
    for k, f in scored:
        if k == HIS_OFFER:
            chosen[f.choice.offer.id] = (k, f)
    easiest = sorted((kf for kf in live if kf[0] in COUNTER_KINDS),
                     key=lambda kf: (concession(his_mc, kf[1].market_change), -kf[1].g_sel))
    for k, f in easiest[:6]:
        chosen.setdefault(f.choice.offer.id, (k, f))
    for k, f in sorted((kf for kf in live if kf[0] == FULL), key=lambda kf: -kf[1].g_sel)[:3]:
        chosen.setdefault(f.choice.offer.id, (k, f))
    if len(chosen) < 2:  # nothing clears the bar: keep the best by gain so a near miss exists
        for k, f in sorted(scored, key=lambda kf: -kf[1].g_sel)[:2]:
            chosen.setdefault(f.choice.offer.id, (k, f))
    fin = list(chosen.values())
    log(f"re-evaluating {len(fin)} offers on {tcfg.scenarios.finalist} fresh scenarios ... [{time.monotonic() - t0:.0f}s]")
    world2 = World(ctx, tcfg.scenarios.finalist, ctx.seed + 104729, log=log)
    ev2 = Evaluator(world2, ctx, fw, frac)
    if tcfg.finalist_policy == "v2":
        attach_v2(ev2, world2)
    for _, f in fin:
        fill_finalist(f, ev2, world2, ctx, tcfg, objective, market, today)

    his = next((f for k, f in fin if k == HIS_OFFER), None)
    ranked = rank_counters(his.market_change if his else 0.0, fin)
    deals_ok = sorted((f for k, f in fin if k == FULL and f.qualifies and f.tier != "Unlikely"), key=lambda f: -f.g.mean)
    best_deal = deals_ok[0] if deals_ok else None
    shown = ([his] if his else []) + [f for _, f in ranked] + ([best_deal] if best_deal else [])
    if not shown:  # no request offer and nothing qualifies: show the nearest miss
        shown = [max((f for _, f in fin), key=lambda f: f.g.mean)]
    nm_target = nm(mine_p)
    if his is not None:
        verdict = verdict_text(his, ranked, best_deal, nm_target, objective)
    elif ranked or best_deal:
        verdict = f"BEST DEALS FOR {nm_target.upper()}: see the cards (no offer was entered)."
    else:
        verdict = f"KEEP {nm_target.upper()}: nothing with this manager clears your bar."
    funnel["clear_bar"] = sum(f.qualifies for _, f in fin)
    rec = Recommendation(ctx, objective, [f for _, f in fin], shown, funnel, market, timing, world.n, world2.n,
                         time.monotonic() - t0, no_trade=not (ranked or best_deal or (his and his.qualifies)),
                         near_miss=None)
    stamp = f"{snap.snapshot_id}|inbound|{pr}|{mine_p}|{offer_p}|{objective}|{ctx.seed}|{tcfg.model_dump_json()}"
    rec.rec_id = f"in_{snap.taken_at:%Y%m%dT%H%M%SZ}_{hashlib.sha256(stamp.encode()).hexdigest()[:6]}"
    return InboundResult(ctx, "ok", objective=objective, partner=pr, my_player=mine_p, their_offer=offer_p, rec=rec,
                         his=his, counters=ranked, best_deal=best_deal, verdict=verdict,
                         surplus=float(his.them["pts"].mean) if his else None, funnel=funnel,
                         elapsed_s=time.monotonic() - t0)


# --- text output ---------------------------------------------------------------------------------
def render_inbound(res: InboundResult) -> str:
    from trade.cards import OBJ_LABEL, _name, _team, card

    ctx, rec = res.ctx, res.rec
    head = f"{rec.rec_id if rec else 'inbound'} · objective: {OBJ_LABEL[res.objective]} · {res.elapsed_s:.0f}s"
    if res.status == "blocked":
        return f"{head}\n\n{res.message}"
    unit = UNIT[OBJ_KEY[res.objective]]
    me_nm = _name(ctx, res.my_player)
    out = [head, f"{_team(ctx, res.partner).upper()} WANTS YOUR {me_nm.upper()}" + (" · he asked first" if res.their_offer else ""), ""]
    if ctx.flags:
        out += ["flags: " + ", ".join(ctx.flags), ""]
    out += [res.verdict, ""]
    hdr = f"{'Option':<36}{'You give':<30}{'You get':<40}{'You':>9}{'Him':>8}  Tier"
    out.append(hdr)

    def line(label: str, f: Finalist) -> str:
        o = f.choice.offer
        give = ", ".join(_name(ctx, p) for p in o.give)
        get = ", ".join(_name(ctx, p) for p in o.get)
        return (f"{label:<36.35}{give:<30.29}{get:<40.39}{f.g.mean:>+9.2f}{f.them['pts'].mean:>+8.2f}  {f.tier}")

    if res.his:
        out.append(line(HIS_OFFER, res.his))
    for k, f in res.counters:
        out.append(line(k, f))
    if res.best_deal:
        out.append(line("Best deal with him (full search)", res.best_deal))
    out.append(f"{'Keep ' + me_nm:<36}{'-':<30}{'-':<40}{0:>+9.2f}{0:>+8.2f}  -")
    out.append(f"\n(your change in {unit} of the chosen objective; his change in expected FP a week)")
    if res.his and res.surplus is not None:
        if res.surplus > 0:
            out.append(f"The room to negotiate: his own offer helps him by {res.surplus:+.2f} FP a week by projection. "
                       "Counters split that surplus differently, which is the argument to make when you send one.")
        else:
            out.append(f"His own offer costs him {res.surplus:+.2f} FP a week by projection, so it rests on reputation.")
    if res.their_offer:
        out.append("His request is evidence: he asked first, so he probably values your player above the market's view. "
                   "No acceptance probability is shown (none can be calibrated yet).")
    fn = res.funnel
    out.append(f"Search: {fn.get('counters_listed', 0)} counters listed, {fn.get('deals_listed', 0):,} deals with him, "
               f"{fn['simulated']} simulated, {fn['clear_bar']} of the {len(rec.finalists)} re-run clear the bar.")
    out += ["", "=" * 78, ""]
    for i, f in enumerate(rec.cards, 1):
        out += [card(rec, f, i, len(rec.cards)), "", "-" * 78, ""]
    return "\n".join(out)

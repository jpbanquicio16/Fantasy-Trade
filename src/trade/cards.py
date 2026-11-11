"""Recommendation cards (design section 11): both sides' change, the alternative beaten, the
uncertainty, the market view and what would reverse it. Every number comes from the result object.
"""

from __future__ import annotations

from trade.evaluate import OBJ_KEY, UNIT
from trade.recommend import Finalist, Recommendation

OBJ_LABEL = {"points": "expected weekly points", "top6": "P(top 6)", "title": "P(title)"}


def _name(ctx, p: str) -> str:
    i = ctx.snap.players.get(p)
    return i.name if i else p


def _player(rec: Recommendation, p: str) -> str:
    ctx, mk = rec.ctx, rec.market
    info = ctx.snap.players.get(p)
    pos = "/".join(info.positions) if info else "?"
    fp = ctx.proj[p].linear_fp(ctx.snap.rules.scoring) if p in ctx.proj else float("nan")
    pick = mk.pick.get(p)
    drafted = f"drafted round {pick[1]} (#{pick[0]})" if pick else "not drafted"
    status = f"  [{info.injury_status}]" if info and info.injury_status else ""
    return f"{_name(ctx, p):<22} {pos:<8} proj {fp:4.1f} FP/G   {drafted}{status}"


def _team(ctx, rid: int) -> str:
    r = ctx.snap.rosters[rid]
    return f"roster {rid}" + (f" ({r.owner_name})" if r.owner_name else "")


def _est(e, unit: str, signed: bool = True) -> str:
    return f"{e.mean:+.2f} {unit} (SE {e.se:.2f})" if signed else f"{e.mean:.2f} {unit}"


def card(rec: Recommendation, f: Finalist, k: int, n: int) -> str:
    ctx, o, ch = rec.ctx, f.choice.offer, f.choice
    t = rec.timing
    L = [f"TRADE IDEA {k} of {n} · with {_team(ctx, o.partner)} · {f.tier} (heuristic, uncalibrated)", ""]
    for i, p in enumerate(o.give):
        L.append(f"{'You give' if i == 0 else '':<10}{_player(rec, p)}")
    for i, p in enumerate(o.get):
        L.append(f"{'You get' if i == 0 else '':<10}{_player(rec, p)}")
    then = []
    if ch.my.adds:
        then.append("you add " + ", ".join(_player(rec, p).split("  ")[0].strip() for p in ch.my.adds))
    if ch.my.drops:
        then.append("you drop " + ", ".join(_name(ctx, p) for p in ch.my.drops))
    if ch.their.adds:
        then.append(f"they add {', '.join(_name(ctx, p) for p in ch.their.adds)}")
    if ch.their.drops:
        then.append(f"they drop {', '.join(_name(ctx, p) for p in ch.their.drops)}")
    L.append(f"{'Then':<10}{' · '.join(then) if then else 'no further moves needed'}")
    when = f"{t['process_at']:%a %d %b %H:%M} {t['process_at'].tzname()}"
    first = f"week {t['first_week']}" + ("" if t["frac"] == 1.0 else f" ({t['frac']:.0%} of it)")
    L.append(f"{'Processes':<10}midnight Pacific after the {t['review_days']}-day review = {when}; "
             f"counts from {first} (rule: {ctx.tcfg.rules.traded_player_week_rule})")
    pend = [s for s in ctx.snap.slots.get(ctx.snap.my_roster_id, []) if s.player_id in o.give and s.state != "unlocked"
            and s.units is not None and ctx.tcfg.rules.pending_score_on_trade == "forfeited_unless_locked"]
    for s in pend:
        L.append(f"{'':<10}lock {_name(ctx, s.player_id)}'s {s.units / ctx.res:.1f} before the trade processes")
    L.append("")
    m, th = f.me, f.them
    L.append(f"{'Your change':<13}{_est(m['pts'], 'FP/wk')} · P(top 6) {_est(m['top6'], 'pts')} · "
             f"P(title) {_est(m['title'], 'pts')}")
    L.append(f"{'':<13}playoff weeks {m['playoff_pts'].mean:+.2f} FP/wk · model band (10th-90th) on "
             f"{OBJ_LABEL[rec.objective]}: {f.g.p10:+.2f} to {f.g.p90:+.2f}")
    L.append(f"{'Their change':<13}{th['pts'].mean:+.2f} FP/wk · P(top 6) {th['top6'].mean:+.1f} pts · "
             f"P(title) {th['title'].mean:+.1f} pts")
    unit = UNIT[OBJ_KEY[rec.objective]]
    alt = ("hold" if ch.alt is None else "best waiver move (" + ", ".join(
        [f"drop {_name(ctx, d)}" for d in ch.alt.drops] + [f"add {_name(ctx, a)}" for a in ch.alt.adds]) + ")")
    L.append(f"{'Beats':<13}{alt}; gain over it {_est(f.g, unit)}")
    ngive, nget = len(o.give), len(o.get)
    L.append(f"{'Market view':<13}partner's market-perceived change {f.market_change:+.1f} FP/G-equivalent "
             f"({ngive} player(s) to them, {nget} to you)")
    L.append("")
    starts = f.extra.get("starts", {})
    inc = "; ".join(f"{_name(ctx, p)} starts {starts.get(p, 0):.0%} of weeks" for p in o.get)
    why = inc + (f"; bench cover adds {f.extra['cover_value']:.1f} FP/wk" if f.extra.get("cover_value") else "")
    L.append(f"{'Why it works':<13}{why}")
    own = f.extra.get("own_effect", {})
    if own:
        L.append(f"{'':<13}your P(top 6) {own['top6']:+.1f} pts from your own roster alone vs "
                 f"{m['top6'].mean:+.1f} with the rival weakened/strengthened")
    fails = []
    if f.them["pts"].mean < -ctx.tcfg.search.partner_loss_tolerance_fp:
        fails.append(f"they lose {-f.them['pts'].mean:.1f} FP/wk by projection")
    if "if_add_lost" in f.extra:
        fails.append(f"if the free agent is claimed first the gain is {f.extra['if_add_lost']:+.2f}")
    for p in o.get:
        i = ctx.snap.players.get(p)
        if i and i.injury_status:
            fails.append(f"{_name(ctx, p)} is listed {i.injury_status}")
    for fl in f.extra.get("flags", []):
        fails.append(fl.replace("_", " "))
    L.append(f"{'Could fail':<13}{' · '.join(fails) if fails else 'no specific risk flagged'}")
    rv = []
    fp_ = f.sens.get("flip_pct")
    rv.append("incoming players' output would have to fall by " + (f"{fp_:.0f}%" if fp_ is not None else ">60%")
              + " to erase the gain")
    rr = f.sens["rule_reading"]
    rv.append(f"under '{rr['rule']}' the gain is {rr['g']:+.2f}")
    rv.append(f"if the other ten teams manage worse: {f.sens['weak_others']:+.2f}")
    L.append(f"{'Reverses if':<13}{' · '.join(rv)}")
    fb = fallback(rec, f)
    L.append(f"{'Fallback ask':<13}{fb}")
    return "\n".join(L)


def fallback(rec: Recommendation, f: Finalist) -> str:
    """A version slightly less favourable to you, from the same search, that still clears the bar."""
    o = f.choice.offer
    pool = [g for g in rec.finalists if g is not f and g.qualifies and g.choice.offer.partner == o.partner
            and set(g.choice.offer.get) == set(o.get) and g.them["pts"].mean > f.them["pts"].mean]
    if not pool:
        return "none in the search clears the bar with a better deal for them"
    g = max(pool, key=lambda x: x.g.mean)
    return (f"give {', '.join(_name(rec.ctx, p) for p in g.choice.offer.give)} instead "
            f"(your gain {g.g.mean:+.2f}, theirs {g.them['pts'].mean:+.2f} FP/wk)")


def render(rec: Recommendation) -> str:
    ctx = rec.ctx
    out = [f"{rec.rec_id} · objective: {OBJ_LABEL[rec.objective]} · {rec.n_sel} scenarios to select, "
           f"{rec.n_fresh} fresh to report · {rec.elapsed_s:.0f}s", ""]
    if ctx.flags:
        out += ["flags: " + ", ".join(ctx.flags), ""]
    if rec.funnel.get("blocked"):
        return "\n".join(out + [f"No recommendation: {rec.funnel['blocked']}."])
    fn = rec.funnel
    out.append(f"search: {fn['candidates']:,} candidate offers -> {fn['legal']:,} legal -> {fn['my_gain_positive']:,} "
               f"you gain (screen) -> {fn['beats_waiver']:,} beat your best waiver move -> {fn['partner_ok']:,} partner "
               f"tolerates -> {fn['dominated']:,} dominated -> {fn['shortlist']} simulated")
    out.append("")
    if rec.no_trade:
        out.append("NO WORTHWHILE TRADE. Nothing clears the bar (gain above the threshold, standard error under a "
                   "third of it, 10th percentile above zero).")
        nm = rec.near_miss
        if nm:
            out += ["", "Closest near-miss:", card(rec, nm, 1, 1), "", f"Why it misses: {nm.why_not}.",
                    "What would change the answer: a return date moving, a free agent you can win, or a partner's new need."]
        return "\n".join(out)
    for i, f in enumerate(rec.cards, 1):
        out += [card(rec, f, i, len(rec.cards)), "", "-" * 78, ""]
    return "\n".join(out)

"""``trade`` command line.

    trade settings                 T0: league facts read from Sleeper, each marked read / open
    trade recommend [options]      whole-league trade search -> cards, or "no worthwhile trade"
    trade nightly                  refresh, search and store the cards (scheduled)
    trade app                      Streamlit cards (needs the 'app' extra)
    trade target PLAYER [options]  offers for one player you name: get him, or move him (price list + cards)
    trade inbound MANAGER PLAYER   a manager asked for your player: judge his offer, counters, best deals
    trade propose REC OFFER        log that you sent an offer (the only acceptance labels that can exist)
    trade respond PROPOSAL OUTCOME log what happened to it
    trade log                      proposals, responses and observed acceptance per tier
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trade import config as C
from trade import store


def cmd_settings(a) -> int:
    from waiver.config import Paths, load_league
    from waiver.ingest.sleeper import SleeperClient
    from waiver.state.league import LeagueRules

    from trade.format import build_format

    paths, league = Paths().ensure(), load_league()
    client = SleeperClient(paths=paths)
    lg = client.league(league.league_id, max_age_s=0)
    rules = LeagueRules.from_league(lg, league)
    state = client.state(max_age_s=0)
    fmt = build_format(client, league.league_id, int(state.week))
    s = lg.settings
    rows = [
        ("teams", rules.total_rosters, "read"),
        ("starting slots", " ".join(rules.starting_slots), "read"),
        ("bench / IL slots", f"{rules.bench_slots} / {rules.reserve_slots}", "read"),
        ("IL statuses", ", ".join(rules.il_statuses), "read"),
        ("trade deadline", f"week {fmt.trade_deadline_week}", "read (settings.trade_deadline)"),
        ("trade review", f"{fmt.review_days} days, processes midnight Pacific after", "read"),
        ("playoff teams / start", f"{fmt.playoff_teams} from week {s.get('playoff_week_start')}", "read"),
        ("round weeks", str(fmt.round_weeks), f"read (playoff_round_type={s.get('playoff_round_type')})"),
        ("bracket", "; ".join(f"R{m.r} m{m.m}: {m.t1[0]}{m.t1[1]} v {m.t2[0]}{m.t2[1]}" for m in fmt.bracket),
         fmt.bracket_source),
        ("seeding", "record, points for, points against; exact ties by team order", "read (playoff_seed_type=0)"),
        ("pick trading", "enabled in the league; picks are excluded from this tool", "read"),
        ("waivers", rules.waiver_type, "read"),
        ("weeks scored so far", fmt.first_unscored_week - 1, "read (last_scored_leg)"),
        ("mid-week trade rule", "can a player who already played be used before next week?", "OPEN: observe one trade"),
        ("outgoing player's open score", "can the old team still lock it after processing?", "OPEN"),
        ("reseeding / playoff ties", "no reseeding; higher seed wins a tie", "ASSUMED"),
    ]
    for k, v, st in rows:
        print(f"{k:<28}{str(v):<70}{st}")
    if fmt.flags:
        print("flags:", ", ".join(fmt.flags))
    return 0


def cmd_recommend(a) -> int:
    from trade.cards import render
    from trade.recommend import recommend

    cfg = C.load()
    if a.scenarios:
        cfg.scenarios.shortlist = a.scenarios
    if a.finalist_scenarios:
        cfg.scenarios.finalist = a.finalist_scenarios
    if a.policy:
        cfg.finalist_policy = a.policy
    if a.rule:
        cfg.rules.traded_player_week_rule = a.rule
    if a.shapes:
        cfg.search.shapes = a.shapes.split(",")
    rec = recommend(cfg, a.objective, a.partner, refresh=not a.no_refresh)
    store.save_recommendation(rec, cfg)
    print("\n" + render(rec))
    if a.json:
        print(json.dumps([store.offer_json(rec, f) for f in rec.cards], indent=1, default=str))
    return 0


def cmd_nightly(a) -> int:
    """Refresh, search, store and write the text report; the Streamlit app reads what this stores."""
    from trade.cards import render
    from trade.config import Paths
    from trade.recommend import recommend

    cfg = C.load()
    rec = recommend(cfg, None, None, refresh=True)
    store.save_recommendation(rec, cfg)
    text = render(rec)
    (Paths().ensure().reports / "latest.txt").write_text(text)
    print(f"{rec.rec_id}: {'no worthwhile trade' if rec.no_trade else str(len(rec.cards)) + ' card(s)'}")
    return 0


def cmd_app(a) -> int:
    import subprocess

    app = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def cmd_target(a) -> int:
    from trade.cards import render  # noqa: F401
    from trade.context import build_context
    from trade.targeted import (AmbiguousPlayer, PlayerNotFound, TargetConstraints, find_players,
                                recommend_for_player, render_targeted, resolve_player)

    cfg = C.load()
    if a.scenarios:
        cfg.scenarios.shortlist = a.scenarios
    if a.finalist_scenarios:
        cfg.scenarios.finalist = a.finalist_scenarios
    if a.policy:
        cfg.finalist_policy = a.policy
    if a.rule:
        cfg.rules.traded_player_week_rule = a.rule
    ctx = build_context(cfg, not a.no_refresh)
    names = {p: i.name for p, i in ctx.snap.players.items()}
    protect = set()
    for q in a.protect or []:
        hits = find_players(names, q)
        if len(hits) != 1:
            print(f"--protect {q!r}: {'no match' if not hits else 'ambiguous'}", file=sys.stderr)
            return 2
        protect.add(hits[0])
    cons = TargetConstraints(protect=protect, max_send=a.max_send,
                             shapes=a.shapes.split(",") if a.shapes else None,
                             teams={int(t) for t in a.teams.split(",")} if a.teams else None)
    try:
        pid = resolve_player(ctx, a.player)
        res = recommend_for_player(cfg, pid, cons, a.objective, ctx=ctx)
    except PlayerNotFound as e:
        print(e, file=sys.stderr)
        return 2
    except AmbiguousPlayer as e:
        print(f"{e}; rerun with a player id:", file=sys.stderr)
        for p in e.matches:
            i = ctx.snap.players[p]
            print(f"  {p:<10}{i.name:<26}{i.team or '-':<5}{'/'.join(i.positions)}", file=sys.stderr)
        return 2
    if res.rec is not None:
        store.save_recommendation(res.rec, cfg)
    print("\n" + render_targeted(res))
    if a.json and res.rec is not None:
        print(json.dumps([store.offer_json(res.rec, p.f) for p in res.price_list], indent=1, default=str))
    return 0


def cmd_inbound(a) -> int:
    from trade.context import build_context
    from trade.interest import InterestConstraints, InvalidRequest, render_inbound, respond_to_interest
    from trade.targeted import AmbiguousPlayer, PlayerNotFound, find_players

    cfg = C.load()
    if a.scenarios:
        cfg.scenarios.shortlist = a.scenarios
    if a.finalist_scenarios:
        cfg.scenarios.finalist = a.finalist_scenarios
    if a.policy:
        cfg.finalist_policy = a.policy
    if a.rule:
        cfg.rules.traded_player_week_rule = a.rule
    ctx = build_context(cfg, not a.no_refresh)
    names = {p: i.name for p, i in ctx.snap.players.items()}
    protect = set()
    for q in a.protect or []:
        hits = find_players(names, q)
        if len(hits) != 1:
            print(f"--protect {q!r}: {'no match' if not hits else 'ambiguous'}", file=sys.stderr)
            return 2
        protect.add(hits[0])
    cons = InterestConstraints(protect=protect, max_extra=a.max_extra, include_full=not a.counters_only)
    try:
        res = respond_to_interest(cfg, a.manager, a.player, a.offer, cons, a.objective, ctx=ctx)
    except (PlayerNotFound, InvalidRequest) as e:
        print(e, file=sys.stderr)
        return 2
    except AmbiguousPlayer as e:
        print(f"{e}; rerun with a player id:", file=sys.stderr)
        for p in e.matches:
            i = ctx.snap.players[p]
            print(f"  {p:<10}{i.name:<26}{i.team or '-':<5}{'/'.join(i.positions)}", file=sys.stderr)
        return 2
    if res.rec is not None:
        store.save_recommendation(res.rec, cfg)
    print("\n" + render_inbound(res))
    if res.his is not None and res.rec is not None and not a.no_log:
        pid = store.log_inbound(res.rec.rec_id, res.partner, ctx.snap.my_roster_id, store.offer_json(res.rec, res.his), a.note or "")
        print(f"logged his request as proposal {pid}; record your answer with: trade respond {pid} accepted|rejected|countered")
    if a.json and res.rec is not None:
        print(json.dumps([store.offer_json(res.rec, f) for f in res.rec.cards], indent=1, default=str))
    return 0


def cmd_propose(a) -> int:
    pid = store.log_proposal(a.rec, a.offer, a.note or "")
    print(f"logged proposal {pid}")
    return 0


def cmd_respond(a) -> int:
    store.log_response(a.proposal, a.outcome, a.note or "", a.counter)
    print("logged")
    return 0


def cmd_log(a) -> int:
    with store.connect() as db:
        for r in db.execute("SELECT p.proposal_id, p.sent_at, p.give, p.get, p.tier_shown, r.outcome "
                            "FROM proposals p LEFT JOIN responses r USING(proposal_id) ORDER BY 1"):
            print(dict(r))
    print("observed acceptance by tier:", store.tier_report())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trade", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("settings").set_defaults(fn=cmd_settings)
    r = sub.add_parser("recommend")
    r.add_argument("--objective", choices=["points", "top6", "title"])
    r.add_argument("--partner", type=int, help="only offers with this roster id")
    r.add_argument("--scenarios", type=int)
    r.add_argument("--finalist-scenarios", type=int)
    r.add_argument("--policy", choices=["shortlist", "v2"], help="finalist policy for the two traded teams")
    r.add_argument("--rule", choices=["next_week_if_played", "immediate"])
    r.add_argument("--shapes", help="comma list, e.g. 1-1,2-1")
    r.add_argument("--no-refresh", action="store_true", help="skip fetching stats and schedule first")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_recommend)
    sub.add_parser("nightly", help="refresh, search and store (schedule with scripts/nightly.sh)").set_defaults(fn=cmd_nightly)
    sub.add_parser("app", help="open the cards in Streamlit").set_defaults(fn=cmd_app)
    t = sub.add_parser("target", help="offers for one player you name")
    t.add_argument("player", help="name (accents, suffixes and punctuation ignored) or Sleeper player id")
    t.add_argument("--objective", choices=["points", "top6", "title"])
    t.add_argument("--protect", nargs="*", help="players you will not give up")
    t.add_argument("--max-send", type=int, default=3, help="most players you send (default 3)")
    t.add_argument("--shapes", help="give-get shapes, e.g. 1-1,2-1,3-1; acquire default 1-1,2-1,3-1,1-2,2-2,3-2")
    t.add_argument("--teams", help="move mode: comma list of roster ids to consider")
    t.add_argument("--scenarios", type=int)
    t.add_argument("--finalist-scenarios", type=int)
    t.add_argument("--policy", choices=["shortlist", "v2"])
    t.add_argument("--rule", choices=["next_week_if_played", "immediate"])
    t.add_argument("--no-refresh", action="store_true")
    t.add_argument("--json", action="store_true")
    t.set_defaults(fn=cmd_target)
    i = sub.add_parser("inbound", help="a manager asked for one of your players")
    i.add_argument("manager", help="roster id, or part of his display name")
    i.add_argument("player", help="your player he wants (name or Sleeper id)")
    i.add_argument("--offer", nargs="*", help="the players he offered (names or ids); omit if he made no offer")
    i.add_argument("--objective", choices=["points", "top6", "title"])
    i.add_argument("--protect", nargs="*", help="your players you will not add to a counter")
    i.add_argument("--max-extra", type=int, default=1, help="most extra players of yours in a counter or deal (default 1)")
    i.add_argument("--counters-only", action="store_true", help="skip the full search with him")
    i.add_argument("--note")
    i.add_argument("--no-log", action="store_true", help="do not log his request in proposals")
    i.add_argument("--scenarios", type=int)
    i.add_argument("--finalist-scenarios", type=int)
    i.add_argument("--policy", choices=["shortlist", "v2"])
    i.add_argument("--rule", choices=["next_week_if_played", "immediate"])
    i.add_argument("--no-refresh", action="store_true")
    i.add_argument("--json", action="store_true")
    i.set_defaults(fn=cmd_inbound)
    q = sub.add_parser("propose")
    q.add_argument("rec")
    q.add_argument("offer")
    q.add_argument("--note")
    q.set_defaults(fn=cmd_propose)
    s = sub.add_parser("respond")
    s.add_argument("proposal", type=int)
    s.add_argument("outcome", choices=["accepted", "rejected", "countered", "expired", "withdrawn"])
    s.add_argument("--note")
    s.add_argument("--counter")
    s.set_defaults(fn=cmd_respond)
    sub.add_parser("log").set_defaults(fn=cmd_log)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

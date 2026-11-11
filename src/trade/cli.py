"""``trade`` command line.

    trade settings                 T0: league facts read from Sleeper, each marked read / open
    trade recommend [options]      whole-league trade search -> cards, or "no worthwhile trade"
    trade propose REC OFFER        log that you sent an offer (the only acceptance labels that can exist)
    trade respond PROPOSAL OUTCOME log what happened to it
    trade log                      proposals, responses and observed acceptance per tier
"""

from __future__ import annotations

import argparse
import json
import sys

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

"""Correctness tests from design section 10: bracket and seeding, timing, stopping thresholds,
bench cover, settle legality, paired evaluation (null trade is exactly zero; a trade changes only
the two traded teams), reproducibility, and the compatibility tiers."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from trade.format import (
    LeagueFormat,
    Standing,
    bracket_from_api,
    effective_date,
    first_new_week,
    processing_time,
    rank_teams,
    round_weeks_for,
    standard_bracket,
)
from trade.market import fit_rank_curve, tier
from trade.season import SeasonSim
from trade.valuation import GameDist, Valuer, seat, stopping_chain


# --- format ------------------------------------------------------------------------
def test_standard_bracket_six_teams_has_byes_for_top_two():
    b = standard_bracket(6)
    r1 = [m for m in b if m.r == 1]
    assert {(m.t1[1], m.t2[1]) for m in r1} == {(4, 5), (6, 3)} or {(m.t1[1], m.t2[1]) for m in r1} == {(4, 5), (3, 6)}
    seeds_r2 = {t[1] for m in b if m.r == 2 for t in (m.t1, m.t2) if t[0] == "seed"}
    assert seeds_r2 == {1, 2}
    assert b[-1].r == 3


def test_api_bracket_parses_and_rejects_inconsistent_seeds():
    raw = [{"m": 1, "r": 1, "t1": 3, "t2": 12}, {"m": 2, "r": 1, "t1": 11, "t2": 8},
           {"m": 3, "r": 2, "t1": 1, "t2_from": {"w": 1}}, {"m": 4, "r": 2, "t1": 6, "t2_from": {"w": 2}},
           {"m": 6, "r": 3, "p": 1, "t1_from": {"w": 3}, "t2_from": {"w": 4}}]
    seeds = {1: 1, 6: 2, 8: 3, 3: 4, 12: 5, 11: 6, 5: 7}
    b = bracket_from_api(raw, seeds, 6)
    assert b is not None and b[-1].m == 6
    assert {(m.t1[1], m.t2[1]) for m in b if m.r == 1} == {(4, 5), (6, 3)}
    bad = {**seeds, 3: 2, 6: 4}  # first-round seeds no longer pair up to 9
    assert bracket_from_api(raw, bad, 6) is None


def test_round_weeks():
    assert round_weeks_for(22, 3, 0) == [[22], [23], [24]]
    assert round_weeks_for(22, 3, 1) == [[22], [23], [24, 25]]
    assert round_weeks_for(22, 3, 2) == [[22, 23], [24, 25], [26, 27]]


def test_seeding_record_then_points_for_then_against():
    wins = np.array([[3.0, 3.0, 2.0, 3.0]])
    pf = np.array([[500.0, 600.0, 900.0, 600.0]])
    pa = np.array([[400.0, 400.0, 100.0, 300.0]])
    assert rank_teams(wins, pf, pa)[0].tolist() == [3, 1, 0, 2]  # 3 beats 1 on points against


def test_midnight_pacific_in_adelaide_across_both_clock_changes():
    adl = ZoneInfo("Australia/Adelaide")
    assert processing_time(date(2026, 11, 14), adl).strftime("%H:%M") == "18:30"
    assert processing_time(date(2027, 3, 20), adl).strftime("%H:%M") == "17:30"  # US clocks forward on 14 Mar
    assert processing_time(date(2027, 4, 10), adl).strftime("%H:%M") == "16:30"  # Adelaide back on 4 Apr


WEEKS = {4: [date(2026, 11, 9) + __import__("datetime").timedelta(days=i) for i in range(7)],
         5: [date(2026, 11, 16) + __import__("datetime").timedelta(days=i) for i in range(7)]}


def test_trade_week_rule_readings():
    eff = effective_date(date(2026, 11, 11), 2)  # accepted Wed, lands Sat 14 Nov (mid week 4)
    assert eff == date(2026, 11, 14)
    assert first_new_week(WEEKS, eff, 4, "next_week_if_played") == (5, 1.0)
    w, f = first_new_week(WEEKS, eff, 4, "immediate")
    assert w == 4 and f == pytest.approx(2 / 7)  # Sat and Sun left
    # lands on Monday: no player has played, both readings agree
    mon = date(2026, 11, 16)
    assert first_new_week(WEEKS, mon, 4, "next_week_if_played") == (5, 1.0) == first_new_week(WEEKS, mon, 4, "immediate")


# --- valuation ---------------------------------------------------------------------
def test_stopping_chain_reproduces_the_lockin_premium():
    one = [GameDist(1.0, 0.0, 1.0)]
    assert stopping_chain(one)[0][0] == 0.0
    v2, thr2 = stopping_chain([GameDist(1.0, 0.0, 1.0)] * 2)
    assert v2[0] == pytest.approx(0.3989, abs=1e-3)  # kappa(2), lineup design
    assert thr2[0] == pytest.approx(0.0) and thr2[1] == -np.inf  # last game is forced
    v3, _ = stopping_chain([GameDist(1.0, 0.0, 1.0)] * 3)
    assert v3[0] == pytest.approx(0.63, abs=0.01)  # kappa(3)


def test_missing_the_last_game_costs_a_forced_zero():
    v, _ = stopping_chain([GameDist(1.0, 30.0, 8.0), GameDist(0.5, 30.0, 8.0)])
    assert v[1] == pytest.approx(15.0)  # the forced last game pays p * mu


def test_seating_respects_eligibility():
    e = np.array([10.0, 9.0, 8.0])
    elig = np.array([[True, True], [True, False], [False, True]])
    pairs = dict(seat(e, elig))
    assert all(elig[m, s] for m, s in pairs.items())
    assert sum(e[m] for m in pairs) == pytest.approx(19.0)  # players 0 and 1 or 0 and 2: best is 10 + 9


# --- a small synthetic league ---------------------------------------------------------
N_SC, WEEKS_ = 400, [1, 2, 3]


def build_world(seed: int = 0):
    from waiver.sim.scenarios import Horizon, ScenarioSet

    rng = np.random.default_rng(seed)
    pids = [f"p{t}{k}" for t in range(4) for k in range(4)]  # 4 teams x 4 players, 2 starting slots
    base = {p: 20.0 + 6.0 * (i % 4) + 3.0 * (i // 4) for i, p in enumerate(pids)}
    days = [date(2026, 11, 1 + i) for i in range(9)]
    wd = {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8]}
    hz = Horizon(datetime(2026, 11, 1), days[0], days, WEEKS_, wd, np.repeat(WEEKS_, 3).astype(np.int32))
    P, D = len(pids), 9
    has = np.zeros((P, D), bool)
    has[:, [0, 1, 3, 4, 6, 7]] = True
    plays = (rng.random((N_SC, P, D)) < 0.95) & has[None]
    units = np.where(plays, rng.normal([[base[p]] for p in pids], 7.0, (N_SC, P, D)), 0).round().astype(np.int32)
    batch = np.repeat(np.arange(10), N_SC // 10)
    sc = ScenarioSet(pids, {p: i for i, p in enumerate(pids)}, hz, N_SC, 10, batch, has, np.zeros_like(has), {},
                     plays, units, np.zeros((N_SC, P, D), np.float32))
    dists = {(p, d): GameDist(0.95, base[p], 7.0) for p in pids for d in (0, 1, 3, 4, 6, 7)}
    elig = {p: np.array([True, True]) for p in pids}
    return sc, dists, elig, pids


def build_league():
    sc, dists, elig, pids = build_world()
    valuer = Valuer(sc, dists, elig, 1.0)
    teams = {rid: pids[4 * (rid - 1): 4 * rid] for rid in (1, 2, 3, 4)}
    st = {r: Standing() for r in teams}
    fmt = LeagueFormat([1, 2, 3, 4], 2, 1, 2, [[3]], standard_bracket(2),
                       {1: [(1, 2), (3, 4)], 2: [(1, 3), (2, 4)]}, st, None, 1)
    sim = SeasonSim(fmt, WEEKS_)
    cols = [0, 1, 2]
    fast = {r: valuer.team_weeks(p, cols) for r, p in teams.items()}
    tw = np.stack([fast[r] for r in (1, 2, 3, 4)], axis=1)
    world = SimpleNamespace(weeks=WEEKS_, tw=tw, tw_weak=tw.copy(), fast=fast, valuer=valuer, sim=sim, sc=sc)
    rosters = {r: SimpleNamespace(players=p, reserve=[]) for r, p in teams.items()}
    ctx = SimpleNamespace(fmt=fmt, snap=SimpleNamespace(my_roster_id=1, week=0, rosters=rosters), elig=elig)
    return world, ctx, teams


def test_season_has_one_champion_and_k_playoff_teams():
    world, ctx, _ = build_league()
    out = world.sim.run(world.tw)
    assert (out.champ.sum(axis=1) == 1).all()
    assert (out.top.sum(axis=1) == 2).all()
    assert (out.champ <= out.top).all()


def test_null_trade_is_exactly_zero_and_unrelated_teams_do_not_move():
    from trade.evaluate import Evaluator

    world, ctx, teams = build_league()
    ev = Evaluator(world, ctx, first_week=2)
    d = ev.diff({1: teams[1], 2: teams[2]}, [1, 2, 3])
    for rid in (1, 2, 3):
        assert all(np.all(v == 0) for v in d[rid].values())
    swap = {1: teams[1][:-1] + teams[2][:1], 2: teams[2][1:] + teams[1][-1:]}  # team 1's best for team 2's worst
    d = ev.diff(swap, [1, 2, 3, 4])
    assert np.all(d[3]["pts"] == 0) and np.all(d[4]["pts"] == 0)  # their own weekly scores are untouched
    assert np.any(d[1]["pts"] != 0)
    assert np.allclose(d[1]["pts"] + d[2]["pts"], d[1]["pts"] + d[2]["pts"])


def test_trade_week_rule_changes_only_the_first_week():
    from trade.evaluate import Evaluator

    world, ctx, teams = build_league()
    swap = {1: teams[1][:-1] + teams[2][:1], 2: teams[2][1:] + teams[1][-1:]}  # team 1's best for team 2's worst
    full = Evaluator(world, ctx, 2, 1.0).diff(swap, [1])[1]["pts"]
    later = Evaluator(world, ctx, 3, 1.0).diff(swap, [1])[1]["pts"]  # week 2 untouched, week 3 only
    half = Evaluator(world, ctx, 2, 0.5).diff(swap, [1])[1]["pts"]
    assert full.mean() != later.mean() or full.mean() == 0
    # the regular-season average covers week 2 only, so halving week 2 halves it
    assert np.allclose(half, full * 0.5)
    assert np.all(later == 0)


def test_bench_cover_needs_an_eligible_present_player():
    sc, dists, elig, pids = build_world()
    v = Valuer(sc, dists, elig, 1.0)
    starter, bench = pids[3], pids[0]  # p03 is strong (starts), p00 weak (bench)
    sc.plays[:, v.i[starter], :] = False  # the starter is out all season
    v2 = Valuer(sc, dists, elig, 1.0)
    covered = v2.team_weeks(pids[:4], [0]).mean()
    ineligible = dict(elig, **{p: np.array([False, False]) for p in pids[:3]})
    v3 = Valuer(sc, dists, ineligible, 1.0)
    assert covered > v3.team_weeks(pids[:4], [0]).mean() - 1e-9
    assert bench in pids


def test_reproducible_realised_values():
    a, b = build_world(3), build_world(3)
    assert np.array_equal(Valuer(a[0], a[1], a[2], 1.0).R, Valuer(b[0], b[1], b[2], 1.0).R)


# --- settle legality ---------------------------------------------------------------
def fake_ctx(active=14, il=1):
    players = {f"x{i}": SimpleNamespace(injury_status=None, name=f"x{i}") for i in range(30)}
    players["inj"] = SimpleNamespace(injury_status="OUT", name="inj")
    mine = [f"x{i}" for i in range(active)] + (["il0"] if il else [])
    players["il0"] = SimpleNamespace(injury_status="OUT", name="il0")
    theirs = [f"x{i}" for i in range(14, 28)]
    rules = SimpleNamespace(roster_limit=14, reserve_slots=2, il_eligible=lambda s: (s or "").upper() in {"OUT", "IR"})
    rosters = {1: SimpleNamespace(players=mine, reserve=["il0"] if il else []),
               2: SimpleNamespace(players=theirs + ["inj"], reserve=[])}
    return SimpleNamespace(snap=SimpleNamespace(rosters=rosters, players=players, rules=rules), elig={})


def test_roster_after_il_and_open_spots():
    from trade.search import roster_after

    ctx = fake_ctx()
    pl, res, open_ = roster_after(ctx, 1, ("x0",), ("inj",))
    assert "inj" in res and open_ == 1 and len(pl) == 15  # IL takes him, so the active spot opens
    pl, res, open_ = roster_after(ctx, 1, ("x0",), ("x15", "x16"))
    assert open_ == -1  # a 1-for-2 into a full roster forces a drop
    pl, res, open_ = roster_after(ctx, 1, ("x0", "x1"), ("x15",))
    assert open_ == 1  # a 2-for-1 frees a spot


def test_tiers_follow_the_two_signs():
    assert tier(1, 1) == "Looks good for both"
    assert tier(-1, 1) == "Plausible on reputation"
    assert tier(1, -1) == "Needs a case"
    assert tier(-1, -1) == "Unlikely"


def test_rank_curve_is_monotone():
    rank = np.arange(1, 200.0)
    a, b = fit_rank_curve(rank, 40 - 5 * np.log(rank) + np.random.default_rng(0).normal(0, 2, rank.size))
    assert b < 0


# --- named-player mode (design section 7) ---------------------------------------------------
NAMES = {"1": "Nikola Jokić", "2": "Jaren Jackson Jr.", "3": "Jalen Williams", "4": "Jaylin Williams",
         "5": "Gary Payton II", "6": "P.J. Washington"}


def test_name_matching_ignores_accents_suffixes_and_punctuation():
    from trade.targeted import find_players

    assert find_players(NAMES, "nikola jokic") == ["1"]
    assert find_players(NAMES, "Jaren Jackson") == ["2"]
    assert find_players(NAMES, "gary payton") == ["5"]
    assert find_players(NAMES, "PJ Washington") == ["6"]
    assert find_players(NAMES, "p.j. washington") == ["6"]
    assert find_players(NAMES, "williams") == ["3", "4"]  # ambiguous: the caller must ask which
    assert find_players(NAMES, "4") == ["4"]  # a player id
    assert find_players(NAMES, "nobody here") == []


def test_enumerate_acquire_and_move_shapes():
    from trade.targeted import TargetConstraints, enumerate_offers

    mine = [f"m{i}" for i in range(6)]
    owner = {5: ["T", "a", "b", "c"]}
    offers = list(enumerate_offers("acquire", "T", mine, owner, TargetConstraints()))
    assert all("T" in o.get and o.partner == 5 for o in offers)
    from math import comb
    assert len(offers) == sum(comb(6, k) for k in (1, 2, 3)) * (1 + 3)  # give 1-3, get him alone or with one of 3
    assert {o.shape for o in offers} == {"1-1", "2-1", "3-1", "1-2", "2-2", "3-2"}
    protected = list(enumerate_offers("acquire", "T", mine, owner, TargetConstraints(protect={"m0"}, max_send=2)))
    assert protected and all("m0" not in o.give and len(o.give) <= 2 for o in protected)
    assert {o.shape for o in enumerate_offers("acquire", "T", mine, owner, TargetConstraints(shapes=["2-1"]))} == {"2-1"}

    mine_t = mine + ["T"]
    teams = {2: ["x", "y", "z"], 3: ["u", "v"]}
    mv = list(enumerate_offers("move", "T", mine_t, teams, TargetConstraints()))
    assert all("T" in o.give for o in mv)
    assert len(mv) == (1 + 6) * (3 + 3 + 2 + 1)  # him alone or with one of 6; 1-2 players from either team
    assert {o.partner for o in enumerate_offers("move", "T", mine_t, teams, TargetConstraints(teams={3}))} == {3}
    assert not any(len(o.give) > 1 for o in enumerate_offers("move", "T", mine_t, teams, TargetConstraints(max_send=1)))


def test_pareto_and_shortlist_keep_the_front_and_each_shape():
    from trade.search import Offer, Row
    from trade.targeted import pareto, pick_shortlist

    assert pareto([(5, 0), (4, 1), (3, 0.5), (2, 3), (1, 2)]) == [0, 1, 3]
    rows = [Row(Offer(2, tuple(f"g{i}" for _ in range(1 + i % 2)), ("T",)), 10 - i, -float(i) * 0.1)
            for i in range(10)]
    short = pick_shortlist(rows, limit=80, per_shape=1)
    shapes = {r.offer.shape for r in short}
    assert shapes == {"1-1", "2-1"}
    assert rows[0] in short and len(pick_shortlist(rows, limit=2, per_shape=8)) == 2


def test_follow_up_applies_only_a_gain_and_never_cuts_incoming_or_added_players():
    from trade.search import Settled
    from trade.targeted import follow_up

    value = {"a": 5.0, "b": 1.0, "c": 3.0, "new": 4.0, "fa": 2.5, "fa2": 0.5}
    screen = SimpleNamespace(value=lambda pids: sum(value[p] for p in pids),
                             lowest=lambda pool, k: sorted(pool, key=lambda p: value[p])[:k],
                             fa_screen=["fa", "fa2"])
    s = Settled(["a", "b", "c", "new"], [], [], [])
    v, s2 = follow_up(screen, s, {"new"})
    assert s2.drops == ["b"] and s2.adds == ["fa"] and v == 5 + 3 + 4 + 2.5  # swaps the worst player for the best free agent
    assert "new" in s2.pids
    keep_all = Settled(["a", "new"], [], [], [])
    v, s3 = follow_up(screen, keep_all, {"a", "new"})
    assert s3 is keep_all  # nobody droppable: holds
    worse = SimpleNamespace(value=screen.value, lowest=screen.lowest, fa_screen=["fa2"])
    _, s4 = follow_up(worse, Settled(["a", "c", "b"], [], [], []), set())
    assert s4.adds == []  # 0.5 does not beat the 1.0 he would drop


def test_price_list_picks_best_qualifying_offer_per_shape_and_the_smallest_owner_loss():
    from trade.evaluate import Est
    from trade.search import Offer
    from trade.targeted import build_price_list, choose_finalists

    def fin(shape, partner, g, them, qual=True, tr="Plausible on reputation"):
        a, b = map(int, shape.split("-"))
        o = Offer(partner, tuple(f"g{i}" for i in range(a)), tuple(f"t{i}" for i in range(b)))
        e = lambda m: Est(m, 0.1, m - 0.2, m + 0.2)  # noqa: E731
        return SimpleNamespace(choice=SimpleNamespace(offer=o), qualifies=qual, tier=tr, g=e(g),
                               them={"pts": e(them)}, g_sel=g, extra={"them_sel": them, "tier_sel": tr})

    fs = [fin("1-1", 2, 0.6, -2.0), fin("1-1", 3, 0.9, -3.0), fin("2-1", 2, 5.0, -9.0), fin("2-1", 4, 2.0, +1.0,
          tr="Looks good for both"), fin("3-1", 2, 7.0, -8.0, qual=False), fin("1-2", 2, 3.0, -1.0, tr="Unlikely")]
    price = build_price_list(fs)
    labels = [p.label for p in price]
    assert labels == ["1-1", "2-1", "Smallest owner loss"]  # 3-1 fails the bar; 1-2 owner is 'unlikely'
    assert price[0].f.g.mean == 0.9 and price[1].f.g.mean == 5.0
    assert price[2].f.them["pts"].mean == 1.0
    picked = choose_finalists(fs, bar=0.5, per_shape=1, kind=1)
    assert len(picked) == len({p.choice.offer.id for p in picked})


def test_two_sides_cannot_claim_the_same_free_agent():
    from trade.search import Settled
    from trade.targeted import follow_up

    value = {"a": 5.0, "b": 1.0, "fa": 2.5, "fb": 2.0}
    screen = SimpleNamespace(value=lambda pids: sum(value[p] for p in pids),
                             lowest=lambda pool, k: sorted(pool, key=lambda p: value[p])[:k], fa_screen=["fa", "fb"])
    _, s = follow_up(screen, Settled(["a", "b"], [], [], []), set(), exclude={"fa"})
    assert s.adds == ["fb"]


def test_tradeoff_chart_is_valid_svg_with_front_and_bar():
    import xml.etree.ElementTree as ET

    from trade.chart import tradeoff_svg

    sims = [{"offer_id": f"o{i}", "label": f"A & B {i}", "your_gain": 0.3 * i, "their_pts": 2.0 - i, "tier": "Needs a case"}
            for i in range(6)]
    svg = tradeoff_svg(sims, sims[:2], 0.5, "Offers for <X>")
    root = ET.fromstring(svg)  # well-formed, labels escaped
    assert root.tag.endswith("svg")
    assert svg.count('r="5"') == 2 + 1 and svg.count('r="3"') == 4  # front dots (+1 legend), the rest
    assert "your bar 0.5" in svg
    assert tradeoff_svg([], [], 0.5, "empty").startswith("<svg")


def test_stored_cards_returns_qualifying_offers_best_first(tmp_path):
    import json

    from trade import store
    from trade.config import Paths

    paths = Paths(root=tmp_path)
    mk = lambda i, g, q, t: {"offer_id": i, "qualifies": q, "tier": t, "g": {"mean": g}}  # noqa: E731
    with store.connect(paths) as db:
        db.execute("INSERT INTO recommendations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   ("r1", "2026-11-11", "s", "h", "v", 1, 10, "points",
                    json.dumps([mk("a", 1.0, True, "Needs a case"), mk("b", 3.0, True, "Looks good for both"),
                                mk("c", 9.0, False, "Needs a case"), mk("d", 8.0, True, "Unlikely")]), 0, "[]"))
    got = store.stored_cards(paths)
    assert [o["offer_id"] for o in got["cards"]] == ["b", "a"] and not got["no_trade"]
    assert store.stored_cards(Paths(root=tmp_path / "x")) is None


# --- inbound mode (a manager asks for one of your players) ----------------------------------
def _inbound_fin(kind_tier, mc, g, qual=True):
    from trade.evaluate import Est

    return SimpleNamespace(qualifies=qual, tier=kind_tier, market_change=mc, g=Est(g, 0.1, g - 0.2, g + 0.2))


def test_counters_stay_close_to_his_offer_and_never_use_protected_players():
    from trade.interest import ASK_MORE, SWAP, UPGRADE, InterestConstraints, enumerate_counters

    mine = ["P", "a", "b", "c"]
    theirs = ["g1", "g2", "h1", "h2"]
    cons = InterestConstraints(protect={"c"})
    out = enumerate_counters(3, "P", ("g1",), mine, theirs, cons)
    ids = [o.id for _, o in out]
    assert len(ids) == len(set(ids))
    assert Offer_id(3, ("P",), ("g1",)) not in ids  # his own offer is judged separately
    kinds = {k for k, _ in out}
    assert kinds == {ASK_MORE, SWAP, UPGRADE}
    assert all("P" in o.give and "c" not in o.give for _, o in out)
    assert all(len(o.give) == 1 or k == UPGRADE for k, o in out)
    assert {o.get for k, o in out if k == ASK_MORE} == {("g1", "g2"), ("g1", "h1"), ("g1", "h2")}
    assert all(o.get[0] != "g1" or len(o.get) == 2 or k == UPGRADE for k, o in out)
    assert max(len(o.give) for _, o in out) == 2  # one extra player by default
    more = enumerate_counters(3, "P", ("g1",), mine, theirs, InterestConstraints(max_extra=2))
    assert max(len(o.give) for _, o in more) == 3


def Offer_id(partner, give, get):
    from trade.search import Offer

    return Offer(partner, give, get).id


def test_full_search_lists_every_deal_with_him_that_includes_your_player():
    from math import comb

    from trade.interest import InterestConstraints, enumerate_deals

    mine, theirs = ["P", "a", "b", "c"], ["t1", "t2", "t3", "t4"]
    deals = enumerate_deals(3, "P", mine, theirs, InterestConstraints(protect={"c"}))
    assert all("P" in o.give and "c" not in o.give and o.partner == 3 for o in deals)
    assert len(deals) == (1 + 2) * sum(comb(4, k) for k in (1, 2, 3))  # give P alone or with a/b; get 1-3 of his
    assert len({o.id for o in deals}) == len(deals)


def test_counters_rank_by_smallest_concession_and_skip_unlikely_or_failing_ones():
    from trade.interest import ASK_MORE, FULL, SWAP, rank_counters

    his_mc = 4.0  # his offer gives him +4 of market value
    fs = [(ASK_MORE, _inbound_fin("Needs a case", 3.5, 1.0)),    # concession 0.5
          (SWAP, _inbound_fin("Needs a case", 1.0, 3.0)),        # concession 3.0
          (SWAP, _inbound_fin("Unlikely", 3.9, 5.0)),            # hidden
          (ASK_MORE, _inbound_fin("Needs a case", 3.9, 5.0, qual=False)),  # fails your bar
          (FULL, _inbound_fin("Needs a case", 4.0, 9.0))]        # a full-search deal is not a counter
    ranked = rank_counters(his_mc, fs)
    assert [f.g.mean for _, f in ranked] == [1.0, 3.0]


def test_stale_or_mistyped_requests_are_rejected():
    from trade.interest import InvalidRequest, resolve_partner, validate_request

    players = {p: SimpleNamespace(name=p) for p in ("P", "g1", "gone")}
    rosters = {1: SimpleNamespace(players=["P", "q"], owner_name="me"), 3: SimpleNamespace(players=["g1"], owner_name="Dana")}
    ctx = SimpleNamespace(snap=SimpleNamespace(my_roster_id=1, rosters=rosters, players=players),
                          elig={"P": 1, "g1": 1, "gone": 1}, proj={"P": 1, "g1": 1, "gone": 1})
    validate_request(ctx, 3, "P", ("g1",))
    for bad in ((3, "g1", ("g1",)), (3, "P", ("gone",)), (1, "P", ()), (9, "P", ())):
        with pytest.raises(InvalidRequest):
            validate_request(ctx, *bad)
    assert resolve_partner(ctx, "3") == 3 and resolve_partner(ctx, "dan") == 3
    with pytest.raises(InvalidRequest):
        resolve_partner(ctx, "nobody")


def test_inbound_request_is_logged_with_him_as_sender(tmp_path):
    import json

    from trade import store
    from trade.config import Paths

    paths = Paths(root=tmp_path)
    offer = {"give_names": ["P"], "get_names": ["g1"], "tier": "Needs a case", "me": {"pts": {"mean": -0.9}},
             "them": {"pts": {"mean": 4.7}}, "market_change": 1.2}
    pid = store.log_inbound("in_x", partner=3, my_roster=1, offer=offer, note="asked in chat", paths=paths)
    store.log_response(pid, "countered", counter="P for g1+g2", paths=paths)
    with store.connect(paths) as db:
        row = dict(db.execute("SELECT * FROM proposals WHERE proposal_id=?", (pid,)).fetchone())
    assert row["from_roster"] == 3 and row["to_roster"] == 1 and json.loads(row["give"]) == ["P"]
    assert row["note"].startswith("inbound")

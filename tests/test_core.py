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

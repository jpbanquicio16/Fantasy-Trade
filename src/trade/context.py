"""Everything a recommendation run needs, built once from the shared ``waiver`` core.

``Context`` holds the league snapshot, format, projections and screener. ``World`` is one shared
scenario stream: the sampled futures, every player's realised weekly value, and every team's
baseline weekly scores (the current week played by the lineup design's lock-aware policy from the
real lock state, later weeks by the shortlist-tier policy). Evaluating an offer replaces two
teams' columns and re-runs the season, so every difference is paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trade.config import TradeConfig
from trade.format import LeagueFormat, build_format
from trade.season import Outcome, SeasonSim
from trade.valuation import GameDist, Valuer


@dataclass
class Context:
    tcfg: TradeConfig
    snap: object
    fmt: LeagueFormat
    proj: dict
    params: object
    lib: dict
    mcfg: object
    scfg: object
    players: list[str]  # scenario players: rostered + free-agent candidates
    fa_ids: list[str]
    horizon: object
    dists: dict[tuple[str, int], GameDist]
    elig: dict[str, np.ndarray]
    rank_table: dict[tuple[str, int], tuple[float, float]]
    seed: int
    flags: list[str] = field(default_factory=list)

    @property
    def res(self) -> float:
        return self.snap.rules.scoring.units_per_point


def build_context(tcfg: TradeConfig, refresh: bool = True, log=print) -> Context:
    from waiver.config import Paths, load_league, load_model, load_sim
    from waiver.data.ingest import ingest
    from waiver.ingest.sleeper import SleeperClient
    from waiver.models.fit import load
    from waiver.models.project import project_players
    from waiver.recommend.engine import candidate_pool, injured_this_season, manual_projection_entries
    from waiver.sim.scenarios import make_horizon, seed_for
    from waiver.sim.static import Screener
    from waiver.state.build import build_snapshot
    from waiver.state.league import SLOT_ELIGIBILITY
    from waiver.state.store import Store
    from datetime import datetime

    paths, league = Paths().ensure(), load_league()
    mcfg, scfg = load_model(paths), load_sim(paths)
    if refresh:
        log("refreshing stats and schedule ...")
        ingest(league, paths, history=False)
    client = SleeperClient(paths=paths)
    store = Store(paths)
    snap = build_snapshot(league, paths, client, store)
    snap.rules.scoring.require_exact()
    params, lib = load(paths)
    flags = list(snap.flags)
    fmt = build_format(client, league.league_id, snap.week, flags)
    flags = fmt.flags

    season_start = datetime.combine(min(g.date for g in snap.schedule), datetime.min.time(),
                                    tzinfo=snap.taken_at.tzinfo)
    proj = project_players(paths, snap.players, snap.season, snap.taken_at, params, mcfg, league.history_seasons,
                           manual_projection_entries(store, snap.taken_at),
                           injured_this_season=injured_this_season(paths, season_start, snap.taken_at))
    pool = candidate_pool(snap, proj, scfg.pool_size)
    fa = [p for p in pool if snap.pool[p].status == "free_agent"]
    rostered = [p for r in snap.rosters.values() for p in r.players if p in proj]
    players = list(dict.fromkeys(rostered + fa))
    horizon = make_horizon(snap, max(0, fmt.final_week - snap.week))
    if horizon.weeks[-1] < fmt.final_week:
        flags.append(f"schedule_ends_week_{horizon.weeks[-1]}_before_final_{fmt.final_week}")
    scr = Screener(snap, proj, params, mcfg, horizon, players)
    sid = scr.idx.players
    dists = {(sid[i], d): g for (i, d), g in scr.dists.items()}
    rank_table = {(sid[i], d): (g.p, g.mean) for (i, d), g in scr.dists.items()}
    slots = snap.rules.starting_slots
    elig = {p: np.array([bool(SLOT_ELIGIBILITY[s] & set(snap.players[p].positions)) for s in slots])
            for p in players if p in snap.players}
    seed = tcfg.seed if tcfg.seed is not None else seed_for(snap.snapshot_id, "trade")
    return Context(tcfg, snap, fmt, proj, params, lib, mcfg, scfg, players, fa, horizon, dists, elig,
                   rank_table, seed, flags)


class World:
    """One shared scenario stream with every team's baseline season."""

    def __init__(self, ctx: Context, n: int, seed: int, policy: str = "stopping", log=print):
        from waiver.sim.scenarios import make_horizon, sample

        self.ctx, self.n, self.seed = ctx, n, seed
        snap, fmt = ctx.snap, ctx.fmt
        self.sc = sample(snap, ctx.proj, ctx.params, ctx.lib, ctx.mcfg, ctx.scfg, ctx.players, ctx.horizon, n, seed)
        self.valuer = Valuer(self.sc, ctx.dists, ctx.elig, ctx.res, policy)
        self.weak = Valuer(self.sc, ctx.dists, ctx.elig, ctx.res, "mean")
        self.weeks = self.valuer.weeks
        self.sim = SeasonSim(fmt, self.weeks)
        self.cur_col = self.weeks.index(snap.week) if snap.week in self.weeks else None
        self.future_cols = [c for c, w in enumerate(self.weeks) if w > snap.week]
        self.tw = np.zeros((n, fmt.n_teams, len(self.weeks)))
        self.tw_weak = np.zeros_like(self.tw)
        self._plan_sc = None
        self._current_week()
        all_cols = list(range(len(self.weeks)))
        self.fast: dict[int, np.ndarray] = {}  # roster id -> fast-tier weeks of the current roster, all columns
        for t, rid in enumerate(fmt.roster_ids):
            pids = [p for p in snap.rosters[rid].players if p in self.valuer.i]
            self.fast[rid] = self.valuer.team_weeks(pids, all_cols)
            self.tw[:, t, self.future_cols] = self.fast[rid][:, self.future_cols]
            self.tw_weak[:, t, self.future_cols] = self.weak.team_weeks(pids, self.future_cols)
        if self.cur_col is not None:
            self.tw_weak[:, :, self.cur_col] = self.tw[:, :, self.cur_col]
        self.base: Outcome = self.sim.run(self.tw)

    # --- the week in progress: the lineup design's lock-aware policy -------------------
    def planning_sc(self):
        if self._plan_sc is None:
            from waiver.sim.scenarios import make_horizon, sample

            c = self.ctx
            self._plan_sc = sample(c.snap, c.proj, c.params, c.lib, c.mcfg, c.scfg, c.players,
                                   make_horizon(c.snap, 0), c.tcfg.scenarios.planning, self.seed + 7919)
        return self._plan_sc

    def rank(self):
        idx, tab = self.sc.index, self.ctx.rank_table
        inv = {i: p for p, i in idx.items()}
        return lambda i, d: tab.get((inv[i], d), (0.0, 0.0))

    def lock_aware_week(self, rid: int, pids: list[str] | None = None, week: int | None = None) -> np.ndarray:
        """Locked points [N] of a roster for ``week`` under ``waiver.sim.policy.simulate_week``."""
        from waiver.sim.policy import RosterPlan, simulate_week
        from waiver.sim.timeline import roster_plan, week_start

        snap, sc = self.ctx.snap, self.sc
        week = week or snap.week
        plan = roster_plan(sc, snap, rid, None)
        if pids is not None:  # a roster that differs from the snapshot's
            names = [p for p in pids if p in sc.index]
            elig = np.array([[bool(s) for s in self.ctx.elig[p]] for p in names])
            plan = RosterPlan(np.array([sc.index[p] for p in names], dtype=np.int64), elig,
                              np.ones((len(names), sc.horizon.n_days), bool), plan.keep_locked_on_drop)
        start = week_start(sc, snap, rid, plan) if week == snap.week else None
        res = simulate_week(sc, self.planning_sc(), plan, week, start, len(snap.rules.starting_slots), self.rank())
        return res.total / self.ctx.res

    def _current_week(self) -> None:
        if self.cur_col is None or self.ctx.snap.week < self.ctx.fmt.first_unscored_week:
            return
        for t, rid in enumerate(self.ctx.fmt.roster_ids):
            self.tw[:, t, self.cur_col] = self.lock_aware_week(rid)

"""Season simulation: weekly team scores -> matchups, standings, seeding, bracket, champion.

Input is ``TW[N, T, W]``: every team's locked points in each simulated week on shared scenarios.
Everything else is deterministic, so differences between two ``TW`` arrays that differ in two
teams are paired and low-noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade.format import LeagueFormat, rank_teams


@dataclass
class Outcome:
    wins: np.ndarray  # [N, T] total wins incl. already played
    top: np.ndarray  # [N, T] bool: made the playoffs
    champ: np.ndarray  # [N, T] bool
    seed_team: np.ndarray  # [N, T] team index at each seed (best first)


class SeasonSim:
    def __init__(self, fmt: LeagueFormat, weeks: list[int]):
        self.fmt, self.weeks = fmt, weeks
        self.col = {w: i for i, w in enumerate(weeks)}
        self.T = fmt.n_teams
        base = fmt.standings
        self.w0 = np.array([base[r].wins for r in fmt.roster_ids])
        self.pf0 = np.array([base[r].pf for r in fmt.roster_ids])
        self.pa0 = np.array([base[r].pa for r in fmt.roster_ids])
        self.reg_weeks = [w for w in range(fmt.first_unscored_week, fmt.reg_last_week + 1) if w in self.col]

    def run(self, tw: np.ndarray) -> Outcome:
        N, T = tw.shape[0], self.T
        wins = np.tile(self.w0, (N, 1))
        pf = np.tile(self.pf0, (N, 1))
        pa = np.tile(self.pa0, (N, 1))
        for w in self.reg_weeks:
            s = tw[:, :, self.col[w]]
            for a, b in self.fmt.pairings.get(w, []):
                ia, ib = self.fmt.tix(a), self.fmt.tix(b)
                sa, sb = s[:, ia], s[:, ib]
                wins[:, ia] += (sa > sb) + 0.5 * (sa == sb)
                wins[:, ib] += (sb > sa) + 0.5 * (sa == sb)
                pf[:, ia] += sa
                pf[:, ib] += sb
                pa[:, ia] += sb
                pa[:, ib] += sa
        order = rank_teams(wins, pf, pa)  # [N, T] best first
        k = self.fmt.playoff_teams
        top = np.zeros((N, T), bool)
        rows = np.arange(N)[:, None]
        top[rows, order[:, :k]] = True
        champ = self._playoffs(tw, order)
        return Outcome(wins, top, champ, order)

    def _playoffs(self, tw: np.ndarray, order: np.ndarray) -> np.ndarray:
        N = tw.shape[0]
        fmt, rows = self.fmt, np.arange(N)
        seed_pos = np.empty_like(order)  # team -> seed index (0 = best)
        seed_pos[rows[:, None], order] = np.arange(self.T)[None, :]
        winner: dict[int, np.ndarray] = {}

        def resolve(src: tuple[str, int]) -> np.ndarray:
            return order[:, src[1] - 1] if src[0] == "seed" else winner[src[1]]

        champ = np.zeros((N, self.T), bool)
        last = None
        for mt in fmt.bracket:
            a, b = resolve(mt.t1), resolve(mt.t2)
            cols = [self.col[w] for w in fmt.round_weeks[mt.r - 1] if w in self.col]
            sa = tw[rows, a][:, cols].sum(axis=1) if cols else np.zeros(N)
            sb = tw[rows, b][:, cols].sum(axis=1) if cols else np.zeros(N)
            a_wins = (sa > sb) | ((sa == sb) & (seed_pos[rows, a] < seed_pos[rows, b]))  # ties: higher seed
            winner[mt.m] = np.where(a_wins, a, b)
            last = mt.m
        if last is not None:
            champ[rows, winner[last]] = True
        return champ

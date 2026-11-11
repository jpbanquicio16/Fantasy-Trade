"""Roster value under Lock-In: the design's shortlist-tier policy (section 5), vectorised.

Every roster is played by the same achievable, non-anticipative policy:

1. At the start of each week seat the best lineup, weighting each player by his one-slot stopping
   value (his expected locked score if his own optimal thresholds are followed, scaled by his
   chance of being available) - a maximum-weight assignment of players to eligible slots.
2. Play each slot with that player's own thresholds: lock his first game whose score reaches the
   value of waiting (the stopping recursion of the lineup design), otherwise take his team's final
   game of the week, or zero if he sits it.
3. A starter who plays no game all week is replaced by the best eligible bench player who does
   (bench cover; a passed or finished game on the bench cannot be locked).

The thresholds depend only on a player's own distribution, never on a drawn score, so a
player-week's realised value ``R`` can be computed once and any roster's week is a sum of ``R``
over its seated players. That is what makes a whole-league offer cost milliseconds.

It is weaker than the lineup design's base policy v2 (bench players only cover whole-week
absences), so it shortlists; ``finalist_policy: v2`` re-plays the two traded teams with the full
policy from ``waiver.sim.policy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

NEG_INF = -np.inf
BIG = 1e6


@dataclass
class GameDist:
    """One player-game: P(plays) and a normal score, in scenario units; ``emax`` from the screener."""

    p: float
    mean: float
    sigma: float

    def emax(self, v: float) -> float:
        from scipy.stats import norm

        z = (v - self.mean) / self.sigma
        return float(v * norm.cdf(z) + self.mean * norm.sf(z) + self.sigma * norm.pdf(z))


def stopping_chain(games: list[GameDist]) -> tuple[list[float], list[float]]:
    """(V_1..V_n, lock thresholds): V_n = p mu, V_k = p E[max(X, V_{k+1})] + (1 - p) V_{k+1}.

    Game k is locked when its score reaches V_{k+1}; the last game is forced (threshold -inf).
    """
    n = len(games)
    v = [0.0] * n
    for k in range(n - 1, -1, -1):
        g = games[k]
        v[k] = g.p * g.mean if k == n - 1 else g.p * g.emax(v[k + 1]) + (1 - g.p) * v[k + 1]
    thr = [v[k + 1] for k in range(n - 1)] + [NEG_INF]
    return v, thr


def seat(e: np.ndarray, elig: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight assignment of players (rows) to slots (columns): [(player, slot)]."""
    if e.size == 0:
        return []
    cost = np.where(elig, -(e[:, None] + 1e-6), BIG)
    r, c = linear_sum_assignment(cost)
    return [(int(a), int(b)) for a, b in zip(r, c) if cost[a, b] < 0]


class Valuer:
    """Expected and realised weekly value of every player on one scenario set.

    ``sc`` is a ``waiver.sim.scenarios.ScenarioSet``; ``dists[(pid, day)]`` the screener's game
    distributions; ``elig[pid]`` a boolean vector over the league's starting slots.
    ``policy="mean"`` is the weak management policy used as a sensitivity run for the other ten
    teams: lock any score above his own average.
    """

    def __init__(self, sc, dists: dict[tuple[str, int], GameDist], elig: dict[str, np.ndarray], res: float,
                 policy: str = "stopping"):
        self.sc, self.res, self.policy = sc, res, policy
        self.weeks = list(sc.horizon.weeks)
        self.pid = list(sc.players)
        self.i = {p: k for k, p in enumerate(self.pid)}
        P, W, N = len(self.pid), len(self.weeks), sc.n
        self.E = np.zeros((P, W))
        self.R = np.zeros((N, P, W), np.float32)
        self.absent = np.ones((N, P, W), bool)
        self.n_slots = len(next(iter(elig.values()))) if elig else 0
        self.elig = np.array([elig.get(p, np.zeros(self.n_slots, bool)) for p in self.pid], bool)
        for p, i in self.i.items():
            for wi, w in enumerate(self.weeks):
                days = [d for d in sc.horizon.week_days[w] if sc.has_game[i, d] and (p, d) in dists]
                if not days:
                    continue
                games = [dists[(p, d)] for d in days]
                v, thr = stopping_chain(games)
                if policy == "mean":
                    thr = [g.mean for g in games[:-1]] + [NEG_INF]
                self.E[i, wi] = v[0] / res
                val = np.zeros(N)
                done = np.zeros(N, bool)
                anyplay = np.zeros(N, bool)
                for d, t in zip(days, thr):
                    pl = sc.plays[:, i, d]
                    x = sc.units[:, i, d] / res
                    anyplay |= pl
                    lock = pl & ~done & (x * res >= t)
                    val = np.where(lock, x, val)
                    done |= lock
                self.R[:, i, wi] = val
                self.absent[:, i, wi] = ~anyplay

    # --- lineups ------------------------------------------------------------------
    def team_weeks(self, pids: list[str], cols: list[int], scale: dict[str, float] | None = None,
                   info: dict | None = None) -> np.ndarray:
        """Locked points [N, len(cols)] of a roster in each week column, on the shared scenarios.

        ``scale`` multiplies named players' output (reversal tests). ``info`` (if given) receives
        ``starts[pid]`` - weeks started - and ``cover_value``.
        """
        N = self.sc.n
        out = np.zeros((N, len(cols)))
        if not pids:
            return out
        idx = np.array([self.i[p] for p in pids if p in self.i])
        names = [p for p in pids if p in self.i]
        f = np.array([(scale or {}).get(p, 1.0) for p in names])
        elig = self.elig[idx]
        starts = {p: 0 for p in names}
        cover = 0.0
        for k, c in enumerate(cols):
            e = self.E[idx, c] * f
            pairs = seat(e, elig)
            seated = {m for m, _ in pairs}
            bench = sorted((m for m in range(len(idx)) if m not in seated and e[m] > 0), key=lambda m: -e[m])
            used = np.zeros((N, len(idx)), bool)
            week = np.zeros(N)
            for m, s in pairs:
                starts[names[m]] += 1
                val = self.R[:, idx[m], c] * f[m]
                miss = self.absent[:, idx[m], c]
                if bench and miss.any():
                    cov = np.zeros(N)
                    for b in bench:
                        if not elig[b, s]:
                            continue
                        ok = miss & ~used[:, b] & ~self.absent[:, idx[b], c]
                        cov = np.where(ok, self.R[:, idx[b], c] * f[b], cov)
                        used[:, b] |= ok
                        miss = miss & ~ok
                    val = val + cov
                    cover += float(cov.sum()) / N
                week += val
            out[:, k] = week
        if info is not None:
            info["starts"] = {p: starts[p] / max(1, len(cols)) for p in names}
            info["cover_value"] = cover / max(1, len(cols))
        return out

    def lineup_value(self, pids: list[str], cols: list[int]) -> float:
        """Deterministic screen value: expected locked FP per week summed over ``cols``."""
        names = [p for p in pids if p in self.i]
        if not names or not cols:
            return 0.0
        idx = np.array([self.i[p] for p in names])
        elig = self.elig[idx]
        total = 0.0
        for c in cols:
            e = self.E[idx, c]
            total += sum(e[m] for m, _ in seat(e, elig))
        return total

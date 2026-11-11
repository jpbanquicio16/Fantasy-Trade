"""Paired, whole-league evaluation of a complete trade against holding and against waiver moves.

A trade replaces two teams' weekly scores from the first week it can count; every other team's
scores are the *same arrays* in both worlds, so differences are paired and a null trade is exactly
zero. Reported per side: expected weekly points (regular season and playoff weeks), P(top 6) and
P(title) in percentage points, each with a paired standard error and a model band (10th-90th
percentile of the scenario batches, where each batch shares one draw of every player's rates).

    G(T) = max_w U(R^T + w) - max_w U(R + w)          (design section 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trade.search import Settled

OBJ_KEY = {"points": "pts", "top6": "top6", "title": "title"}
UNIT = {"pts": "FP/wk", "playoff_pts": "FP/wk", "top6": "pts", "title": "pts"}


@dataclass
class Est:
    mean: float
    se: float
    p10: float
    p90: float

    @classmethod
    def of(cls, d: np.ndarray, batch: np.ndarray) -> "Est":
        n = len(d)
        bm = np.array([d[batch == b].mean() for b in np.unique(batch)])
        return cls(float(d.mean()), float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                   float(np.percentile(bm, 10)), float(np.percentile(bm, 90)))

    def to_json(self) -> dict:
        return {"mean": self.mean, "se": self.se, "p10": self.p10, "p90": self.p90}


@dataclass
class Choice:
    """Every decision inside one evaluated offer, fixed so it can be re-run on fresh scenarios."""

    offer: object
    my: Settled
    their: Settled
    alt: Settled | None  # best waiver-only roster for you (None = hold)


@dataclass
class Result:
    me: dict[str, np.ndarray]
    them: dict[str, np.ndarray]
    alt: dict[str, np.ndarray] | None
    g: np.ndarray  # per-scenario G(T) in the objective's unit


class Evaluator:
    def __init__(self, world, ctx, first_week: int, frac: float = 1.0, v2=None):
        self.w, self.ctx, self.frac = world, ctx, frac
        weeks, fmt = world.weeks, ctx.fmt
        self.first_col = next((c for c, wk in enumerate(weeks) if wk >= max(first_week, ctx.snap.week)), len(weeks))
        self.cols = list(range(self.first_col, len(weeks)))
        self.reg = [c for c in self.cols if weeks[c] <= fmt.reg_last_week]
        self.po = [c for c, wk in enumerate(weeks) if wk in set(fmt.playoff_weeks)]
        self.first_week = first_week
        self._base: dict = {}
        self.v2 = v2  # optional callable(rid, pids, cols) -> [N, len(cols)] from the lock-aware policy

    # --- baselines -----------------------------------------------------------------
    def base(self, others: str, partner: int | None):
        """(tw, outcome) with every roster as it stands. ``weak``: the other ten teams play the weak
        policy, the two traded teams the normal one."""
        key = (others, partner if others == "weak" else None)
        if key not in self._base:
            tw = self.w.tw
            if others == "weak":
                tw = self.w.tw_weak.copy()
                for rid in (self.ctx.snap.my_roster_id, partner):
                    if rid is not None:
                        t = self.ctx.fmt.tix(rid)
                        tw[:, t, :] = self.w.tw[:, t, :]
            self._base[key] = (tw, self.w.sim.run(tw))
        return self._base[key]

    # --- applying a roster change --------------------------------------------------
    def team_cols(self, rid: int, pids: list[str], scale=None) -> np.ndarray:
        """New weekly scores of ``rid`` in self.cols, minus what the current roster scores there."""
        names = [p for p in pids if p in self.w.valuer.i]
        new = self.w.valuer.team_weeks(names, self.cols, scale)
        old = self.w.fast[rid][:, self.cols]
        delta = new - old
        if self.cols:
            delta[:, 0] *= self.frac
        return delta

    def metrics(self, tw: np.ndarray, out, t: int) -> dict[str, np.ndarray]:
        m = {"top6": out.top[:, t] * 100.0, "title": out.champ[:, t] * 100.0}
        m["pts"] = tw[:, t, self.reg].mean(axis=1) if self.reg else np.zeros(tw.shape[0])
        m["playoff_pts"] = tw[:, t, self.po].mean(axis=1) if self.po else np.zeros(tw.shape[0])
        return m

    def diff(self, rosters: dict[int, list[str]], watch: list[int], others: str = "stopping",
             partner: int | None = None, scale=None) -> dict[int, dict[str, np.ndarray]]:
        """Paired change of each watched team's metrics when ``rosters`` replace the current ones."""
        tw0, out0 = self.base(others, partner)
        tw = tw0.copy()
        if self.v2 is not None:  # the traded teams' "before" is played by the same policy as their "after"
            tw0 = tw0.copy()
            for rid in set(rosters) | set(watch):
                t = self.ctx.fmt.tix(rid)
                cur = [p for p in self.ctx.snap.rosters[rid].players if p in self.w.valuer.i]
                tw0[:, t, self.cols] = tw[:, t, self.cols] = self._v2(rid, cur)
            out0 = self.w.sim.run(tw0)
        for rid, pids in rosters.items():
            t = self.ctx.fmt.tix(rid)
            if self.v2 is not None:
                tw[:, t, self.cols] = self._v2(rid, pids)
            else:
                tw[:, t, self.cols] += self.team_cols(rid, pids, scale)
        out1 = self.w.sim.run(tw)
        res = {}
        for rid in watch:
            t = self.ctx.fmt.tix(rid)
            a, b = self.metrics(tw, out1, t), self.metrics(tw0, out0, t)
            res[rid] = {k: a[k] - b[k] for k in a}
        return res

    def _v2(self, rid: int, pids: list[str]) -> np.ndarray:
        key = (rid, frozenset(pids))
        cache = self.__dict__.setdefault("_v2cache", {})
        if key not in cache:
            cache[key] = self.v2(rid, sorted(pids), self.cols)
        return cache[key]

    # --- G(T) ----------------------------------------------------------------------
    def run(self, ch: Choice, objective: str, others: str = "stopping", scale=None,
            hold_partner: bool = False) -> Result:
        me, pr = self.ctx.snap.my_roster_id, ch.offer.partner
        rosters = {me: ch.my.pids}
        if not hold_partner:
            rosters[pr] = ch.their.pids
        d = self.diff(rosters, [me, pr], others, pr, scale)
        alt = None
        if ch.alt is not None:
            alt = self.diff({me: ch.alt.pids}, [me], others, pr)[me]
        key = OBJ_KEY[objective]
        base_alt = alt[key] if alt is not None else 0.0
        return Result(d[me], d[pr], alt, d[me][key] - base_alt)


def best_alt(ev: Evaluator, options: list[Settled], objective: str) -> tuple[Settled | None, float]:
    """Best waiver-only roster by mean objective; None (hold) if nothing beats zero."""
    me = ev.ctx.snap.my_roster_id
    key = OBJ_KEY[objective]
    best, best_v = None, 0.0
    for o in options:
        v = float(ev.diff({me: o.pids}, [me])[me][key].mean())
        if v > best_v:
            best, best_v = o, v
    return best, best_v

"""Market view and the heuristic acceptance tier (design sections 4 and 7).

Market value is what other managers are likely to *believe* a player is worth, kept apart from
what the simulation says he is worth to a roster. Perceived value is Sleeper's consensus
``search_rank`` converted to a per-game scale by a smooth monotone fit (FP per game on ln rank) of projected FP per
game on rank, so it is comparable with production but never blended into it. Draft price (pick
and round in this league's draft) is shown beside it. Acceptance is a tier from two signs, never
a probability: no proposal-and-response data exist to calibrate one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIERS = ("Looks good for both", "Plausible on reputation", "Needs a case", "Unlikely")


def fit_rank_curve(rank: np.ndarray, fp: np.ndarray) -> tuple[float, float]:
    """Smooth, strictly monotone map from consensus rank to FP per game: fp = a + b ln(rank), b <= 0.

    (An isotonic fit was tried first; it is piecewise constant, so neighbouring ranks tie and a
    trade between them looks market-neutral.)
    """
    x = np.log(np.maximum(rank, 1.0))
    b, a = np.polyfit(x, fp, 1)
    return float(a), float(min(b, -1e-6))


@dataclass
class Market:
    rank: dict[str, float] = field(default_factory=dict)
    value: dict[str, float] = field(default_factory=dict)  # perceived FP per game
    pick: dict[str, tuple[int, int]] = field(default_factory=dict)  # pid -> (pick_no, round)
    repl: float = 0.0
    available: bool = True

    def perceived(self, p: str) -> float:
        return max(self.value.get(p, self.repl) - self.repl, 0.0)

    def change(self, receives: tuple[str, ...], gives: tuple[str, ...]) -> float:
        """Market-perceived change for the side that gets ``receives`` and loses ``gives``."""
        if not self.available:
            return 0.0
        return sum(self.perceived(p) for p in receives) - sum(self.perceived(p) for p in gives)


def build_market(ctx, client=None) -> Market:
    import polars as pl
    from waiver.config import Paths
    from waiver.data import tables as T

    m = Market()
    df = T.read(Paths(), "players")
    if df is None or "search_rank" not in df.columns:
        m.available = False
        ctx.flags.append("market_unavailable")
        return m
    rank = {r["player_id"]: float(r["search_rank"]) for r in df.iter_rows(named=True)
            if r["search_rank"] is not None}
    pids = [p for p in ctx.players if p in rank and p in ctx.proj]
    if len(pids) < 10:
        m.available = False
        ctx.flags.append("market_unavailable")
        return m
    pids.sort(key=lambda p: rank[p])
    fp = np.array([ctx.proj[p].linear_fp(ctx.snap.rules.scoring) for p in pids])
    a, b = fit_rank_curve(np.array([rank[p] for p in pids]), fp)
    fit = np.array([a + b * np.log(max(rank[p], 1.0)) for p in pids])
    m.rank = {p: rank[p] for p in pids}
    m.value = dict(zip(pids, fit))
    fa = sorted((p for p in ctx.fa_ids if p in m.value), key=lambda p: rank[p])[:10]
    m.repl = float(np.median([m.value[p] for p in fa])) if fa else float(np.percentile(fit, 25))
    # unranked players are priced at replacement
    for p in ctx.players:
        if p not in m.value and p in ctx.proj:
            m.value[p] = m.repl
    if client is not None:
        try:
            drafts = client._get("v1", f"/league/{ctx.snap.league_id}/drafts", None, 86400) or []
            for d in drafts:
                for pk in client._get("v1", f"/draft/{d['draft_id']}/picks", None, 86400) or []:
                    m.pick[str(pk["player_id"])] = (int(pk["pick_no"]), int(pk["round"]))
        except Exception:  # noqa: BLE001 - draft price is context only
            ctx.flags.append("draft_picks_unavailable")
    return m


def tier(partner_points_change: float, partner_market_change: float) -> str:
    """The compatibility tier (heuristic, uncalibrated)."""
    gain, mgain = partner_points_change > 0, partner_market_change > 0
    if gain and mgain:
        return TIERS[0]
    if not gain and mgain:
        return TIERS[1]
    if gain and not mgain:
        return TIERS[2]
    return TIERS[3]

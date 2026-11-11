"""Trade-tool configuration: paths, choices and the open-rule switches (design sections 1-2, 7).

Data, projection parameters and league config are shared with ``waiver-bot`` (the design's "shared
core package"); this repository keeps only its own recommendations and proposals log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def db(self) -> Path:
        return self.data / "trade.sqlite"

    @property
    def reports(self) -> Path:
        return self.data / "reports"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def ensure(self) -> "Paths":
        self.data.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        return self


class ScenarioCfg(BaseModel):
    shortlist: int = 1000
    finalist: int = 2000
    planning: int = 300


class SearchCfg(BaseModel):
    shortlist_size: int = 40
    finalists: int = 12
    per_partner: int = 10
    partner_loss_tolerance_fp: float = 3.0
    fa_candidates: int = 6
    drop_candidates: int = 3
    shapes: list[str] = Field(default_factory=lambda: ["1-1", "2-1", "1-2", "2-2"])


class BarCfg(BaseModel):
    min_gain: float = 0.5
    min_gain_prob_points: float = 1.0
    se_fraction: float = 1 / 3
    require_p10_positive: bool = True


class CardsCfg(BaseModel):
    max_cards: int = 5
    max_per_partner: int = 2


class RulesCfg(BaseModel):
    # Open (design section 2): can a player who already played this week be used by his new team
    # before next week?  next_week_if_played = conservative default; immediate = optimistic.
    traded_player_week_rule: Literal["next_week_if_played", "immediate"] = "next_week_if_played"
    pending_score_on_trade: Literal["forfeited_unless_locked", "kept"] = "forfeited_unless_locked"
    assume_accepted_in_days: int = 0
    ties_go_to: Literal["higher_seed"] = "higher_seed"


class TradeConfig(BaseModel):
    objective: Literal["points", "top6", "title"] = "points"
    scenarios: ScenarioCfg = Field(default_factory=ScenarioCfg)
    search: SearchCfg = Field(default_factory=SearchCfg)
    bar: BarCfg = Field(default_factory=BarCfg)
    cards: CardsCfg = Field(default_factory=CardsCfg)
    rules: RulesCfg = Field(default_factory=RulesCfg)
    finalist_policy: Literal["shortlist", "v2"] = "shortlist"
    seed: int | None = None

    def bar_for(self, objective: str) -> float:
        return self.bar.min_gain if objective == "points" else self.bar.min_gain_prob_points


def load(paths: Paths | None = None) -> TradeConfig:
    paths = paths or Paths()
    for name in ("trade.yaml", "trade.example.yaml"):
        p = paths.configs / name
        if p.exists():
            with p.open() as f:
                return TradeConfig(**(yaml.safe_load(f) or {}))
    return TradeConfig()

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PortfolioConstraints:
    max_position_weight: float = 0.08
    benchmark_sector_band: float = 0.05
    max_adv_participation: float = 0.10
    turnover_penalty: float = 0.10
    concentration_penalty: float = 0.10


DEFAULT_CONSTRAINTS = PortfolioConstraints()

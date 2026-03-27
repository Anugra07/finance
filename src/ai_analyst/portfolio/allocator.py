from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.portfolio.constraints import PortfolioConstraints
from ai_analyst.reporting.io import resolve_latest_report_path
from ai_analyst.warehouse.database import connect


def _latest_report_path(settings: Settings) -> Path | None:
    return resolve_latest_report_path(settings)


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_sector_weights(settings: Settings) -> dict[str, float]:
    conn = connect(settings)
    try:
        try:
            universe = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY ticker
                            ORDER BY snapshot_at DESC
                        ) AS row_num
                    FROM v1_universe
                )
                SELECT sector
                FROM ranked
                WHERE row_num = 1
                """
            ).df()
        except Exception:
            return {}
    finally:
        conn.close()
    if universe.empty:
        return {}
    counts = universe["sector"].value_counts(normalize=True)
    return {str(sector): float(weight) for sector, weight in counts.items()}


def build_rebalance_plan(
    settings: Settings,
    *,
    report_path: Path | None = None,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, object]:
    constraints = constraints or PortfolioConstraints()
    path = report_path or _latest_report_path(settings)
    if path is None or not path.exists():
        return {"status": "no_report"}

    report = _load_report(path)
    top_stocks = pd.DataFrame(report.get("top_stocks") or [])
    if top_stocks.empty:
        return {"status": "no_candidates", "report_path": str(path)}

    top_stocks["score"] = (
        pd.to_numeric(top_stocks["expected_excess_alpha_rank"], errors="coerce").fillna(0.0) - 0.5
    ).clip(lower=0.0) * pd.to_numeric(top_stocks["confidence_score"], errors="coerce").fillna(0.0)
    candidates = top_stocks.loc[top_stocks["score"] > 0].copy()
    if candidates.empty:
        return {"status": "no_positive_scores", "report_path": str(path)}

    candidates["weight"] = candidates["score"] / candidates["score"].sum()
    candidates["weight"] = candidates["weight"].clip(upper=constraints.max_position_weight)
    candidates["weight"] = candidates["weight"] / candidates["weight"].sum()

    benchmark_weights = _benchmark_sector_weights(settings)
    if benchmark_weights:
        for sector, benchmark_weight in benchmark_weights.items():
            sector_mask = candidates["sector"] == sector
            if not sector_mask.any():
                continue
            current_sector_weight = float(candidates.loc[sector_mask, "weight"].sum())
            max_sector_weight = min(1.0, benchmark_weight + constraints.benchmark_sector_band)
            if current_sector_weight > max_sector_weight:
                scale = max_sector_weight / current_sector_weight
                candidates.loc[sector_mask, "weight"] *= scale
        candidates["weight"] = candidates["weight"] / candidates["weight"].sum()

    allocations = [
        {
            "ticker": row["ticker"],
            "sector": row.get("sector"),
            "target_weight": round(float(row["weight"]), 6),
            "score": round(float(row["score"]), 6),
        }
        for _, row in candidates.sort_values("weight", ascending=False).iterrows()
    ]
    sector_weights = (
        candidates.groupby("sector", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )
    return {
        "status": "ok",
        "report_path": str(path),
        "constraints": asdict(constraints),
        "allocations": allocations,
        "sector_weights": [
            {"sector": row["sector"], "target_weight": round(float(row["weight"]), 6)}
            for _, row in sector_weights.iterrows()
        ],
        "notes": [
            "heuristic_allocation",
            "position_caps_applied",
            "sector_band_soft_clip_applied" if benchmark_weights else "no_benchmark_sector_weights",
        ],
    }

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.events.exposures import compute_sector_context_shocks
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def build_sector_opportunity_frame(
    settings: Settings,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    conn = connect(settings)
    try:
        theme_daily = conn.execute("SELECT * FROM theme_intensity_daily").df()
        sector_exposures = conn.execute("SELECT * FROM sector_theme_exposure").df()
        features = conn.execute("SELECT * FROM feature_matrix").df()
    finally:
        conn.close()

    if features.empty:
        return pd.DataFrame()

    features["date"] = pd.to_datetime(features["date"])
    selected_date = pd.to_datetime(as_of).date() if as_of else features["date"].max().date()
    finance_slice = features.loc[features["date"].dt.date == selected_date].copy()
    if finance_slice.empty:
        return pd.DataFrame()

    finance_slice["finance_component"] = (
        finance_slice["ret_5d"].fillna(0.0)
        + 0.5 * finance_slice["ret_20d"].fillna(0.0)
        - 0.25 * finance_slice["realized_vol_20d"].fillna(0.0)
    )
    finance_scores = (
        finance_slice.groupby("sector", as_index=False)["finance_component"]
        .mean()
        .rename(columns={"finance_component": "finance_score_raw"})
    )
    finance_scores["finance_score"] = _zscore(finance_scores["finance_score_raw"])
    finance_scores = finance_scores.drop(columns=["finance_score_raw"])

    context_scores = pd.DataFrame(
        {
            "as_of_date": [selected_date] * finance_scores["sector"].nunique(),
            "sector": finance_scores["sector"].tolist(),
            "context_shock": 0.0,
            "top_theme": [None] * finance_scores["sector"].nunique(),
            "supporting_themes": [[] for _ in range(finance_scores["sector"].nunique())],
        }
    )
    if not theme_daily.empty and not sector_exposures.empty:
        theme_daily["date"] = pd.to_datetime(theme_daily["date"])
        selected_theme = theme_daily.loc[theme_daily["date"].dt.date == selected_date].copy()
        context_scores = compute_sector_context_shocks(selected_theme, sector_exposures)
        if context_scores.empty:
            context_scores = pd.DataFrame(
                {
                    "as_of_date": [selected_date] * finance_scores["sector"].nunique(),
                    "sector": finance_scores["sector"].tolist(),
                    "context_shock": 0.0,
                    "top_theme": [None] * finance_scores["sector"].nunique(),
                    "supporting_themes": [[] for _ in range(finance_scores["sector"].nunique())],
                }
            )

    rankings = finance_scores.merge(context_scores, on="sector", how="left")
    rankings["as_of_date"] = pd.to_datetime(rankings.get("as_of_date", selected_date)).dt.date
    rankings["context_shock"] = rankings["context_shock"].fillna(0.0)
    rankings["sector_score"] = rankings["finance_score"] * 0.35 + rankings["context_shock"] * 0.65
    rankings = rankings.sort_values(
        ["sector_score", "sector"],
        ascending=[False, True],
    ).reset_index(drop=True)
    rankings["rank_desc"] = np.arange(1, len(rankings) + 1)
    rankings["rank_asc"] = rankings["rank_desc"][::-1].to_numpy()
    rankings["solution_bucket"] = np.where(
        rankings["sector_score"] >= 0.20,
        "beneficiary",
        np.where(rankings["sector_score"] <= -0.20, "pressured", "watch"),
    )
    rankings["transform_loaded_at"] = datetime.now(tz=UTC)
    return rankings[
        [
            "as_of_date",
            "sector",
            "sector_score",
            "context_shock",
            "finance_score",
            "rank_desc",
            "rank_asc",
            "top_theme",
            "supporting_themes",
            "solution_bucket",
            "transform_loaded_at",
        ]
    ]


def materialize_sector_rankings(
    settings: Settings,
    *,
    as_of: date | None = None,
) -> list[Path]:
    rankings = build_sector_opportunity_frame(settings, as_of=as_of)
    if rankings.empty:
        return []

    outputs: list[Path] = []
    for as_of_date, frame in rankings.groupby("as_of_date"):
        out_path = warehouse_partition_path(
            settings,
            domain="rankings/sector",
            partition_date=as_of_date,
            stem=f"sector_rankings_{as_of_date.isoformat()}",
        )
        write_parquet(frame, out_path)
        outputs.append(out_path)
    return outputs


def load_sector_rankings(
    settings: Settings,
    *,
    as_of: date | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    conn = connect(settings)
    try:
        rankings = conn.execute("SELECT * FROM sector_rankings").df()
    finally:
        conn.close()

    if rankings.empty:
        return rankings
    rankings["as_of_date"] = pd.to_datetime(rankings["as_of_date"])
    target_date = pd.to_datetime(as_of).date() if as_of else rankings["as_of_date"].max().date()
    return (
        rankings.loc[rankings["as_of_date"].dt.date == target_date]
        .sort_values("rank_desc")
        .head(limit)
    )

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path


def build_calibration_metrics(
    outcomes: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    if outcomes.empty or labels.empty:
        return pd.DataFrame()

    frame = outcomes.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of"], utc=True).dt.date
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    merged = frame.merge(
        labels[["date", "ticker", "excess_alpha_5d"]],
        left_on=["as_of_date", "ticker"],
        right_on=["date", "ticker"],
        how="left",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["actual_direction"] = np.where(
        merged["excess_alpha_5d"].fillna(0.0) > 0,
        1.0,
        0.0,
    )
    merged["predicted_up"] = np.where(merged["direction"] == "up", 1.0, 0.0)
    merged["brier_component"] = (merged["confidence"].fillna(0.0) - merged["actual_direction"]) ** 2
    merged["hit"] = np.where(
        (merged["direction"] == "up") & (merged["actual_direction"] == 1.0),
        1.0,
        np.where(
            (merged["direction"] == "down") & (merged["actual_direction"] == 0.0),
            1.0,
            np.where(merged["direction"] == "neutral", np.nan, 0.0),
        ),
    )
    merged["confidence_bucket"] = pd.cut(
        merged["confidence"].fillna(0.0),
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    ).astype(str)

    rows: list[dict[str, object]] = []
    as_of_date = pd.Timestamp.utcnow().date()
    rows.append(
        {
            "as_of_date": as_of_date,
            "metric_name": "brier_score",
            "metric_value": float(merged["brier_component"].mean()),
            "metric_bucket": "all",
            "source": "label_matrix_5d_proxy",
            "transform_loaded_at": datetime.now(tz=UTC),
        }
    )
    rows.append(
        {
            "as_of_date": as_of_date,
            "metric_name": "abstention_rate",
            "metric_value": float(merged["abstain"].mean()),
            "metric_bucket": "all",
            "source": "forecast_outcomes",
            "transform_loaded_at": datetime.now(tz=UTC),
        }
    )
    hit_series = merged["hit"].dropna()
    rows.append(
        {
            "as_of_date": as_of_date,
            "metric_name": "hit_rate",
            "metric_value": float(hit_series.mean()) if not hit_series.empty else 0.0,
            "metric_bucket": "all",
            "source": "label_matrix_5d_proxy",
            "transform_loaded_at": datetime.now(tz=UTC),
        }
    )
    for bucket, group in merged.groupby("confidence_bucket", dropna=False):
        if group.empty:
            continue
        rows.append(
            {
                "as_of_date": as_of_date,
                "metric_name": "bucket_hit_rate",
                "metric_value": float(group["hit"].dropna().mean())
                if not group["hit"].dropna().empty
                else 0.0,
                "metric_bucket": str(bucket),
                "source": "label_matrix_5d_proxy",
                "transform_loaded_at": datetime.now(tz=UTC),
            }
        )
    return pd.DataFrame(rows)


def materialize_calibration_metrics(settings: Settings) -> list[Path]:
    conn = connect(settings)
    try:
        outcomes = conn.execute("SELECT * FROM forecast_outcomes").df()
        labels = conn.execute("SELECT * FROM label_matrix").df()
    finally:
        conn.close()

    frame = build_calibration_metrics(outcomes, labels)
    paths: list[Path] = []
    if frame.empty:
        return paths
    trade_date = pd.to_datetime(frame["as_of_date"]).dt.date.max()
    out_path = warehouse_partition_path(
        settings,
        domain="forecast/calibration_metrics",
        partition_date=trade_date,
        stem=f"forecast_calibration_metrics_{trade_date.isoformat()}",
    )
    write_parquet(frame, out_path)
    paths.append(out_path)
    return paths

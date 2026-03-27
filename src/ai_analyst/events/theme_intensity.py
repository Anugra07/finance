from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path


def _list_len(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, list | tuple | set):
        return len(value)
    if isinstance(value, np.ndarray):
        return int(value.size)
    return 1


def _weight_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    if frame.empty:
        return frame

    for column in ["severity", "confidence", "novelty", "market_relevance", "duration_hours"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    duration_scale = np.log1p(frame["duration_hours"].clip(lower=0.0)) / np.log(25.0)
    duration_scale = duration_scale.clip(lower=0.0, upper=1.0).fillna(0.0)
    frame["intensity_component"] = (
        frame["severity"].clip(lower=0.0)
        * frame["confidence"].clip(lower=0.0)
        * (1.0 + frame["novelty"].clip(lower=0.0))
        * np.maximum(frame["market_relevance"].clip(lower=0.0), 0.25)
        * (1.0 + duration_scale)
    )
    return frame


def build_theme_intensity_frames(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _weight_events(events)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = frame.dropna(subset=["theme", "event_time"]).copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["bucket_time"] = frame["event_time"].dt.floor("h")
    frame["date"] = frame["event_time"].dt.date
    frame["region"] = frame["region"].fillna("unknown")
    frame["affected_sector_count"] = frame["affected_sectors"].apply(_list_len)

    def _aggregate(group: pd.DataFrame, *, key: str) -> dict[str, object]:
        return {
            key: group.iloc[0][key],
            "theme": group.iloc[0]["theme"],
            "intensity": float(group["intensity_component"].sum()),
            "event_count": int(len(group)),
            "avg_severity": float(group["severity"].mean()),
            "avg_novelty": float(group["novelty"].mean()),
            "event_dispersion_score": float(
                (group["region"].nunique() + group["affected_sector_count"].sum())
                / max(len(group), 1)
            ),
            "latest_event_time": group["event_time"].max(),
            "transform_loaded_at": datetime.now(tz=group["event_time"].dt.tz),
        }

    hourly = pd.DataFrame(
        [
            _aggregate(group, key="bucket_time")
            for _, group in frame.groupby(["bucket_time", "theme"], sort=True)
        ]
    )
    daily = pd.DataFrame(
        [_aggregate(group, key="date") for _, group in frame.groupby(["date", "theme"], sort=True)]
    )
    return hourly, daily


def theme_intensity_wide(daily_frame: pd.DataFrame) -> pd.DataFrame:
    if daily_frame.empty:
        return pd.DataFrame()
    pivot = (
        daily_frame.pivot_table(index="date", columns="theme", values="intensity", aggfunc="sum")
        .fillna(0.0)
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def materialize_theme_intensity_tables(settings: Settings) -> tuple[list[Path], list[Path]]:
    conn = connect(settings)
    try:
        events = conn.execute("SELECT * FROM normalized_events").df()
    finally:
        conn.close()

    hourly, daily = build_theme_intensity_frames(events)
    hourly_paths: list[Path] = []
    daily_paths: list[Path] = []

    if not hourly.empty:
        hourly["bucket_time"] = pd.to_datetime(hourly["bucket_time"], utc=True)
        for bucket_date, frame in hourly.groupby(hourly["bucket_time"].dt.date):
            out_path = warehouse_partition_path(
                settings,
                domain="themes/hourly",
                partition_date=bucket_date,
                stem=f"theme_intensity_hourly_{bucket_date.isoformat()}",
            )
            write_parquet(frame, out_path)
            hourly_paths.append(out_path)

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        for trade_date, frame in daily.groupby(daily["date"].dt.date):
            out_path = warehouse_partition_path(
                settings,
                domain="themes/daily",
                partition_date=trade_date,
                stem=f"theme_intensity_daily_{trade_date.isoformat()}",
            )
            write_parquet(frame, out_path)
            daily_paths.append(out_path)

    return hourly_paths, daily_paths

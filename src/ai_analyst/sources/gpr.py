"""Caldara-Iacoviello Geopolitical Risk (GPR) Index collector.

The GPR index is a freely available daily/monthly measure of geopolitical
tensions computed from newspaper article counts.  The data is published as
an Excel file at matteoiacoviello.com and is updated weekly.

This module downloads the spreadsheet, extracts the daily and monthly
series, and writes them into the warehouse as Parquet files under the
``macro_vintages`` schema so they integrate seamlessly with the existing
FRED-based macro pipeline.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.utils.dates import utc_now
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.layout import raw_snapshot_path, warehouse_partition_path

logger = logging.getLogger(__name__)

GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

GPR_SERIES_MAPPING = {
    "GPRD": "GPR_DAILY",
    "GPRD_ACT": "GPR_DAILY_ACTS",
    "GPRD_THREAT": "GPR_DAILY_THREATS",
    "GPR": "GPR_MONTHLY",
    "GPRA": "GPR_MONTHLY_ACTS",
    "GPRT": "GPR_MONTHLY_THREATS",
}


def _download_excel(url: str, *, timeout: float = 30.0) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _parse_daily_gpr(content: bytes) -> pd.DataFrame:
    """Parse the daily GPR Excel file into a long-format DataFrame."""
    df = pd.read_excel(io.BytesIO(content), sheet_name=0)
    date_col = [col for col in df.columns if "date" in str(col).lower()]
    if not date_col:
        date_col = [df.columns[0]]
    df = df.rename(columns={date_col[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    value_cols = [col for col in df.columns if col != "date"]
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        observation_date = row["date"]
        for col in value_cols:
            val = row[col]
            if pd.isna(val):
                continue
            series_id = GPR_SERIES_MAPPING.get(str(col).strip(), f"GPR_{col}")
            rows.append(
                {
                    "series_id": series_id,
                    "observation_date": observation_date,
                    "value": float(val),
                }
            )
    return pd.DataFrame(rows)


def _parse_monthly_gpr(content: bytes) -> pd.DataFrame:
    """Parse the monthly GPR Excel file into long-format rows."""
    df = pd.read_excel(io.BytesIO(content), sheet_name=0)
    date_col = [col for col in df.columns if "date" in str(col).lower()]
    if not date_col:
        date_col = [df.columns[0]]
    df = df.rename(columns={date_col[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    value_cols = [col for col in df.columns if col != "date"]
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        observation_date = row["date"]
        for col in value_cols:
            val = row[col]
            if pd.isna(val):
                continue
            series_id = GPR_SERIES_MAPPING.get(str(col).strip(), f"GPR_{col}")
            rows.append(
                {
                    "series_id": series_id,
                    "observation_date": observation_date,
                    "value": float(val),
                }
            )
    return pd.DataFrame(rows)


def collect_gpr(settings: Settings) -> tuple[Path, Path]:
    """Download the GPR daily and monthly Excel files and save as raw snapshots."""
    snapshot_at = utc_now()
    daily_content = _download_excel(GPR_DAILY_URL)
    monthly_content = _download_excel(GPR_MONTHLY_URL)

    daily_path = raw_snapshot_path(
        settings,
        source="gpr",
        stem="daily",
        snapshot_at=snapshot_at,
    )
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.with_suffix(".xls").write_bytes(daily_content)

    monthly_path = raw_snapshot_path(
        settings,
        source="gpr",
        stem="monthly",
        snapshot_at=snapshot_at,
    )
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_path.with_suffix(".xls").write_bytes(monthly_content)

    logger.info(
        "Collected GPR daily (%d bytes) and monthly (%d bytes).",
        len(daily_content),
        len(monthly_content),
    )
    return daily_path.with_suffix(".xls"), monthly_path.with_suffix(".xls")


def transform_gpr(settings: Settings) -> list[Path]:
    """Transform raw GPR Excel files into warehouse Parquet partitions.

    The output conforms to the ``macro_vintages`` schema so GPR data joins
    naturally with FRED/ALFRED data in downstream queries.
    """
    raw_root = settings.raw_root / "gpr"
    if not raw_root.exists():
        logger.info("No GPR raw data found — skipping transform.")
        return []

    all_rows: list[pd.DataFrame] = []
    for xls_path in sorted(raw_root.rglob("*.xls")):
        content = xls_path.read_bytes()
        try:
            if "daily" in xls_path.stem.lower():
                parsed = _parse_daily_gpr(content)
            else:
                parsed = _parse_monthly_gpr(content)
            if not parsed.empty:
                all_rows.append(parsed)
        except Exception as exc:
            logger.warning("Failed to parse GPR file %s: %s", xls_path, exc)

    if not all_rows:
        return []

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["series_id", "observation_date"], keep="last")
    combined["observation_date"] = pd.to_datetime(combined["observation_date"])

    # Add macro_vintages-compatible columns
    known_at = utc_now()
    combined["realtime_start"] = combined["observation_date"]
    combined["realtime_end"] = pd.Timestamp("9999-12-31")
    combined["known_at"] = known_at
    combined["source"] = "gpr_caldara_iacoviello"

    output_paths: list[Path] = []
    for series_id, group in combined.groupby("series_id"):
        partition_date = group["observation_date"].max().date()
        out_path = warehouse_partition_path(
            settings,
            domain="macro/gpr",
            partition_date=partition_date,
            stem=str(series_id).lower(),
        )
        write_parquet(group, out_path)
        output_paths.append(out_path)

    logger.info(
        "Transformed %d GPR series into %d partition files.",
        combined["series_id"].nunique(),
        len(output_paths),
    )
    return output_paths

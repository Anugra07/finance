from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.utils.dates import utc_now
from ai_analyst.utils.io import read_json, write_json, write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import (
    raw_snapshot_path,
    snapshot_time_from_path,
    warehouse_partition_path,
)

logger = logging.getLogger(__name__)


def _normalize_cik(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits.zfill(10) if digits else ""


def _load_constituent_tables(url: str) -> list[pd.DataFrame]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "local-ai-analyst universe collector",
            "Accept": "text/html,application/xhtml+xml",
        }
    )
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        logger.warning(
            "SSL verification failed for %s. Retrying with certificate verification disabled.",
            url,
        )
        response = session.get(url, timeout=60, verify=False)
        response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def latest_sp500_tickers(
    settings: Settings,
    *,
    limit: int | None = None,
    include_benchmark: bool = True,
) -> list[str]:
    conn = connect(settings)
    try:
        frame = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    ticker,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_at DESC) AS row_num
                FROM universe_membership
                WHERE ticker IS NOT NULL
                  AND TRIM(ticker) <> ''
            )
            SELECT ticker
            FROM ranked
            WHERE row_num = 1
            ORDER BY ticker
            """
        ).df()
    finally:
        conn.close()

    tickers = frame["ticker"].astype(str).str.upper().tolist() if not frame.empty else []
    if limit is not None:
        tickers = tickers[:limit]
    if include_benchmark and "SPY" not in tickers:
        tickers.append("SPY")
    return tickers


def collect_sp500_constituents(settings: Settings, *, snapshot_at: datetime | None = None) -> Path:
    snapshot_at = snapshot_at or utc_now()
    tables = _load_constituent_tables(settings.sp500_constituents_url)
    if not tables:
        raise ValueError("No tables returned from S&P 500 source.")
    df = tables[0]
    payload = df.to_dict(orient="records")
    path = raw_snapshot_path(
        settings,
        source="sp500_constituents",
        stem="sp500_current",
        snapshot_at=snapshot_at,
    )
    write_json(path, payload)
    return path


def transform_sp500_constituents(settings: Settings) -> list[Path]:
    root = settings.raw_root / "sp500_constituents"
    if not root.exists():
        return []

    outputs: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        transform_loaded_at = utc_now()
        df = pd.DataFrame(payload).rename(
            columns={
                "Symbol": "ticker",
                "Security": "security",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
                "CIK": "cik",
            }
        )
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df["cik"] = df["cik"].map(_normalize_cik)
        df["as_of_date"] = snapshot_at.date()
        df["snapshot_at"] = snapshot_at
        df["source_snapshot"] = str(path)
        df["transform_loaded_at"] = transform_loaded_at
        df = df[
            [
                "as_of_date",
                "ticker",
                "security",
                "sector",
                "sub_industry",
                "cik",
                "snapshot_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ]
        out_path = warehouse_partition_path(
            settings,
            domain="universe/sp500_current",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(df, out_path)
        outputs.append(out_path)
    return outputs

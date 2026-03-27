from __future__ import annotations

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.sources.universe import latest_sp500_tickers, transform_sp500_constituents
from ai_analyst.utils.io import write_json, write_parquet
from ai_analyst.warehouse.database import connect, refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_latest_sp500_tickers_returns_latest_unique_names(tmp_path) -> None:
    settings = _settings(tmp_path)
    universe = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-03-31"),
                "ticker": "AAPL",
                "security": "Apple Inc.",
                "sector": "Information Technology",
                "sub_industry": "Technology Hardware",
                "cik": "0000320193",
                "snapshot_at": pd.Timestamp("2024-03-31T00:00:00Z"),
                "source_snapshot": "older.json",
                "transform_loaded_at": pd.Timestamp("2024-03-31T00:05:00Z"),
            },
            {
                "as_of_date": pd.Timestamp("2024-04-30"),
                "ticker": "AAPL",
                "security": "Apple Inc.",
                "sector": "Information Technology",
                "sub_industry": "Technology Hardware",
                "cik": "0000320193",
                "snapshot_at": pd.Timestamp("2024-04-30T00:00:00Z"),
                "source_snapshot": "newer.json",
                "transform_loaded_at": pd.Timestamp("2024-04-30T00:05:00Z"),
            },
            {
                "as_of_date": pd.Timestamp("2024-04-30"),
                "ticker": "MSFT",
                "security": "Microsoft Corp.",
                "sector": "Information Technology",
                "sub_industry": "Systems Software",
                "cik": "0000789019",
                "snapshot_at": pd.Timestamp("2024-04-30T00:00:00Z"),
                "source_snapshot": "newer.json",
                "transform_loaded_at": pd.Timestamp("2024-04-30T00:05:00Z"),
            },
        ]
    )
    write_parquet(
        universe,
        warehouse_partition_path(
            settings,
            domain="universe/sp500_current",
            partition_date=pd.Timestamp("2024-04-30").date(),
            stem="universe",
        ),
    )
    refresh_views(settings)

    tickers = latest_sp500_tickers(settings, include_benchmark=True)
    limited = latest_sp500_tickers(settings, limit=1, include_benchmark=False)
    limited_with_benchmark = latest_sp500_tickers(settings, limit=1, include_benchmark=True)

    assert tickers == ["AAPL", "MSFT", "SPY"]
    assert limited == ["AAPL"]
    assert limited_with_benchmark == ["AAPL", "SPY"]


def test_transform_sp500_constituents_normalizes_cik(tmp_path) -> None:
    settings = _settings(tmp_path)
    raw_root = settings.raw_root / "sp500_constituents" / "2024-04-30"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / "sp500_current_20240430T000000Z.json"
    write_json(
        raw_path,
        [
            {
                "Symbol": "AAPL",
                "Security": "Apple Inc.",
                "GICS Sector": "Information Technology",
                "GICS Sub-Industry": "Technology Hardware",
                "CIK": 320193,
            }
        ],
    )

    outputs = transform_sp500_constituents(settings)
    refresh_views(settings)

    assert outputs
    conn = connect(settings)
    try:
        cik = conn.execute(
            "select cik from universe_membership where ticker = 'AAPL' limit 1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert cik == "0000320193"

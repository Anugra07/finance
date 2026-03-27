from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path
from ai_analyst.warehouse.snapshot_builder import SnapshotBuilder


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_snapshot_builder_respects_as_of(tmp_path) -> None:
    settings = _settings(tmp_path)
    macro = pd.DataFrame(
        [
            {
                "series_id": "DGS10",
                "observation_date": pd.Timestamp("2024-01-02"),
                "value": 4.1,
                "realtime_start": pd.Timestamp("2024-01-03"),
                "realtime_end": pd.Timestamp("2024-01-09"),
                "known_at": pd.Timestamp("2024-01-03T12:00:00Z"),
                "source_snapshot": "macro_a.json",
                "transform_loaded_at": pd.Timestamp("2024-01-03T12:05:00Z"),
            },
            {
                "series_id": "DGS10",
                "observation_date": pd.Timestamp("2024-01-02"),
                "value": 4.4,
                "realtime_start": pd.Timestamp("2024-01-10"),
                "realtime_end": pd.Timestamp("9999-12-31"),
                "known_at": pd.Timestamp("2024-01-10T12:00:00Z"),
                "source_snapshot": "macro_b.json",
                "transform_loaded_at": pd.Timestamp("2024-01-10T12:05:00Z"),
            },
        ]
    )
    companyfacts = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "entity_name": "Apple Inc.",
                "taxonomy": "us-gaap",
                "metric_name": "Assets",
                "unit": "USD",
                "value": 100.0,
                "period_end": pd.Timestamp("2023-12-31"),
                "filing_date": pd.Timestamp("2024-01-05"),
                "frame": None,
                "fy": 2024,
                "fp": "Q1",
                "form": "10-Q",
                "snapshot_at": pd.Timestamp("2024-01-06T00:00:00Z"),
                "source_snapshot": "facts_a.json",
                "transform_loaded_at": pd.Timestamp("2024-01-06T00:10:00Z"),
            },
            {
                "cik": "0000320193",
                "entity_name": "Apple Inc.",
                "taxonomy": "us-gaap",
                "metric_name": "Assets",
                "unit": "USD",
                "value": 120.0,
                "period_end": pd.Timestamp("2023-12-31"),
                "filing_date": pd.Timestamp("2024-01-12"),
                "frame": None,
                "fy": 2024,
                "fp": "Q1",
                "form": "10-Q",
                "snapshot_at": pd.Timestamp("2024-01-12T00:00:00Z"),
                "source_snapshot": "facts_b.json",
                "transform_loaded_at": pd.Timestamp("2024-01-12T00:10:00Z"),
            },
        ]
    )
    submissions = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "company_name": "Apple Inc.",
                "accession_number": "000032019324000001",
                "form": "10-Q",
                "filing_date": pd.Timestamp("2024-01-05"),
                "acceptance_datetime": pd.Timestamp("2024-01-05T20:00:00Z"),
                "primary_document": "aapl10q.htm",
                "primary_doc_description": "10-Q",
                "snapshot_at": pd.Timestamp("2024-01-06T00:00:00Z"),
                "source_snapshot": "submissions_a.json",
                "transform_loaded_at": pd.Timestamp("2024-01-06T00:10:00Z"),
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2024-01-04"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_volume": 1000.0,
                "volume": 1000.0,
                "div_cash": 0.0,
                "split_factor": 1.0,
                "known_at": pd.Timestamp("2024-01-04T21:00:00Z"),
                "source_snapshot": "prices_a.json",
                "transform_loaded_at": pd.Timestamp("2024-01-05T00:00:00Z"),
            }
        ]
    )
    universe = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-01-06"),
                "ticker": "AAPL",
                "security": "Apple Inc.",
                "sector": "Information Technology",
                "sub_industry": "Technology Hardware",
                "cik": "0000320193",
                "snapshot_at": pd.Timestamp("2024-01-06T00:00:00Z"),
                "source_snapshot": "sp500.json",
                "transform_loaded_at": pd.Timestamp("2024-01-06T00:10:00Z"),
            }
        ]
    )

    write_parquet(
        macro,
        warehouse_partition_path(
            settings,
            domain="macro/vintages",
            partition_date=macro["known_at"].dt.date.iloc[0],
            stem="macro",
        ),
    )
    write_parquet(
        companyfacts,
        warehouse_partition_path(
            settings,
            domain="edgar/companyfacts",
            partition_date=companyfacts["snapshot_at"].dt.date.iloc[0],
            stem="companyfacts",
        ),
    )
    write_parquet(
        submissions,
        warehouse_partition_path(
            settings,
            domain="edgar/submissions",
            partition_date=submissions["snapshot_at"].dt.date.iloc[0],
            stem="submissions",
        ),
    )
    write_parquet(
        prices,
        warehouse_partition_path(
            settings,
            domain="prices/daily",
            partition_date=prices["known_at"].dt.date.iloc[0],
            stem="prices",
        ),
    )
    write_parquet(
        universe,
        warehouse_partition_path(
            settings,
            domain="universe/sp500_current",
            partition_date=universe["snapshot_at"].dt.date.iloc[0],
            stem="universe",
        ),
    )

    refresh_views(settings)
    bundle = SnapshotBuilder(settings).build(as_of=datetime(2024, 1, 8, tzinfo=UTC))

    assert bundle.macro.iloc[0]["value"] == 4.1
    assert bundle.companyfacts.iloc[0]["value"] == 100.0
    assert bundle.submissions.iloc[0]["accession_number"] == "000032019324000001"
    assert bundle.prices.iloc[0]["ticker"] == "AAPL"
    assert bundle.universe.iloc[0]["sector"] == "Information Technology"

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.sources.sec import (
    latest_universe_ciks,
    latest_v1_universe_ciks,
    transform_companyfacts,
    transform_submissions,
)
from ai_analyst.utils.io import write_json, write_parquet
from ai_analyst.warehouse.database import refresh_views
from ai_analyst.warehouse.layout import (
    raw_snapshot_path,
    warehouse_partition_path,
)


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_sec_transforms_and_universe_ciks(tmp_path) -> None:
    settings = _settings(tmp_path)
    snapshot_at = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)

    universe = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-05-31"),
                "ticker": "AAPL",
                "security": "Apple Inc.",
                "sector": "Information Technology",
                "sub_industry": "Technology Hardware",
                "cik": "320193",
                "snapshot_at": pd.Timestamp("2024-05-31T00:00:00Z"),
                "source_snapshot": "universe.json",
                "transform_loaded_at": pd.Timestamp("2024-05-31T00:05:00Z"),
            },
            {
                "as_of_date": pd.Timestamp("2024-05-31"),
                "ticker": "MSFT",
                "security": "Microsoft Corp.",
                "sector": "Information Technology",
                "sub_industry": "Systems Software",
                "cik": "789019",
                "snapshot_at": pd.Timestamp("2024-05-31T00:00:00Z"),
                "source_snapshot": "universe.json",
                "transform_loaded_at": pd.Timestamp("2024-05-31T00:05:00Z"),
            },
        ]
    )
    write_parquet(
        universe,
        warehouse_partition_path(
            settings,
            domain="universe/sp500_current",
            partition_date=snapshot_at.date(),
            stem="universe",
        ),
    )
    refresh_views(settings)

    ciks = latest_universe_ciks(settings)
    assert ciks == ["0000320193", "0000789019"]

    submissions_payload = {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000001"],
                "filingDate": ["2024-05-02"],
                "acceptanceDateTime": ["2024-05-02T20:15:00.000Z"],
                "form": ["10-Q"],
                "primaryDocument": ["aapl-20240330x10q.htm"],
                "primaryDocDescription": ["10-Q"],
            }
        },
    }
    companyfacts_payload = {
        "cik": "0000320193",
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "end": "2024-03-30",
                                "val": 15_550_000_000,
                                "fy": 2024,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2024-05-02",
                                "frame": None,
                            }
                        ]
                    }
                }
            }
        },
    }
    write_json(
        raw_snapshot_path(
            settings,
            source="sec_submissions",
            stem="cik0000320193",
            snapshot_at=snapshot_at,
        ),
        submissions_payload,
    )
    write_json(
        raw_snapshot_path(
            settings,
            source="sec_companyfacts",
            stem="cik0000320193",
            snapshot_at=snapshot_at,
        ),
        companyfacts_payload,
    )

    submissions_outputs, filing_index_outputs = transform_submissions(settings)
    companyfacts_outputs = transform_companyfacts(settings)

    assert submissions_outputs
    assert filing_index_outputs
    assert companyfacts_outputs

    submissions_frame = pd.read_parquet(submissions_outputs[0])
    filing_index_frame = pd.read_parquet(filing_index_outputs[0])
    companyfacts_frame = pd.read_parquet(companyfacts_outputs[0])

    assert submissions_frame.iloc[0]["accession_number"] == "000032019324000001"
    assert submissions_frame.iloc[0]["company_name"] == "Apple Inc."
    assert filing_index_frame.iloc[0]["primary_document"] == "aapl-20240330x10q.htm"
    assert companyfacts_frame.iloc[0]["metric_name"] == "CommonStockSharesOutstanding"
    assert float(companyfacts_frame.iloc[0]["value"]) == 15_550_000_000.0


def test_latest_v1_universe_ciks(tmp_path) -> None:
    settings = _settings(tmp_path)
    snapshot_at = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    v1 = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-05-31"),
                "ticker": "AAPL",
                "security": "Apple Inc.",
                "sector": "Information Technology",
                "sub_industry": "Technology Hardware",
                "cik": "320193",
                "snapshot_at": pd.Timestamp("2024-05-31T00:00:00Z"),
                "source_snapshot": "v1.json",
                "transform_loaded_at": pd.Timestamp("2024-05-31T00:05:00Z"),
            },
            {
                "as_of_date": pd.Timestamp("2024-05-31"),
                "ticker": "MSFT",
                "security": "Microsoft Corp.",
                "sector": "Information Technology",
                "sub_industry": "Systems Software",
                "cik": "789019",
                "snapshot_at": pd.Timestamp("2024-05-31T00:00:00Z"),
                "source_snapshot": "v1.json",
                "transform_loaded_at": pd.Timestamp("2024-05-31T00:05:00Z"),
            },
        ]
    )
    write_parquet(
        v1,
        warehouse_partition_path(
            settings,
            domain="universe/v1_top150",
            partition_date=snapshot_at.date(),
            stem="v1",
        ),
    )
    refresh_views(settings)

    assert latest_v1_universe_ciks(settings) == ["0000320193", "0000789019"]

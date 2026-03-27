from __future__ import annotations

from datetime import UTC

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.features.engineering import (
    _attach_geo_context_features,
    build_feature_and_label_frames,
    materialize_v1_universe,
)
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
        v1_universe_size=2,
    )


def test_feature_engineering_builds_sector_features_and_labels(tmp_path) -> None:
    settings = _settings(tmp_path)
    trade_dates = pd.bdate_range("2024-01-01", periods=80, tz=UTC)
    tickers = ["AAPL", "MSFT", "SPY"]
    rows = []
    for idx, ticker in enumerate(tickers, start=1):
        for offset, trade_dt in enumerate(trade_dates, start=1):
            base = 100 + idx * 10 + offset
            rows.append(
                {
                    "ticker": ticker,
                    "date": trade_dt.date(),
                    "open": base,
                    "high": base + 1,
                    "low": base - 1,
                    "close": base + 0.5,
                    "adj_close": base + 0.5,
                    "adj_open": base,
                    "adj_high": base + 1,
                    "adj_low": base - 1,
                    "adj_volume": 1_000_000 + offset * 1000,
                    "volume": 1_000_000 + offset * 1000,
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                    "known_at": trade_dt,
                    "source_snapshot": "prices.json",
                    "transform_loaded_at": trade_dt,
                }
            )
    prices = pd.DataFrame(rows)
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
                "source_snapshot": "sp500.json",
                "transform_loaded_at": pd.Timestamp("2024-03-31T00:00:00Z"),
            },
            {
                "as_of_date": pd.Timestamp("2024-03-31"),
                "ticker": "MSFT",
                "security": "Microsoft Corp.",
                "sector": "Information Technology",
                "sub_industry": "Systems Software",
                "cik": "0000789019",
                "snapshot_at": pd.Timestamp("2024-03-31T00:00:00Z"),
                "source_snapshot": "sp500.json",
                "transform_loaded_at": pd.Timestamp("2024-03-31T00:00:00Z"),
            },
        ]
    )
    shares = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "entity_name": "Apple Inc.",
                "taxonomy": "us-gaap",
                "metric_name": "CommonStockSharesOutstanding",
                "unit": "shares",
                "value": 10_000_000.0,
                "period_end": pd.Timestamp("2024-03-31"),
                "filing_date": pd.Timestamp("2024-03-31"),
                "frame": None,
                "fy": 2024,
                "fp": "Q1",
                "form": "10-Q",
                "snapshot_at": pd.Timestamp("2024-03-31T00:00:00Z"),
                "source_snapshot": "facts.json",
                "transform_loaded_at": pd.Timestamp("2024-03-31T00:00:00Z"),
            },
            {
                "cik": "0000789019",
                "entity_name": "Microsoft Corp.",
                "taxonomy": "us-gaap",
                "metric_name": "CommonStockSharesOutstanding",
                "unit": "shares",
                "value": 8_000_000.0,
                "period_end": pd.Timestamp("2024-03-31"),
                "filing_date": pd.Timestamp("2024-03-31"),
                "frame": None,
                "fy": 2024,
                "fp": "Q1",
                "form": "10-Q",
                "snapshot_at": pd.Timestamp("2024-03-31T00:00:00Z"),
                "source_snapshot": "facts.json",
                "transform_loaded_at": pd.Timestamp("2024-03-31T00:00:00Z"),
            },
        ]
    )

    write_parquet(
        prices,
        warehouse_partition_path(
            settings, domain="prices/daily", partition_date=trade_dates[-1].date(), stem="prices"
        ),
    )
    write_parquet(
        universe,
        warehouse_partition_path(
            settings,
            domain="universe/sp500_current",
            partition_date=pd.Timestamp("2024-03-31").date(),
            stem="universe",
        ),
    )
    write_parquet(
        shares,
        warehouse_partition_path(
            settings,
            domain="edgar/companyfacts",
            partition_date=pd.Timestamp("2024-03-31").date(),
            stem="shares",
        ),
    )

    refresh_views(settings)
    materialize_v1_universe(settings)
    refresh_views(settings)
    features, labels = build_feature_and_label_frames(settings)

    assert not features.empty
    assert not labels.empty
    assert "ret_1d_sector_pct" in features.columns
    assert "share_turnover" in features.columns
    assert "sector_context_shock" in features.columns
    assert "oil_supply_risk_1d" in features.columns
    assert "event_dispersion_score" in features.columns
    assert "analog_failure_risk" in features.columns
    assert "pricing_divergence_score" in features.columns
    assert "causal_chain_count" in features.columns
    assert labels["excess_alpha_rank"].between(0, 1).all()
    assert set(features["ticker"].unique()) == {"AAPL", "MSFT"}


def test_geo_context_merge_keeps_canonical_columns() -> None:
    features = pd.DataFrame(
        [
            {
                "ticker": "XOM",
                "date": pd.Timestamp("2026-03-25"),
                "sector": "Energy",
            }
        ]
    )
    theme_daily = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-25"),
                "theme": "oil_supply_risk",
                "intensity": 2.5,
                "avg_novelty": 0.3,
                "event_dispersion_score": 0.2,
            }
        ]
    )
    sector_exposures = pd.DataFrame(
        [
            {
                "sector": "Energy",
                "theme": "oil_supply_risk",
                "exposure": 0.9,
            }
        ]
    )
    stock_exposures = pd.DataFrame(
        [
            {
                "ticker": "XOM",
                "theme": "oil_supply_risk",
                "exposure": 0.1,
            }
        ]
    )

    enriched = _attach_geo_context_features(
        features,
        theme_daily=theme_daily,
        sector_exposures=sector_exposures,
        stock_exposures=stock_exposures,
    )

    assert "oil_supply_risk_1d" in enriched.columns
    assert "commodity_shock_score" in enriched.columns
    assert "sector_context_shock" in enriched.columns
    assert "stock_context_shock" in enriched.columns
    assert "analog_failure_risk" in enriched.columns
    assert "pricing_divergence_score" in enriched.columns
    assert not any(column.endswith("_x") or column.endswith("_y") for column in enriched.columns)
    assert enriched.loc[0, "oil_supply_risk_1d"] == 2.5
    assert enriched.loc[0, "sector_context_shock"] == 2.25
    assert enriched.loc[0, "stock_context_shock"] == 2.5

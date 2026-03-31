from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.shortlist.engine import build_shortlist
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
        small_account_default_capital=5000.0,
        small_account_default_currency="INR",
    )


def test_build_shortlist_returns_india_candidates_with_primary_pick(tmp_path) -> None:
    settings = _settings(tmp_path)
    trade_date = pd.Timestamp("2026-03-30")
    loaded_at = pd.Timestamp("2026-03-30T10:00:00Z")

    predictions = pd.DataFrame(
        [
            {
                "date": trade_date,
                "ticker": "RELIANCE",
                "market_code": "IN",
                "sector": "Energy",
                "horizon_days": 21,
                "prediction_context": "live_latest",
                "prediction": 0.88,
                "prob_outperform": 0.88,
                "confidence_score": 0.76,
                "observed_excess_alpha": None,
                "observed_excess_alpha_rank": None,
                "split_no": 0,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            },
            {
                "date": trade_date,
                "ticker": "INFY",
                "market_code": "IN",
                "sector": "Technology",
                "horizon_days": 21,
                "prediction_context": "live_latest",
                "prediction": 0.70,
                "prob_outperform": 0.70,
                "confidence_score": 0.40,
                "observed_excess_alpha": None,
                "observed_excess_alpha_rank": None,
                "split_no": 0,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            },
            {
                "date": trade_date,
                "ticker": "RELIANCE",
                "market_code": "IN",
                "sector": "Energy",
                "horizon_days": 5,
                "prediction_context": "live_latest",
                "prediction": 0.72,
                "prob_outperform": 0.72,
                "confidence_score": 0.44,
                "observed_excess_alpha": None,
                "observed_excess_alpha_rank": None,
                "split_no": 0,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            },
            {
                "date": trade_date,
                "ticker": "INFY",
                "market_code": "IN",
                "sector": "Technology",
                "horizon_days": 5,
                "prediction_context": "live_latest",
                "prediction": 0.42,
                "prob_outperform": 0.42,
                "confidence_score": 0.16,
                "observed_excess_alpha": None,
                "observed_excess_alpha_rank": None,
                "split_no": 0,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            },
        ]
        + [
            {
                "date": pd.Timestamp("2026-02-01") + pd.Timedelta(days=idx),
                "ticker": f"CAL{idx}",
                "market_code": "IN",
                "sector": "Energy",
                "horizon_days": 21,
                "prediction_context": "walkforward_test",
                "prediction": 0.55 + (idx * 0.02),
                "prob_outperform": 0.55 + (idx * 0.02),
                "confidence_score": 0.20 + (idx * 0.05),
                "observed_excess_alpha": 0.01 + (idx * 0.002),
                "observed_excess_alpha_rank": 0.50 + (idx * 0.03),
                "split_no": 1,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            }
            for idx in range(6)
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "market_code": "IN",
                "country_code": "IN",
                "exchange_code": "NSE",
                "currency": "INR",
                "instrument_type": "equity",
                "tradeable": True,
                "symbol_native": "RELIANCE",
                "symbol_vendor": "RELIANCE.NS",
                "date": trade_date.date(),
                "open": 2400.0,
                "high": 2430.0,
                "low": 2390.0,
                "close": 2420.0,
                "adj_close": 2420.0,
                "adj_open": 2400.0,
                "adj_high": 2430.0,
                "adj_low": 2390.0,
                "adj_volume": 1_000_000.0,
                "volume": 1_000_000.0,
                "div_cash": 0.0,
                "split_factor": 1.0,
                "known_at": loaded_at,
                "source_snapshot": "prices.json",
                "transform_loaded_at": loaded_at,
            },
            {
                "ticker": "INFY",
                "market_code": "IN",
                "country_code": "IN",
                "exchange_code": "NSE",
                "currency": "INR",
                "instrument_type": "equity",
                "tradeable": True,
                "symbol_native": "INFY",
                "symbol_vendor": "INFY.NS",
                "date": trade_date.date(),
                "open": 1800.0,
                "high": 1820.0,
                "low": 1790.0,
                "close": 1810.0,
                "adj_close": 1810.0,
                "adj_open": 1800.0,
                "adj_high": 1820.0,
                "adj_low": 1790.0,
                "adj_volume": 900_000.0,
                "volume": 900_000.0,
                "div_cash": 0.0,
                "split_factor": 1.0,
                "known_at": loaded_at,
                "source_snapshot": "prices.json",
                "transform_loaded_at": loaded_at,
            },
        ]
    )
    security_master = pd.DataFrame(
        [
            {
                "ticker": "RELIANCE",
                "market_code": "IN",
                "country_code": "IN",
                "exchange_code": "NSE",
                "currency": "INR",
                "instrument_type": "equity",
                "tradeable": True,
                "symbol_native": "RELIANCE",
                "symbol_vendor": "RELIANCE.NS",
                "security": "Reliance Industries Ltd",
                "sector": "Energy",
                "sub_industry": "Integrated Oil & Gas",
                "cik": "",
                "security_group": "nifty200",
                "listing_date": pd.Timestamp("1995-11-29").date(),
                "snapshot_at": loaded_at,
                "source_snapshot": "master.json",
                "transform_loaded_at": loaded_at,
            },
            {
                "ticker": "INFY",
                "market_code": "IN",
                "country_code": "IN",
                "exchange_code": "NSE",
                "currency": "INR",
                "instrument_type": "equity",
                "tradeable": True,
                "symbol_native": "INFY",
                "symbol_vendor": "INFY.NS",
                "security": "Infosys Ltd",
                "sector": "Technology",
                "sub_industry": "IT Services",
                "cik": "",
                "security_group": "nifty200",
                "listing_date": pd.Timestamp("1993-06-14").date(),
                "snapshot_at": loaded_at,
                "source_snapshot": "master.json",
                "transform_loaded_at": loaded_at,
            },
        ]
    )
    feature_rows = pd.DataFrame(
        [
            {
                "date": trade_date.date(),
                "ticker": "RELIANCE",
                "market_code": "IN",
                "sector": "Energy",
                "cross_asset_aggregate_confirmation": 0.8,
                "pricing_discipline_market_confirmation": 0.75,
                "trade_readiness_timing_quality": 0.8,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            },
            {
                "date": trade_date.date(),
                "ticker": "INFY",
                "market_code": "IN",
                "sector": "Technology",
                "cross_asset_aggregate_confirmation": 0.25,
                "pricing_discipline_market_confirmation": 0.2,
                "trade_readiness_timing_quality": 0.3,
                "known_at": loaded_at,
                "transform_loaded_at": loaded_at,
            },
        ]
    )
    themes = pd.DataFrame(
        [
            {
                "date": trade_date.date(),
                "theme": "oil_supply_risk",
                "intensity": 1.4,
                "event_count": 2,
                "avg_severity": 0.6,
                "avg_novelty": 0.4,
                "event_dispersion_score": 0.2,
                "latest_event_time": loaded_at,
                "transform_loaded_at": loaded_at,
            }
        ]
    )
    sectors = pd.DataFrame(
        [
            {
                "as_of_date": trade_date.date(),
                "sector": "Energy",
                "sector_score": 1.1,
                "context_shock": 0.8,
                "finance_score": 0.6,
                "rank_desc": 1,
                "rank_asc": 2,
                "top_theme": "oil_supply_risk",
                "supporting_themes": ["oil_supply_risk"],
                "solution_bucket": "beneficiary",
                "transform_loaded_at": loaded_at,
            }
        ]
    )
    benchmark_metrics = pd.DataFrame(
        [
            {
                "as_of_date": trade_date.date(),
                "market_code": "IN",
                "horizon_days": 21,
                "strategy_name": "nifty200_momentum30_style",
                "metric_name": "rank_ic",
                "metric_value": 0.2,
                "source": "walkforward_style_baseline",
                "transform_loaded_at": loaded_at,
            },
            {
                "as_of_date": trade_date.date(),
                "market_code": "IN",
                "horizon_days": 21,
                "strategy_name": "nifty200_momentum30_style",
                "metric_name": "hit_rate",
                "metric_value": 0.5,
                "source": "walkforward_style_baseline",
                "transform_loaded_at": loaded_at,
            },
            {
                "as_of_date": trade_date.date(),
                "market_code": "IN",
                "horizon_days": 21,
                "strategy_name": "nifty_alpha50_style",
                "metric_name": "rank_ic",
                "metric_value": 0.15,
                "source": "walkforward_style_baseline",
                "transform_loaded_at": loaded_at,
            },
            {
                "as_of_date": trade_date.date(),
                "market_code": "IN",
                "horizon_days": 21,
                "strategy_name": "nifty_alpha50_style",
                "metric_name": "hit_rate",
                "metric_value": 0.45,
                "source": "walkforward_style_baseline",
                "transform_loaded_at": loaded_at,
            },
        ]
    )

    for name, frame, domain in [
        ("predictions", predictions, "forecast/model_predictions"),
        ("prices", prices, "prices/daily"),
        ("security_master", security_master, "universe/security_master"),
        ("features", feature_rows, "features/daily"),
        ("themes", themes, "themes/daily"),
        ("sectors", sectors, "rankings/sector"),
        (
            "benchmark_metrics",
            benchmark_metrics,
            "forecast/benchmark_strategy_metrics",
        ),
    ]:
        write_parquet(
            frame,
            warehouse_partition_path(
                settings,
                domain=domain,
                partition_date=trade_date.date(),
                stem=name,
            ),
        )

    refresh_views(settings)

    payload = build_shortlist(
        settings,
        as_of=datetime(2026, 3, 30, 10, 0, tzinfo=UTC),
        market_scope="IN",
        capital=5000.0,
        base_currency="INR",
        target_horizon_days=21,
        max_candidates=3,
    )

    assert payload["system_mode"] == "shortlist"
    assert payload["market_scope"] == "IN"
    assert len(payload["shortlist"]) == 2
    assert payload["shortlist"][0]["ticker"] == "RELIANCE"
    assert payload["shortlist"][0]["decision_mode"] == "actionable_bullish"
    assert payload["shortlist"][0]["pricing_layer"]["pricing_status"] == "confirmed"
    assert payload["shortlist"][0]["cost_tradability"]["suggested_share_count"] >= 2
    assert payload["primary_candidate"]["ticker"] == "RELIANCE"
    assert payload["market_summary"]["benchmark_gate_passed"] is True
    assert "failed_timing_gate" in payload["shortlist"][1]["downgrade_reasons"]
    assert payload["shortlist"][1]["decision_mode"] == "watch"

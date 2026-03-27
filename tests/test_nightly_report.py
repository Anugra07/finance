from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.reporting.nightly import build_ranked_report
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_nightly_report_includes_geo_energy_sections(tmp_path) -> None:
    settings = _settings(tmp_path)
    as_of = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)

    theme_daily = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-03"),
                "theme": "oil_supply_risk",
                "intensity": 1.2,
                "event_count": 2,
                "avg_severity": 0.85,
                "avg_novelty": 0.65,
                "event_dispersion_score": 0.4,
                "latest_event_time": as_of,
                "transform_loaded_at": as_of,
            },
            {
                "date": pd.Timestamp("2024-06-03"),
                "theme": "shipping_stress",
                "intensity": 0.7,
                "event_count": 1,
                "avg_severity": 0.7,
                "avg_novelty": 0.55,
                "event_dispersion_score": 0.3,
                "latest_event_time": as_of,
                "transform_loaded_at": as_of,
            },
        ]
    )
    normalized_events = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "event_time": as_of,
                "ingest_time": as_of,
                "source": "test",
                "topic": "Refinery outage in exporter",
                "event_family": "refinery_outage",
                "theme": "oil_supply_risk",
                "region": "Middle East",
                "geography": "Middle East",
                "severity": 0.9,
                "confidence": 0.8,
                "novelty": 0.7,
                "duration_hours": 48.0,
                "market_relevance": 0.95,
                "affected_commodities": ["oil"],
                "affected_sectors": ["Energy"],
                "affected_entities": ["OPEC"],
                "raw_ref": "refinery",
                "transform_loaded_at": as_of,
            }
        ]
    )
    sector_rankings = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-06-03"),
                "sector": "Energy",
                "sector_score": 1.1,
                "context_shock": 0.9,
                "finance_score": 0.4,
                "rank_desc": 1,
                "rank_asc": 2,
                "top_theme": "oil_supply_risk",
                "supporting_themes": ["oil_supply_risk"],
                "solution_bucket": "beneficiary",
                "transform_loaded_at": as_of,
            },
            {
                "as_of_date": pd.Timestamp("2024-06-03"),
                "sector": "Consumer Discretionary",
                "sector_score": -0.8,
                "context_shock": -0.7,
                "finance_score": -0.2,
                "rank_desc": 2,
                "rank_asc": 1,
                "top_theme": "oil_supply_risk",
                "supporting_themes": ["oil_supply_risk", "shipping_stress"],
                "solution_bucket": "pressured",
                "transform_loaded_at": as_of,
            },
        ]
    )
    solutions = pd.DataFrame(
        [
            {
                "theme": "oil_supply_risk",
                "solution_type": "beneficiary",
                "label": "Domestic producers and oilfield services",
                "beneficiary_sector": "Energy",
                "hedge_role": "inflation_hedge",
                "rationale": "Oil stress benefits domestic producers.",
                "source": "hand_coded_v1",
                "transform_loaded_at": as_of,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-03"),
                "ticker": "XOM",
                "sector": "Energy",
                "prediction": 0.88,
                "prob_outperform": 0.88,
                "confidence_score": 0.76,
                "excess_alpha_5d": 0.04,
                "split_no": 1,
            },
            {
                "date": pd.Timestamp("2024-06-03"),
                "ticker": "DAL",
                "sector": "Consumer Discretionary",
                "prediction": 0.21,
                "prob_outperform": 0.21,
                "confidence_score": 0.58,
                "excess_alpha_5d": -0.03,
                "split_no": 1,
            },
        ]
    )

    write_parquet(
        theme_daily,
        warehouse_partition_path(
            settings,
            domain="themes/daily",
            partition_date=as_of.date(),
            stem="theme_daily",
        ),
    )
    write_parquet(
        normalized_events,
        warehouse_partition_path(
            settings,
            domain="events/normalized",
            partition_date=as_of.date(),
            stem="normalized_events",
        ),
    )
    write_parquet(
        sector_rankings,
        warehouse_partition_path(
            settings,
            domain="rankings/sector",
            partition_date=as_of.date(),
            stem="sector_rankings",
        ),
    )
    write_parquet(
        solutions,
        warehouse_partition_path(
            settings,
            domain="solutions/default",
            partition_date=as_of.date(),
            stem="solutions",
        ),
    )
    refresh_views(settings)

    report = build_ranked_report(predictions, as_of=as_of, settings=settings)

    assert report["geo_energy_condition_summary"]["top_condition_drivers"][0]["theme"] == (
        "oil_supply_risk"
    )
    assert report["theme_intensity_dashboard"][0]["theme"] == "oil_supply_risk"
    assert report["sector_boom_stress_rankings"]["geo_beneficiaries"][0]["sector"] == "Energy"
    assert report["beneficiary_hedge_solution_ideas"][0]["label"] == (
        "Domestic producers and oilfield services"
    )
    assert report["stock_ranking_within_top_sectors"][0]["ticker"] == "XOM"
    assert "calibration_and_analog_summary" in report

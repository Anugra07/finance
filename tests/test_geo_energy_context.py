from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.events.context import LocalMacroContextSource
from ai_analyst.events.exposures import seed_geo_energy_reference_data
from ai_analyst.events.sector_opportunity import materialize_sector_rankings
from ai_analyst.events.theme_intensity import materialize_theme_intensity_tables
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect, refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_geo_energy_context_builds_theme_intensity_and_sector_rankings(tmp_path) -> None:
    settings = _settings(tmp_path)
    as_of = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)

    normalized_events = pd.DataFrame(
        [
            {
                "event_id": "evt_oil",
                "event_time": as_of,
                "ingest_time": as_of,
                "source": "test",
                "topic": "Refinery outage in key exporter",
                "event_family": "refinery_outage",
                "theme": "oil_supply_risk",
                "region": "ME",
                "geography": "Middle East",
                "severity": 0.9,
                "confidence": 0.8,
                "novelty": 0.7,
                "duration_hours": 72.0,
                "market_relevance": 0.9,
                "affected_commodities": ["oil"],
                "affected_sectors": ["Energy", "Consumer Discretionary"],
                "affected_entities": ["OPEC"],
                "raw_ref": "oil",
                "transform_loaded_at": as_of,
            },
            {
                "event_id": "evt_shipping",
                "event_time": as_of,
                "ingest_time": as_of,
                "source": "test",
                "topic": "Chokepoint disruption",
                "event_family": "shipping_disruption",
                "theme": "shipping_stress",
                "region": "Red Sea",
                "geography": "Red Sea",
                "severity": 0.8,
                "confidence": 0.9,
                "novelty": 0.6,
                "duration_hours": 48.0,
                "market_relevance": 0.85,
                "affected_commodities": ["oil", "lng"],
                "affected_sectors": ["Industrials", "Consumer Discretionary"],
                "affected_entities": ["Bab el-Mandeb"],
                "raw_ref": "shipping",
                "transform_loaded_at": as_of,
            },
        ]
    )
    features = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-03"),
                "ticker": "XOM",
                "sector": "Energy",
                "ret_5d": 0.04,
                "ret_20d": 0.06,
                "realized_vol_20d": 0.02,
            },
            {
                "date": pd.Timestamp("2024-06-03"),
                "ticker": "DAL",
                "sector": "Consumer Discretionary",
                "ret_5d": -0.03,
                "ret_20d": -0.01,
                "realized_vol_20d": 0.03,
            },
            {
                "date": pd.Timestamp("2024-06-03"),
                "ticker": "LMT",
                "sector": "Industrials",
                "ret_5d": 0.01,
                "ret_20d": 0.03,
                "realized_vol_20d": 0.02,
            },
        ]
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
        features,
        warehouse_partition_path(
            settings,
            domain="features/daily",
            partition_date=as_of.date(),
            stem="features",
        ),
    )

    seed_geo_energy_reference_data(settings)
    refresh_views(settings)
    hourly_paths, daily_paths = materialize_theme_intensity_tables(settings)
    refresh_views(settings)
    ranking_paths = materialize_sector_rankings(settings, as_of=as_of.date())
    refresh_views(settings)

    assert hourly_paths
    assert daily_paths
    assert ranking_paths

    context = LocalMacroContextSource(settings)
    theme_intensities = context.get_theme_intensities(as_of=as_of)
    sector_rankings = context.get_sector_rankings(as_of=as_of)
    solution_ideas = context.get_solution_ideas(as_of=as_of)
    conn = connect(settings)
    try:
        industry_exposure_count = conn.execute(
            "SELECT COUNT(*) FROM industry_theme_exposure"
        ).fetchone()[0]
        stock_exposure_count = conn.execute("SELECT COUNT(*) FROM stock_theme_exposure").fetchone()[
            0
        ]
    finally:
        conn.close()

    assert set(theme_intensities["theme"]) >= {"oil_supply_risk", "shipping_stress"}
    assert sector_rankings.iloc[0]["sector"] == "Energy"
    assert "Domestic producers and oilfield services" in solution_ideas["label"].tolist()
    assert industry_exposure_count > 0
    assert stock_exposure_count > 0

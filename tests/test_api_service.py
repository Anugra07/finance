from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.api.service import analyst_brief_payload, analyst_health_payload
from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import refresh_views
from ai_analyst.warehouse.layout import warehouse_partition_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_analyst_health_payload_reports_warehouse_counts(tmp_path) -> None:
    settings = _settings(tmp_path)
    refresh_views(settings)

    payload = analyst_health_payload(settings)

    assert payload["status"] == "ok"
    assert "warehouse_counts" in payload


def test_analyst_brief_payload_returns_report_shape(tmp_path) -> None:
    settings = _settings(tmp_path)
    as_of = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    theme_daily = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-03"),
                "theme": "oil_supply_risk",
                "intensity": 1.4,
                "event_count": 1,
                "avg_severity": 0.8,
                "avg_novelty": 0.5,
                "event_dispersion_score": 0.3,
                "latest_event_time": as_of,
                "transform_loaded_at": as_of,
            }
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
    refresh_views(settings)

    payload = analyst_brief_payload(settings, as_of=as_of)

    assert payload["as_of"].startswith("2024-06-03")
    assert "geo_energy_condition_summary" in payload


def test_analyst_brief_payload_prefers_latest_pointer(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    latest_path = settings.reports_path / "nightly_latest.json"
    latest_path.write_text(
        '{"as_of":"2024-06-04T20:00:00+00:00","top_stocks":[{"ticker":"XOM"}]}',
        encoding="utf-8",
    )

    payload = analyst_brief_payload(settings, as_of=datetime(2024, 6, 4, 20, 0, tzinfo=UTC))

    assert payload["as_of"].startswith("2024-06-04")
    assert payload["top_stocks"][0]["ticker"] == "XOM"

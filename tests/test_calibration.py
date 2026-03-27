from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.calibration.metrics import build_calibration_metrics
from ai_analyst.calibration.persistence import persist_decision_forecast
from ai_analyst.config import Settings
from ai_analyst.core.models import ContextPack


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_persist_decision_forecast_writes_outcomes_and_override_logs(tmp_path) -> None:
    settings = _settings(tmp_path)
    pack = ContextPack(
        ticker="XOM",
        as_of=datetime(2024, 6, 3, 20, 0, tzinfo=UTC),
        market_snapshot={"prices": [{"ticker": "XOM", "close": 100.0}]},
        macro_snapshot={"theme_intensities": [{"theme": "oil_supply_risk", "intensity": 1.2}]},
        top_events=[{"topic": "Refinery outage"}],
        mode="decision",
        causal_state={"regime": {"label": {"value": "geo_elevated"}}},
    )
    payload = {
        "ticker": "XOM",
        "horizon_verdicts": [{"horizon": "1-3 weeks", "verdict": "outperform", "confidence": 0.72}],
        "abstain": False,
        "invalidation_triggers": ["theme_intensity_fades"],
        "key_risks": ["Policy relief"],
        "critic": {
            "force_abstain": False,
            "confidence_adjustment": -0.1,
            "missing_evidence": ["No direct stock-level mediator confirmed."],
        },
        "research_packet": {"active_chains": ["oil_supply_risk -> supply_repricing -> brent"]},
    }

    outcome_paths, override_paths = persist_decision_forecast(
        settings,
        context_pack=pack,
        decision_output=payload,
    )

    assert outcome_paths
    assert override_paths


def test_build_calibration_metrics_computes_brier_and_buckets() -> None:
    outcomes = pd.DataFrame(
        [
            {
                "forecast_id": "a",
                "ticker": "XOM",
                "as_of": pd.Timestamp("2024-06-03T20:00:00Z"),
                "mode": "decision",
                "horizon": "1-3 weeks",
                "direction": "up",
                "confidence": 0.8,
                "abstain": False,
            },
            {
                "forecast_id": "b",
                "ticker": "DAL",
                "as_of": pd.Timestamp("2024-06-03T20:00:00Z"),
                "mode": "decision",
                "horizon": "1-3 weeks",
                "direction": "down",
                "confidence": 0.7,
                "abstain": False,
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-06-03"), "ticker": "XOM", "excess_alpha_5d": 0.04},
            {"date": pd.Timestamp("2024-06-03"), "ticker": "DAL", "excess_alpha_5d": -0.02},
        ]
    )

    metrics = build_calibration_metrics(outcomes, labels)

    assert not metrics.empty
    assert {"brier_score", "hit_rate", "abstention_rate"} <= set(metrics["metric_name"])

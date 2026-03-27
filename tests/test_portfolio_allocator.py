from __future__ import annotations

import json

from ai_analyst.config import Settings
from ai_analyst.portfolio.allocator import build_rebalance_plan


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_build_rebalance_plan_from_latest_report(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_path / "nightly_ranked_report_2024-06-03.json"
    report_path.write_text(
        json.dumps(
            {
                "top_stocks": [
                    {
                        "ticker": "XOM",
                        "sector": "Energy",
                        "expected_excess_alpha_rank": 0.82,
                        "confidence_score": 0.7,
                    },
                    {
                        "ticker": "LMT",
                        "sector": "Industrials",
                        "expected_excess_alpha_rank": 0.74,
                        "confidence_score": 0.6,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_rebalance_plan(settings)

    assert payload["status"] == "ok"
    assert payload["allocations"]


def test_build_rebalance_plan_accepts_nightly_latest_pointer(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_path / "nightly_latest.json"
    report_path.write_text(
        json.dumps(
            {
                "top_stocks": [
                    {
                        "ticker": "XOM",
                        "sector": "Energy",
                        "expected_excess_alpha_rank": 0.81,
                        "confidence_score": 0.65,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_rebalance_plan(settings)

    assert payload["status"] == "ok"
    assert payload["report_path"].endswith("nightly_latest.json")

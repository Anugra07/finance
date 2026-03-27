from __future__ import annotations

from datetime import UTC, datetime

from ai_analyst.core.models import ContextPack
from ai_analyst.llm.forecast import run_two_pass_forecast


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(model)
        if "Forecast JSON" in prompt:
            return {
                "strongest_counterarguments": ["Macro data is stale."],
                "missing_data_checks": ["No SHAP drivers were available."],
                "disconfirming_evidence": ["Theme intensity is mixed."],
                "confidence_adjustment": 15,
            }
        return {
            "summary": "Energy remains supported over the 5-day window.",
            "thesis": "Oil stress and positive sector ranking support XOM.",
            "drivers": ["oil_supply_risk", "Energy sector ranking"],
            "risks": ["Policy relief", "Weakening macro demand"],
            "confidence": 70,
            "horizon": "Short-Term",
            "verdict": "Strong Buy",
            "citations": ["https://example.com/not-allowed"],
        }


def test_two_pass_forecast_merges_forecast_and_critic() -> None:
    pack = ContextPack(
        ticker="XOM",
        as_of=datetime(2024, 6, 3, 20, 0, tzinfo=UTC),
        market_snapshot={"prices": [{"ticker": "XOM", "close": 100.0}]},
        macro_snapshot={"theme_intensities": [{"theme": "oil_supply_risk", "intensity": 1.2}]},
        top_events=[{"topic": "Refinery outage", "theme": "oil_supply_risk"}],
        freshness_flags={"price_rows": 1, "event_rows": 1},
        sector_rankings=[{"sector": "Energy", "sector_score": 1.0}],
        solution_ideas=[{"label": "Domestic producers and oilfield services"}],
    )

    client = FakeOllamaClient()
    output = run_two_pass_forecast(
        pack,
        forecast_model="llama3.2",
        critic_model="llama3.2",
        client=client,
    )

    assert output["ticker"] == "XOM"
    assert output["forecast"]["verdict"] == "outperform"
    assert output["forecast"]["confidence"] == 0.7
    assert output["forecast"]["horizon"] == "5 trading days"
    assert output["forecast"]["citations"] == ["context_pack"]
    assert output["critic"]["strongest_counterarguments"] == ["Macro data is stale."]
    assert output["final_confidence"] == 0.55
    assert output["abstain_reasons"] == []
    assert client.calls == ["llama3.2", "llama3.2"]


def test_two_pass_forecast_abstains_when_price_data_is_stale() -> None:
    pack = ContextPack(
        ticker="AAPL",
        as_of=datetime(2026, 3, 25, 20, 0, tzinfo=UTC),
        market_snapshot={"prices": []},
        macro_snapshot={"theme_intensities": []},
        top_events=[],
        freshness_flags={"price_is_stale": True, "theme_is_stale": False},
    )

    client = FakeOllamaClient()
    output = run_two_pass_forecast(
        pack,
        forecast_model="llama3.2",
        critic_model="llama3.2",
        client=client,
    )

    assert output["forecast"]["verdict"] == "low_conviction"
    assert output["final_confidence"] <= 0.35
    assert output["abstain_reasons"] == ["price_data_stale"]

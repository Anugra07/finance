from __future__ import annotations

from datetime import UTC, datetime

from ai_analyst.core.models import ContextPack
from ai_analyst.llm.reasoning import run_decision_mode, run_research_mode


class FakeReasoningClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, object],
    ) -> dict[str, object]:
        self.prompts.append(prompt)
        schema_props = set((response_schema.get("properties") or {}).keys())
        if "critic_veto" in schema_props:
            return {
                "critic_veto": False,
                "force_abstain": False,
                "forced_mode_change": "decision",
                "confidence_adjustment": -0.1,
                "added_risks": ["Mediator confirmation is weak."],
                "missing_evidence": ["No direct stock-level mediator confirmed."],
                "critic_reason_codes": [],
            }
        if "decision_summary" in schema_props:
            return {
                "decision_summary": "Confirmed energy pressure with moderate follow-through.",
                "horizon_verdicts": [
                    {"horizon": "1-3 days", "verdict": "watch", "confidence": 0.6},
                    {"horizon": "1-3 weeks", "verdict": "outperform", "confidence": 0.7},
                ],
                "conviction": 0.72,
                "abstain": False,
                "invalidation_triggers": ["theme_intensity_fades"],
                "key_risks": ["Policy relief"],
                "model_disagreement_flags": ["geo_signal_outpaces_market_confirmation"],
            }
        if "base_case" in schema_props:
            return {
                "base_case": "Energy risk is elevated and transmission is constructive for XOM.",
                "pricing_view": "Pricing confirms part of the thesis but not the full move yet.",
                "active_chains": ["shipping_stress -> rerouting -> freight_rates -> Industrials"],
                "competing_hypotheses": [
                    {
                        "hypothesis": "shipping_stress_dominant_transmission",
                        "confidence": 0.68,
                        "status": "supported",
                        "why": "Theme intensity and sector context line up.",
                    }
                ],
                "unknowns": ["Crude confirmation remains mixed."],
                "missing_evidence": ["No direct company filing support."],
                "falsification_triggers": ["theme_intensity_fades"],
            }
        return {
            "judgments": ["Role summary"],
            "evidence_ids": ["evt::refinery"],
            "uncertainties": ["unknown_a"],
            "confidence": 0.7,
            "objection": "Need stronger direct confirmation.",
        }


def _context_pack() -> ContextPack:
    return ContextPack(
        ticker="XOM",
        as_of=datetime(2024, 6, 3, 20, 0, tzinfo=UTC),
        market_snapshot={"prices": [{"ticker": "XOM", "close": 100.0}]},
        macro_snapshot={"theme_intensities": [{"theme": "oil_supply_risk", "intensity": 1.4}]},
        top_events=[{"topic": "Refinery outage", "theme": "oil_supply_risk"}],
        sector_rankings=[{"sector": "Energy", "sector_score": 1.1}],
        solution_ideas=[{"label": "Domestic producers and oilfield services"}],
        causal_state={"regime": {"label": {"value": "geo_elevated"}}},
        causal_chains=[{"theme": "oil_supply_risk", "channel": "supply_repricing"}],
        analog_matches={"1-3 days": [{"analog_key": "theme_state:2024-01-15"}]},
        uncertainty_map={"analogs": "supported"},
        competing_hypotheses=[{"hypothesis": "oil_supply_risk_dominant_transmission"}],
        missing_evidence=["No direct stock-level mediator confirmed."],
        evidence_index={
            "evt::refinery": {
                "evidence_id": "evt::refinery",
                "source_type": "worldmonitor",
                "source_ref": "refinery-outage",
                "timestamp": datetime(2024, 6, 3, 18, 0, tzinfo=UTC),
                "reliability": 0.8,
                "freshness_class": "market",
                "content_hash": "evt123",
            },
            "price::XOM::2024-06-03": {
                "evidence_id": "price::XOM::2024-06-03",
                "source_type": "price",
                "source_ref": "XOM",
                "timestamp": datetime(2024, 6, 3, 20, 0, tzinfo=UTC),
                "reliability": 0.95,
                "freshness_class": "decision_critical",
                "content_hash": "px123",
            },
        },
        narrative_risk={
            "deception_risk": {"value": 0.2, "status": "supported"},
        },
        cross_asset_confirmation={
            "aggregate_confirmation": {"value": 0.75, "status": "supported"},
        },
        confidence_breakdown={
            "data_confidence": {"value": 0.8},
            "state_confidence": {"value": 0.7, "status": "supported"},
            "model_confidence": {"value": 0.7},
            "pricing_confidence": {"value": 0.7},
            "analog_confidence": {"value": 0.6},
            "decision_confidence_uncapped": 0.72,
            "decision_confidence": 0.72,
            "cap_reasons": [],
        },
        model_interpretation={"pricing_disagreement": {"divergence_score": 0.4}},
        version_metadata={"reasoning_schema_version": "v1"},
    )


def test_research_mode_returns_structured_packet() -> None:
    client = FakeReasoningClient()
    output = run_research_mode(
        _context_pack(),
        model="local-model",
        client=client,
    )

    assert output["ticker"] == "XOM"
    assert output["active_chains"]
    assert output["analogs_by_horizon"]["1-3 days"][0]["analog_key"] == "theme_state:2024-01-15"
    assert output["role_outputs"]
    assert any("You are NOT the source of truth." in prompt for prompt in client.prompts)
    assert output["fact_layer"]["source_quality"] == {}
    assert output["pricing_layer"]["trade_readiness"] == {}


def test_decision_mode_returns_horizon_verdicts_and_critic() -> None:
    client = FakeReasoningClient()
    output = run_decision_mode(
        _context_pack(),
        model="local-model",
        critic_model="local-model",
        client=client,
    )

    assert output["ticker"] == "XOM"
    assert output["horizon_verdicts"]
    assert output["conviction"] == 0.62
    assert output["critic"]["missing_evidence"] == ["No direct stock-level mediator confirmed."]
    assert "trust_tier_effect" in output["decision_layer"]
    assert any("Adversarial Critic" in prompt for prompt in client.prompts)

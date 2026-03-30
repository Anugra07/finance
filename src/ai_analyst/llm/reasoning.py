from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ai_analyst.causal.governance import apply_trust_tier_governance
from ai_analyst.causal.types import DecisionForecast, ResearchAnswer
from ai_analyst.core.models import ContextPack
from ai_analyst.llm.forecast import OllamaClient

ANALYST_EXECUTION_CORE = """
You are the analyst layer of a local geopolitical-energy-finance forecasting system.

You are NOT the source of truth.
You are NOT allowed to invent backend truth.
You are NOT a free-form market pundit.

You must reason only over the structured, local, evidence-backed ContextPack.

Non-negotiable rules:
- You must use only facts present in the ContextPack.
- You must cite only evidence_ids present in the evidence_index.
- You must not invent evidence, sources, events, prices, analogs, model outputs, or exposures.
- You must not fill packet gaps with generic finance knowledge.
- You must not smooth over missing fields or stale evidence.
- You must not treat analog summaries as proof.
- You must not convert moderate causal plausibility into tradeability.
- You must not convert weak narrative coherence into high confidence.
- You must keep fact, interpretation, pricing, decision, and falsification separate.
- You must default to research_only when decision quality is weak.
- You must respect freshness, trust tier, critic output, missing evidence, and downgrade rules.
- You must not imply an actionable directional recommendation outside decision_layer.
- If decision_layer.mode = research_only, all other layers must remain non-prescriptive.
- If a claim cannot be tied to at least one valid evidence_id in the evidence_index,
  you must not include it.
- You must return valid JSON only.
- You must not include markdown.
- You must not include prose before or after the JSON object.
- You must not include fields outside the required schema.

In every answer, you must output exactly these five layers:
1. fact_layer
2. interpretation_layer
3. pricing_layer
4. decision_layer
5. falsification_layer

If critic_veto is true, you must downgrade decision mode to research_only.
If pricing or decision evidence is stale, you must weaken or block pricing and decision claims.
If trust-tier rules block action, you must return decision_layer.mode = research_only.
If evidence is weak, conflicted, stale, or deceptive, you must say so plainly.

Keep the final answer concise, skeptical, auditable, and explicit about uncertainty.
""".strip()

ANALYST_ARCHITECTURE_APPENDIX = """
System context:
- Data layer reconstructs point-in-time truth.
- Causal layer provides normalized events, relations, themes, regimes, analogs, pricing discipline,
  cross-asset confirmation, trade readiness, and decomposed confidence.
- Quant layer provides ranking context and model interpretation.
- Analyst layer explains the packet and may be downgraded by the critic or trust gate.

Reasoning order:
1. facts
2. causal state
3. source and narrative quality
4. analog context
5. market confirmation
6. model context
7. actionability
8. uncertainty and falsification

Operational distinctions:
- PricingDisciplineState answers whether market behavior is confirming, late, overextended, or
  showing follow-through.
- TradeReadinessState answers whether the setup is actionable now given timing, liquidity, and
  risk/reward.
- CrossAssetConfirmationState must be used by family before aggregation: energy, vol, credit,
  rates, FX, sector-flow, and peers.
- Missing cross-asset confirmation weakens pricing confidence more than fact-layer interpretation.
""".strip()

ROLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "uncertainties": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "confidence": {"type": "number"},
        "objection": {"type": "string"},
    },
    "required": ["judgments", "evidence_ids", "uncertainties", "confidence", "objection"],
}

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_case": {"type": "string"},
        "pricing_view": {"type": "string"},
        "active_chains": {"type": "array", "items": {"type": "string"}},
        "competing_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "confidence": {"type": "number"},
                    "status": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["hypothesis", "confidence", "status", "why"],
            },
        },
        "unknowns": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "falsification_triggers": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "base_case",
        "pricing_view",
        "active_chains",
        "competing_hypotheses",
        "unknowns",
        "missing_evidence",
        "falsification_triggers",
    ],
}

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision_summary": {"type": "string"},
        "horizon_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "string"},
                    "verdict": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["horizon", "verdict", "confidence"],
            },
        },
        "conviction": {"type": "number"},
        "abstain": {"type": "boolean"},
        "invalidation_triggers": {"type": "array", "items": {"type": "string"}},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "model_disagreement_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "decision_summary",
        "horizon_verdicts",
        "conviction",
        "abstain",
        "invalidation_triggers",
        "key_risks",
        "model_disagreement_flags",
    ],
}

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "critic_veto": {"type": "boolean"},
        "force_abstain": {"type": "boolean"},
        "forced_mode_change": {"type": "string"},
        "confidence_adjustment": {"type": "number"},
        "added_risks": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "critic_reason_codes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "critic_veto",
        "force_abstain",
        "forced_mode_change",
        "confidence_adjustment",
        "added_risks",
        "missing_evidence",
        "critic_reason_codes",
    ],
}

ROLE_PROMPTS = {
    "geo": (
        "You are the Geo Analyst. Return only compact JSON judgments about regime, escalation, "
        "geography, and direct causal relevance. Never cite facts outside the evidence IDs in "
        "the context."
    ),
    "energy": (
        "You are the Energy Analyst. Return only compact JSON judgments about supply, "
        "chokepoints, and physical mediator risk."
    ),
    "market": (
        "You are the Market Analyst. Return only compact JSON judgments about pricing, "
        "cross-asset confirmation, and sector/flow response."
    ),
    "quant": (
        "You are the Quant Interpreter. Return only compact JSON judgments about ranking context, "
        "model disagreement, and analog support."
    ),
}

ROLE_DISCIPLINE = (
    "You must return schema-first output only. Each role may return at most 3 judgments, "
    "3 evidence_ids, 2 uncertainties, 1 confidence, and 1 objection. No essays."
)


def _compact_context(context_pack: ContextPack) -> str:
    def _compact_state(payload: Any) -> Any:
        if isinstance(payload, dict):
            compact: dict[str, Any] = {}
            for key, value in payload.items():
                if key in {"supporting_evidence", "version_metadata", "rows"}:
                    continue
                compact[key] = _compact_state(value)
            return compact
        if isinstance(payload, list):
            return [_compact_state(item) for item in payload[:3]]
        if isinstance(payload, float):
            return round(payload, 4)
        return payload

    market_prices = context_pack.market_snapshot.get("prices", [])
    macro_themes = context_pack.macro_snapshot.get("theme_intensities", [])
    evidence_index = {
        evidence_id: {
            "source_type": payload.get("source_type"),
            "timestamp": payload.get("timestamp"),
            "reliability": payload.get("reliability"),
            "freshness_class": payload.get("freshness_class"),
        }
        for evidence_id, payload in list(context_pack.evidence_index.items())[:12]
    }
    analog_matches = {
        horizon: [
            {
                "analog_key": match.get("analog_key"),
                "similarity_score": match.get("similarity_score"),
                "analogy_strength": match.get("analogy_strength"),
                "misleading_analogy": match.get("misleading_analogy"),
                "important_differences": match.get("important_differences"),
            }
            for match in matches[:2]
        ]
        for horizon, matches in context_pack.analog_matches.items()
    }
    top_events = [
        {
            "event_id": event.get("event_id"),
            "topic": event.get("topic"),
            "theme": event.get("theme"),
            "source": event.get("source"),
            "region": event.get("region"),
            "geography": event.get("geography"),
            "severity": event.get("severity"),
            "novelty": event.get("novelty"),
            "market_relevance": event.get("market_relevance"),
        }
        for event in context_pack.top_events[:5]
    ]
    theme_intensities = [
        {
            "theme": row.get("theme"),
            "intensity": round(float(row.get("intensity", 0.0) or 0.0), 4),
            "event_count": row.get("event_count"),
            "avg_severity": row.get("avg_severity"),
            "avg_novelty": row.get("avg_novelty"),
            "latest_event_time": row.get("latest_event_time"),
        }
        for row in macro_themes[:5]
    ]
    compact_state = {
        "regime": _compact_state(context_pack.causal_state.get("regime", {})),
        "themes": _compact_state(context_pack.causal_state.get("themes", {})),
        "transmission": _compact_state(context_pack.causal_state.get("transmission", {})),
        "mediators": _compact_state(context_pack.causal_state.get("mediators", {})),
        "exposure": _compact_state(context_pack.causal_state.get("exposure", {})),
        "horizon_views": _compact_state(context_pack.causal_state.get("horizon_views", [])),
        "missing_evidence": context_pack.causal_state.get("missing_evidence", [])[:5],
        "version_metadata": context_pack.causal_state.get("version_metadata", {}),
    }
    compact_payload = {
        "ticker": context_pack.ticker,
        "as_of": context_pack.as_of,
        "mode": context_pack.mode,
        "trust_tier": context_pack.trust_tier,
        "market_snapshot": {
            "prices": market_prices[:5],
        },
        "macro_snapshot": {
            "theme_intensities": theme_intensities,
        },
        "top_events": top_events,
        "freshness_flags": context_pack.freshness_flags,
        "sector_rankings": context_pack.sector_rankings[:5],
        "solution_ideas": context_pack.solution_ideas[:5],
        "causal_state": compact_state,
        "causal_chains": context_pack.causal_chains[:5],
        "analog_matches": analog_matches,
        "model_interpretation": _compact_state(context_pack.model_interpretation),
        "uncertainty_map": context_pack.uncertainty_map,
        "competing_hypotheses": context_pack.competing_hypotheses[:3],
        "missing_evidence": context_pack.missing_evidence[:5],
        "source_assessment": _compact_state(context_pack.source_assessment),
        "narrative_risk": _compact_state(context_pack.narrative_risk),
        "cross_asset_confirmation": _compact_state(context_pack.cross_asset_confirmation),
        "pricing_discipline": _compact_state(context_pack.pricing_discipline),
        "trade_readiness": _compact_state(context_pack.trade_readiness),
        "confidence_breakdown": _compact_state(context_pack.confidence_breakdown),
        "evidence_index": evidence_index,
        "version_metadata": context_pack.version_metadata,
    }
    return json.dumps(compact_payload, indent=2, default=str)


def _mode_directive(mode: str) -> str:
    if mode == "decision":
        return (
            "Decision mode rules: You may issue a directional decision only if evidence is fresh "
            "enough, pricing confirmation is adequate, critic logic does not veto, trust tier "
            "allows it, and confidence thresholds are met. Otherwise you must return "
            "decision_layer.mode = research_only."
        )
    return (
        "Research mode rules: You must emphasize active chains, analogs, missing evidence, "
        "competing hypotheses, and falsification. You must not force a trade recommendation."
    )


FAILURE_BEHAVIOR = """
Failure behavior:
- If the packet is sparse, you must lower confidence, surface missing evidence, and default to
  research_only.
- If evidence is stale, you may keep background explanation but you must weaken or block pricing
  and decision claims.
- If the model and causal story disagree, you must state the disagreement explicitly.
- If pricing confirmation is weak, you must not upgrade to a confident decision.
- Missing evidence lowers confidence and may force research_only.
- Conflicted evidence lowers conviction and may force research_only.
- Deceptive or signaling-heavy evidence weakens actionability more than background interpretation.
""".strip()

PIPELINE_STAGE_NOTE = """
This system runs four specialist analyst passes:
- Geo Analyst
- Energy Analyst
- Market Analyst
- Quant Interpreter

It then runs:
- Senior Synthesizer
- Adversarial Critic
""".strip()


def _normalize_confidence(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return round(max(0.0, min(1.0, numeric)), 4)


def _normalize_list(value: object, *, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        items = []
    else:
        text = str(value).strip()
        items = [text] if text else []
    return items[:limit] if limit is not None else items


def _allowed_evidence_ids(context_pack: ContextPack) -> set[str]:
    return set(context_pack.evidence_index.keys())


def _sanitize_evidence_ids(
    context_pack: ContextPack,
    raw_ids: object,
    *,
    limit: int = 3,
) -> list[str]:
    allowed = _allowed_evidence_ids(context_pack)
    return [item for item in _normalize_list(raw_ids, limit=limit) if item in allowed]


def _layer_evidence_ids(context_pack: ContextPack, *, layer: str) -> list[str]:
    evidence_ids = list(context_pack.evidence_index.keys())
    if layer == "fact":
        return evidence_ids[:3]
    if layer == "interpretation":
        return [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id.startswith("evt::") or evidence_id.startswith("theme::")
        ][:3]
    if layer == "pricing":
        return [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id.startswith("price::")
        ][:3]
    if layer == "decision":
        pricing_ids = _layer_evidence_ids(context_pack, layer="pricing")
        return (pricing_ids + _layer_evidence_ids(context_pack, layer="interpretation"))[:3]
    return evidence_ids[:3]


def _run_role(
    *,
    client: OllamaClient,
    model: str,
    role_name: str,
    context_pack: ContextPack,
) -> dict[str, Any]:
    prompt = (
        f"{ANALYST_EXECUTION_CORE}\n\n"
        f"{_mode_directive(context_pack.mode)}\n"
        f"{FAILURE_BEHAVIOR}\n\n"
        f"{PIPELINE_STAGE_NOTE}\n\n"
        f"{ANALYST_ARCHITECTURE_APPENDIX}\n\n"
        f"{ROLE_PROMPTS[role_name]}\n"
        f"{ROLE_DISCIPLINE}\n"
        "Use only the context pack. Return strict JSON. Keep each field compact.\n\n"
        f"Context pack:\n{_compact_context(context_pack)}"
    )
    payload = client.generate_json(model=model, prompt=prompt, response_schema=ROLE_SCHEMA)
    return {
        "role": role_name,
        "judgments": _normalize_list(payload.get("judgments"), limit=3),
        "evidence_ids": _sanitize_evidence_ids(
            context_pack,
            payload.get("evidence_ids"),
            limit=3,
        ),
        "uncertainties": _normalize_list(payload.get("uncertainties"), limit=2),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "objection": str(payload.get("objection") or "").strip(),
    }


def _build_fact_layer(context_pack: ContextPack) -> dict[str, Any]:
    source_summary = (
        context_pack.source_assessment.get("summary")
        if isinstance(context_pack.source_assessment, dict)
        else {}
    ) or {}
    narrative_summary = (
        context_pack.narrative_risk.get("summary")
        if isinstance(context_pack.narrative_risk, dict)
        else {}
    ) or {}
    return {
        "summary": (
            f"{len(context_pack.top_events)} recent events and "
            f"{len(context_pack.sector_rankings)} sector ranks available."
        ),
        "events": context_pack.top_events[:3],
        "source_quality": source_summary,
        "narrative_caveat": narrative_summary,
        "freshness_flags": context_pack.freshness_flags,
        "evidence_ids": _layer_evidence_ids(context_pack, layer="fact"),
    }


def _build_interpretation_layer(
    context_pack: ContextPack,
    synthesis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "summary": str(synthesis.get("base_case") or "").strip(),
        "regime": context_pack.causal_state.get("regime", {}),
        "themes": context_pack.causal_state.get("themes", {}),
        "active_chains": synthesis.get("active_chains") or context_pack.causal_chains[:5],
        "analogs_by_horizon": context_pack.analog_matches,
        "competing_hypotheses": synthesis.get("competing_hypotheses")
        or context_pack.competing_hypotheses,
        "evidence_ids": _layer_evidence_ids(context_pack, layer="interpretation"),
    }


def _build_pricing_layer(
    context_pack: ContextPack,
    synthesis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "summary": str(synthesis.get("pricing_view") or "").strip(),
        "pricing_disagreement": context_pack.model_interpretation.get("pricing_disagreement", {}),
        "cross_asset_confirmation": context_pack.cross_asset_confirmation,
        "pricing_discipline": context_pack.pricing_discipline,
        "trade_readiness": context_pack.trade_readiness,
        "evidence_ids": _layer_evidence_ids(context_pack, layer="pricing"),
    }


def _build_falsification_layer(
    context_pack: ContextPack,
    synthesis: dict[str, Any],
) -> dict[str, Any]:
    triggers = _normalize_list(synthesis.get("falsification_triggers"), limit=5)
    if not triggers:
        triggers = [
            trigger
            for view in context_pack.causal_state.get("horizon_views", [])[:2]
            for trigger in view.get("invalidation_triggers", [])[:2]
        ]
    return {
        "triggers": triggers,
        "missing_evidence": (
            _normalize_list(synthesis.get("missing_evidence")) or context_pack.missing_evidence
        ),
        "unknowns": _normalize_list(synthesis.get("unknowns")) or context_pack.missing_evidence,
        "what_to_monitor_next": triggers[:3],
    }


def run_research_mode(
    context_pack: ContextPack,
    *,
    model: str,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    client = client or OllamaClient()
    role_outputs = [
        _run_role(client=client, model=model, role_name=role_name, context_pack=context_pack)
        for role_name in ("geo", "energy", "market", "quant")
    ]
    synthesis_prompt = (
        f"{ANALYST_EXECUTION_CORE}\n\n"
        f"{_mode_directive('research')}\n"
        f"{FAILURE_BEHAVIOR}\n\n"
        f"{PIPELINE_STAGE_NOTE}\n\n"
        f"{ANALYST_ARCHITECTURE_APPENDIX}\n\n"
        "You are the Senior Synthesizer. Separate fact, interpretation, pricing, decision, and "
        "falsification logic. Use only the context pack and role outputs. Return strict JSON for "
        "research mode. In research mode, decision_layer.mode must remain research_only unless a "
        "schema-required placeholder says otherwise. Do not output an actionable directional "
        "decision in research mode.\n\n"
        f"Context pack:\n{_compact_context(context_pack)}\n\n"
        f"Role outputs:\n{json.dumps(role_outputs, indent=2, default=str)}"
    )
    synthesis = client.generate_json(
        model=model,
        prompt=synthesis_prompt,
        response_schema=SYNTHESIS_SCHEMA,
    )

    fact_layer = _build_fact_layer(context_pack)
    interpretation_layer = _build_interpretation_layer(context_pack, synthesis)
    pricing_layer = _build_pricing_layer(context_pack, synthesis)
    falsification_layer = _build_falsification_layer(context_pack, synthesis)
    confidence_breakdown = context_pack.confidence_breakdown
    answer = ResearchAnswer(
        ticker=context_pack.ticker,
        as_of=context_pack.as_of,
        active_chains=[{"chain": item} for item in _normalize_list(synthesis.get("active_chains"))]
        or context_pack.causal_chains[:5],
        analogs_by_horizon=context_pack.analog_matches,
        uncertainty_map=context_pack.uncertainty_map,
        unknowns=_normalize_list(synthesis.get("unknowns")) or context_pack.missing_evidence,
        missing_evidence=_normalize_list(synthesis.get("missing_evidence"))
        or context_pack.missing_evidence,
        competing_hypotheses=synthesis.get("competing_hypotheses")
        or context_pack.competing_hypotheses,
        causal_state=context_pack.causal_state,
        fact_layer=fact_layer,
        interpretation_layer=interpretation_layer,
        pricing_layer=pricing_layer,
        decision_layer={"mode": "research_only", "reason": "default_research_mode"},
        falsification_layer=falsification_layer,
        confidence_breakdown=confidence_breakdown,
        trust_tier=context_pack.trust_tier,
        requested_mode="research",
        resolved_mode="research",
        downgrade_reason_category=None,
        critic_outcome={},
    )
    payload = asdict(answer)
    payload["role_outputs"] = role_outputs
    return payload


def run_decision_mode(
    context_pack: ContextPack,
    *,
    model: str,
    critic_model: str | None = None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    client = client or OllamaClient()
    research = run_research_mode(context_pack, model=model, client=client)
    decision_prompt = (
        f"{ANALYST_EXECUTION_CORE}\n\n"
        f"{_mode_directive('decision')}\n"
        f"{FAILURE_BEHAVIOR}\n\n"
        f"{PIPELINE_STAGE_NOTE}\n\n"
        f"{ANALYST_ARCHITECTURE_APPENDIX}\n\n"
        "You are the Senior Synthesizer in decision mode. Keep pricing logic separate from "
        "causal plausibility, and decision logic separate from explanatory elegance. Use only "
        "the context pack and research packet. Return strict JSON. If causal interpretation, "
        "model context, and pricing confirmation materially disagree, you must default toward "
        "research_only unless the packet explicitly resolves the disagreement. You must apply "
        "critic and trust-tier downgrade behavior exactly.\n\n"
        f"Context pack:\n{_compact_context(context_pack)}\n\n"
        f"Research packet:\n{json.dumps(research, indent=2, default=str)}"
    )
    decision = client.generate_json(
        model=model,
        prompt=decision_prompt,
        response_schema=DECISION_SCHEMA,
    )
    critic_prompt = (
        f"{ANALYST_EXECUTION_CORE}\n\n"
        f"{FAILURE_BEHAVIOR}\n\n"
        f"{PIPELINE_STAGE_NOTE}\n\n"
        f"{ANALYST_ARCHITECTURE_APPENDIX}\n\n"
        "You are the Adversarial Critic. Attack the conclusion. Identify stale evidence, weak "
        "pricing confirmation, misleading analogs, regime fragility, and poor actionability. "
        "Use only the context pack and decision packet. Return strict JSON that can veto, "
        "downgrade, or lower confidence. You must return explicit reason codes for any veto, "
        "downgrade, or confidence reduction.\n\n"
        f"Context pack:\n{_compact_context(context_pack)}\n\n"
        f"Decision packet:\n{json.dumps(decision, indent=2, default=str)}"
    )
    critic = client.generate_json(
        model=critic_model or model,
        prompt=critic_prompt,
        response_schema=CRITIC_SCHEMA,
    )

    horizon_verdicts = []
    for row in decision.get("horizon_verdicts") or []:
        horizon_verdicts.append(
            {
                "horizon": str(row.get("horizon") or ""),
                "verdict": str(row.get("verdict") or "low_conviction"),
                "confidence": _normalize_confidence(row.get("confidence")),
            }
        )
    decision_confidence_base = dict(context_pack.confidence_breakdown)
    uncapped = float(decision_confidence_base.get("decision_confidence_uncapped", 0.0) or 0.0)
    capped = float(decision_confidence_base.get("decision_confidence", 0.0) or 0.0)
    critic_adjustment = min(0.0, float(critic.get("confidence_adjustment", 0.0) or 0.0))
    adjusted_confidence = round(max(0.0, min(1.0, capped + critic_adjustment)), 4)
    decision_confidence_base["decision_confidence_uncapped"] = uncapped
    decision_confidence_base["decision_confidence"] = adjusted_confidence

    pricing_evidence_ids = _layer_evidence_ids(context_pack, layer="pricing")
    decision_evidence_ids = _layer_evidence_ids(context_pack, layer="decision")
    state_payload = context_pack.causal_state if isinstance(context_pack.causal_state, dict) else {}
    governance_confidence_breakdown = dict(
        state_payload.get("confidence_breakdown") or context_pack.confidence_breakdown
    )
    governance_confidence_breakdown["decision_confidence_uncapped"] = uncapped
    governance_confidence_breakdown["decision_confidence"] = adjusted_confidence
    governance = apply_trust_tier_governance(
        requested_mode="decision",
        trust_tier=context_pack.trust_tier,
        confidence_breakdown=governance_confidence_breakdown,
        narrative_risk=state_payload.get("narrative_risk") or context_pack.narrative_risk,
        cross_asset_confirmation=state_payload.get("cross_asset_confirmation")
        or context_pack.cross_asset_confirmation,
        critic_veto=bool(critic.get("critic_veto")) or bool(critic.get("force_abstain")),
        critic_reason_codes=_normalize_list(critic.get("critic_reason_codes")),
        pricing_evidence_ids=pricing_evidence_ids,
        decision_evidence_ids=decision_evidence_ids,
        evidence_index=context_pack.evidence_index,
        as_of=context_pack.as_of,
    )

    fact_layer = dict(research["fact_layer"])
    interpretation_layer = dict(research["interpretation_layer"])
    pricing_layer = dict(research["pricing_layer"])
    pricing_layer["evidence_ids"] = pricing_evidence_ids
    pricing_layer["pricing_summary"] = str(decision.get("decision_summary") or "").strip()

    if governance["resolved_mode"] == "research":
        decision_layer = {
            "mode": "research_only",
            "reason": governance["downgrade_reason_category"] or "forced_mode_change",
            "trust_tier_effect": (
                f"Requested decision mode was downgraded under {context_pack.trust_tier} rules."
            ),
        }
    else:
        decision_layer = {
            "mode": "decision",
            "summary": str(decision.get("decision_summary") or "").strip(),
            "horizon_verdicts": horizon_verdicts,
            "ranking_context": {
                "sector_rankings": context_pack.sector_rankings[:5],
                "model_interpretation": context_pack.model_interpretation,
            },
            "model_disagreement_flags": _normalize_list(
                decision.get("model_disagreement_flags"),
            ),
            "trust_tier_effect": (
                f"Decision mode remained allowed under {context_pack.trust_tier} rules."
            ),
            "evidence_ids": decision_evidence_ids,
        }

    falsification_layer = dict(research["falsification_layer"])
    falsification_layer["key_risks"] = _normalize_list(
        decision.get("key_risks"),
        limit=5,
    ) + _normalize_list(critic.get("added_risks"), limit=5)
    critic_outcome = {
        "critic_veto": bool(critic.get("critic_veto")) or bool(critic.get("force_abstain")),
        "confidence_adjustment": critic_adjustment,
        "critic_reason_codes": governance["critic_reason_codes"],
        "forced_mode_change": governance["forced_mode_change"],
        "missing_evidence": _normalize_list(critic.get("missing_evidence")),
    }
    forecast = DecisionForecast(
        ticker=context_pack.ticker,
        as_of=context_pack.as_of,
        horizon_verdicts=horizon_verdicts,
        conviction=governance["decision_confidence"],
        abstain=governance["resolved_mode"] != "decision" or bool(decision.get("abstain")),
        invalidation_triggers=_normalize_list(decision.get("invalidation_triggers")),
        key_risks=falsification_layer["key_risks"],
        ranking_context={
            "sector_rankings": context_pack.sector_rankings[:5],
            "model_interpretation": context_pack.model_interpretation,
        },
        model_disagreement_flags=_normalize_list(decision.get("model_disagreement_flags")),
        causal_state=context_pack.causal_state,
        fact_layer=fact_layer,
        interpretation_layer=interpretation_layer,
        pricing_layer=pricing_layer,
        decision_layer=decision_layer,
        falsification_layer=falsification_layer,
        confidence_breakdown=decision_confidence_base,
        trust_tier=context_pack.trust_tier,
        requested_mode="decision",
        resolved_mode=str(governance["resolved_mode"]),
        downgrade_reason_category=governance["downgrade_reason_category"],
        critic_outcome=critic_outcome,
    )
    payload = asdict(forecast)
    payload["research_packet"] = research
    payload["critic"] = {
        "critic_veto": critic_outcome["critic_veto"],
        "force_abstain": critic_outcome["critic_veto"],
        "forced_mode_change": "research" if governance["forced_mode_change"] else "decision",
        "confidence_adjustment": critic_adjustment,
        "missing_evidence": critic_outcome["missing_evidence"],
        "critic_reason_codes": governance["critic_reason_codes"],
    }
    payload["override_applied"] = bool(critic_outcome["critic_veto"]) or bool(
        governance["forced_mode_change"]
    )
    return payload

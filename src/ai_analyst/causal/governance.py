from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ai_analyst.causal.types import (
    CausalValue,
    ConfidenceBreakdown,
    CrossAssetConfirmationState,
    NarrativeRiskState,
    PricingDisciplineState,
    SourceAssessmentState,
    TradeReadinessState,
)
from ai_analyst.core.models import TrustTier


class GovernanceConfigError(ValueError):
    pass


def _load_yaml_asset(name: str) -> dict[str, Any]:
    with (
        importlib.resources.files("ai_analyst.causal")
        .joinpath(f"assets/{name}")
        .open("r", encoding="utf-8") as handle
    ):
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise GovernanceConfigError(f"Invalid YAML asset: {name}")
    return payload


def load_source_profiles() -> dict[str, Any]:
    payload = _load_yaml_asset("source_profiles.yaml")
    if "profiles" not in payload or "default" not in payload:
        raise GovernanceConfigError("source_profiles.yaml must define profiles and default")
    return payload


def load_narrative_rules() -> dict[str, Any]:
    payload = _load_yaml_asset("narrative_intent_rules.yaml")
    if "rules" not in payload or "default" not in payload:
        raise GovernanceConfigError("narrative_intent_rules.yaml must define rules and default")
    return payload


def load_trust_tiers() -> dict[str, Any]:
    payload = _load_yaml_asset("trust_tiers.yaml")
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict):
        raise GovernanceConfigError("trust_tiers.yaml must define tiers")
    for key in ("experimental", "paper", "trusted"):
        if key not in tiers:
            raise GovernanceConfigError(f"Missing trust tier: {key}")
    return payload


def load_evidence_freshness() -> dict[str, Any]:
    payload = _load_yaml_asset("evidence_freshness.yaml")
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise GovernanceConfigError("evidence_freshness.yaml must define classes")
    for key in ("background", "context", "market", "decision_critical"):
        if key not in classes:
            raise GovernanceConfigError(f"Missing evidence freshness class: {key}")
    return payload


def _status_from_score(score: float) -> str:
    if score >= 0.7:
        return "supported"
    if score >= 0.45:
        return "weakly_supported"
    if score <= 0.2:
        return "missing"
    return "unsupported"


def _signed_status(value: float, confidence: float) -> str:
    if confidence <= 0.2:
        return "missing"
    if abs(value) <= 0.1 and confidence >= 0.45:
        return "conflicted"
    return _status_from_score(confidence)


def _clip(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _grade_to_numeric(grade: str) -> float:
    return {"A": 0.95, "B": 0.8, "C": 0.65, "D": 0.45, "F": 0.2}.get(grade.upper(), 0.5)


def _numeric_to_grade(value: float) -> str:
    if value >= 0.9:
        return "A"
    if value >= 0.75:
        return "B"
    if value >= 0.6:
        return "C"
    if value >= 0.4:
        return "D"
    return "F"


def _freshness_score(*, event_time: datetime, ingest_time: datetime) -> float:
    delay_hours = max(0.0, (ingest_time - event_time).total_seconds() / 3600)
    if delay_hours <= 1:
        return 1.0
    if delay_hours <= 6:
        return 0.8
    if delay_hours <= 24:
        return 0.6
    if delay_hours <= 72:
        return 0.4
    return 0.2


def classify_event_freshness(record: dict[str, Any]) -> str:
    market_relevance = float(record.get("market_relevance", 0.0) or 0.0)
    severity = float(record.get("severity", 0.0) or 0.0)
    if market_relevance >= 0.85 or severity >= 0.85:
        return "decision_critical"
    if market_relevance >= 0.55:
        return "market"
    if market_relevance >= 0.25:
        return "context"
    return "background"


def build_event_governance_rows(
    *,
    source: str,
    record: dict[str, Any],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_profiles = load_source_profiles()
    narrative_rules = load_narrative_rules()

    source_profile = source_profiles["profiles"].get(source.lower(), source_profiles["default"])
    event_family = str(record.get("event_family") or "").strip().lower()
    rule = narrative_rules["rules"].get(event_family, narrative_rules["default"])

    event_time = pd.to_datetime(record.get("event_time"), utc=True, errors="coerce")
    ingest_time = pd.to_datetime(record.get("ingest_time"), utc=True, errors="coerce")
    if pd.isna(event_time):
        event_time = pd.Timestamp.now(tz=UTC)
    if pd.isna(ingest_time):
        ingest_time = pd.Timestamp.now(tz=UTC)

    freshness = _freshness_score(
        event_time=event_time.to_pydatetime(),
        ingest_time=ingest_time.to_pydatetime(),
    )
    source_row = {
        "event_id": record["event_id"],
        "source": source,
        "reliability_score": _clip(source_profile["reliability_score"]),
        "claim_verifiability": _clip(source_profile["claim_verifiability"]),
        "source_grade": str(source_profile["source_grade"]),
        "freshness_score": freshness,
        "provenance_quality": _clip(source_profile["provenance_quality"]),
        "assessment_summary": (
            f"{str(source_profile['source_grade']).upper()} source with "
            f"{round(_clip(source_profile['reliability_score']), 2)} reliability."
        ),
        "transform_loaded_at": record.get("transform_loaded_at"),
    }

    novelty = _clip(record.get("novelty", 0.0))
    confidence = _clip(record.get("confidence", 0.0))
    event_market_relevance = _clip(record.get("market_relevance", 0.0))
    repetition_decay = round(max(0.0, 1.0 - novelty), 4)
    actionability = round(
        min(1.0, (_clip(rule["actionability_score"]) + event_market_relevance + confidence) / 3.0),
        4,
    )
    deception_risk = round(
        min(
            1.0,
            _clip(rule["deception_risk"])
            * (1.0 - _clip(source_profile["reliability_score"]) * 0.35)
            + repetition_decay * 0.15,
        ),
        4,
    )
    narrative_row = {
        "event_id": record["event_id"],
        "source": source,
        "deception_risk": deception_risk,
        "signaling_vs_capability": round(_clip(rule["signaling_vs_capability"]), 4),
        "novelty_score": round(max(novelty, _clip(rule["novelty_score"]) * confidence), 4),
        "actionability_score": actionability,
        "repetition_decay": repetition_decay,
        "propaganda_risk": round(_clip(rule["propaganda_risk"]), 4),
        "risk_summary": (
            "Narrative appears mostly actionable."
            if actionability >= 0.6 and deception_risk < 0.45
            else "Narrative requires caution."
        ),
        "transform_loaded_at": record.get("transform_loaded_at"),
    }

    evidence_preview = str(record.get("topic") or "")[:200]
    evidence_payload = {
        "topic": record.get("topic"),
        "theme": record.get("theme"),
        "source": source,
        "event_time": event_time.isoformat(),
    }
    evidence_catalog = {
        "evidence_id": f"evt::{record['event_id']}",
        "event_id": record["event_id"],
        "source_type": source,
        "source_ref": str(record.get("raw_ref") or record["event_id"]),
        "timestamp": event_time.to_pydatetime(),
        "reliability": float(source_row["reliability_score"]),
        "freshness_class": classify_event_freshness(record),
        "content_hash": hashlib.sha1(
            json.dumps(evidence_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24],
        "content_preview": evidence_preview,
        "transform_loaded_at": record.get("transform_loaded_at"),
    }
    return source_row, narrative_row, evidence_catalog


def _weighted_average(frame: pd.DataFrame, column: str, weight: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    weights = pd.to_numeric(frame.get(weight), errors="coerce").fillna(0.0)
    if float(weights.sum()) <= 0:
        return float(numeric.mean() or 0.0)
    return float((numeric * weights).sum() / weights.sum())


def summarize_source_assessment(frame: pd.DataFrame) -> SourceAssessmentState:
    if frame.empty:
        missing = CausalValue(value=0.0, confidence=0.0, status="missing")
        return SourceAssessmentState(
            summary=missing,
            reliability_score=missing,
            claim_verifiability=missing,
            source_grade=CausalValue(value="unknown", confidence=0.0, status="missing"),
            freshness=missing,
            provenance_quality=missing,
        )

    reliability = _weighted_average(frame, "reliability_score", "freshness_score")
    verifiability = _weighted_average(frame, "claim_verifiability", "freshness_score")
    freshness = _weighted_average(frame, "freshness_score", "freshness_score")
    provenance = _weighted_average(frame, "provenance_quality", "freshness_score")
    confidence = min(1.0, max(0.0, (reliability + verifiability + freshness + provenance) / 4.0))
    status = _status_from_score(confidence)
    grade = _numeric_to_grade((reliability + provenance) / 2.0)
    summary = CausalValue(
        value=round(confidence, 4),
        confidence=round(confidence, 4),
        status=status,
        supporting_evidence=frame.head(3).to_dict(orient="records"),
    )
    return SourceAssessmentState(
        summary=summary,
        reliability_score=CausalValue(
            round(reliability, 4),
            confidence,
            _status_from_score(reliability),
        ),
        claim_verifiability=CausalValue(
            round(verifiability, 4),
            confidence,
            _status_from_score(verifiability),
        ),
        source_grade=CausalValue(
            grade,
            _grade_to_numeric(grade),
            _status_from_score(_grade_to_numeric(grade)),
        ),
        freshness=CausalValue(round(freshness, 4), confidence, _status_from_score(freshness)),
        provenance_quality=CausalValue(
            round(provenance, 4),
            confidence,
            _status_from_score(provenance),
        ),
    )


def summarize_narrative_risk(frame: pd.DataFrame) -> NarrativeRiskState:
    if frame.empty:
        missing = CausalValue(value=0.0, confidence=0.0, status="missing")
        return NarrativeRiskState(
            summary=missing,
            deception_risk=missing,
            signaling_vs_capability=missing,
            novelty_score=missing,
            actionability_score=missing,
            repetition_decay=missing,
            propaganda_risk=missing,
        )

    actionability = _weighted_average(frame, "actionability_score", "novelty_score")
    novelty = _weighted_average(frame, "novelty_score", "novelty_score")
    deception = _weighted_average(frame, "deception_risk", "novelty_score")
    signaling = _weighted_average(frame, "signaling_vs_capability", "novelty_score")
    propaganda = _weighted_average(frame, "propaganda_risk", "novelty_score")
    repetition = _weighted_average(frame, "repetition_decay", "novelty_score")
    confidence = min(1.0, max(0.0, (actionability + novelty + (1.0 - deception)) / 3.0))
    status = (
        "conflicted"
        if deception >= 0.65 and actionability >= 0.55
        else _status_from_score(confidence)
    )
    summary = CausalValue(
        value=round(actionability - deception, 4),
        confidence=round(confidence, 4),
        status=status,
        supporting_evidence=frame.head(3).to_dict(orient="records"),
    )
    return NarrativeRiskState(
        summary=summary,
        deception_risk=CausalValue(
            round(deception, 4),
            confidence,
            _status_from_score(1.0 - deception),
        ),
        signaling_vs_capability=CausalValue(
            round(signaling, 4),
            confidence,
            _signed_status(signaling - 0.5, confidence),
        ),
        novelty_score=CausalValue(round(novelty, 4), confidence, _status_from_score(novelty)),
        actionability_score=CausalValue(
            round(actionability, 4),
            confidence,
            _status_from_score(actionability),
        ),
        repetition_decay=CausalValue(
            round(repetition, 4),
            confidence,
            _status_from_score(1.0 - repetition),
        ),
        propaganda_risk=CausalValue(
            round(propaganda, 4),
            confidence,
            _status_from_score(1.0 - propaganda),
        ),
    )


def build_cross_asset_confirmation_state(
    *,
    macro: pd.DataFrame,
    prices: pd.DataFrame,
    sector_rankings: pd.DataFrame,
) -> CrossAssetConfirmationState:
    def _series_change(series_id: str) -> float:
        if macro.empty:
            return 0.0
        frame = macro.loc[macro["series_id"] == series_id].sort_values("observation_date")
        if len(frame) < 2:
            return 0.0
        values = pd.to_numeric(frame["value"], errors="coerce").dropna()
        if len(values) < 2:
            return 0.0
        return float(values.iloc[-1] - values.iloc[-2])

    def _proxy_move(tickers: list[str]) -> float:
        if prices.empty:
            return 0.0
        frame = prices.loc[prices["ticker"].isin(tickers)].copy()
        if frame.empty:
            return 0.0
        frame = frame.sort_values(["ticker", "date"])
        frame["ret_1d"] = frame.groupby("ticker")["adj_close"].pct_change()
        return float(frame["ret_1d"].tail(len(tickers) * 2).abs().mean() or 0.0)

    energy_move = max(
        abs(_series_change("DCOILWTICO")),
        abs(_series_change("DHHNGSP")),
        _proxy_move(["USO", "UNG", "XLE"]),
    )
    vol_move = max(abs(_series_change("VIXCLS")) / 10.0, _proxy_move(["SPY"]))
    credit_move = max(abs(_series_change("BAMLH0A0HYM2")) / 2.0, _proxy_move(["HYG"]))
    rates_move = max(abs(_series_change("DGS10")) / 2.0, abs(_series_change("T10Y2Y")) / 2.0)
    fx_move = abs(_series_change("DTWEXBGS")) / 5.0
    sector_flow = (
        float(sector_rankings["sector_score"].abs().head(5).mean() or 0.0) / 2.0
        if not sector_rankings.empty
        else 0.0
    )
    peer_move = _proxy_move(["XLE", "XLI", "XLY", "ITA", "SOXX", "JETS", "IYT"])
    aggregate = np.mean(
        [
            min(1.0, energy_move),
            min(1.0, vol_move),
            min(1.0, credit_move),
            min(1.0, rates_move),
            min(1.0, fx_move),
            min(1.0, sector_flow),
            min(1.0, peer_move),
        ]
    )

    def _cv(value: float) -> CausalValue:
        value = round(min(1.0, max(0.0, value)), 4)
        return CausalValue(value=value, confidence=value, status=_status_from_score(value))

    return CrossAssetConfirmationState(
        energy_confirmation=_cv(energy_move),
        vol_confirmation=_cv(vol_move),
        credit_confirmation=_cv(credit_move),
        rates_confirmation=_cv(rates_move),
        fx_confirmation=_cv(fx_move),
        sector_flow_confirmation=_cv(sector_flow),
        peer_confirmation=_cv(peer_move),
        aggregate_confirmation=_cv(float(aggregate)),
    )


def build_pricing_discipline_state(
    *,
    pricing_disagreement: Any,
    cross_asset_confirmation: CrossAssetConfirmationState,
    prices: pd.DataFrame,
) -> PricingDisciplineState:
    market_confirmation = min(
        1.0,
        max(
            0.0,
            (
                float(pricing_disagreement.market_response_strength.confidence)
                + float(cross_asset_confirmation.aggregate_confirmation.confidence)
            )
            / 2.0,
        ),
    )
    move_lateness = min(
        1.0,
        max(0.0, float(pricing_disagreement.crowdedness_proxy.value or 0.0) * 4.0),
    )
    overextension = min(
        1.0,
        max(0.0, float(pricing_disagreement.divergence_score.value or 0.0) / 2.5),
    )
    reaction_follow = min(
        1.0,
        max(0.0, float(pricing_disagreement.follow_through_status.confidence)),
    )
    summary_value = market_confirmation - (move_lateness * 0.4 + overextension * 0.4)
    summary_conf = min(1.0, max(0.0, (market_confirmation + reaction_follow) / 2.0))

    def _cv(value: float, confidence: float | None = None) -> CausalValue:
        value = round(float(value), 4)
        conf = round(float(value if confidence is None else confidence), 4)
        return CausalValue(
            value=value,
            confidence=max(0.0, min(1.0, conf)),
            status=_signed_status(value, conf),
        )

    return PricingDisciplineState(
        summary=_cv(summary_value, summary_conf),
        market_confirmation=_cv(market_confirmation),
        move_lateness=_cv(move_lateness),
        overextension=_cv(overextension),
        reaction_vs_follow_through=_cv(reaction_follow),
    )


def build_trade_readiness_state(
    *,
    source_assessment: SourceAssessmentState,
    narrative_risk: NarrativeRiskState,
    cross_asset_confirmation: CrossAssetConfirmationState,
    pricing_discipline: PricingDisciplineState,
    prices: pd.DataFrame,
) -> TradeReadinessState:
    liquidity = 0.0
    if not prices.empty and "volume" in prices.columns:
        liquidity = min(
            1.0,
            max(
                0.0,
                float(
                    pd.to_numeric(prices["volume"], errors="coerce").tail(20).mean() or 0.0
                )
                / 5_000_000,
            ),
        )
    thesis_validity = min(
        1.0,
        max(
            0.0,
            (
                float(source_assessment.reliability_score.value)
                + float(narrative_risk.actionability_score.value)
                + float(cross_asset_confirmation.aggregate_confirmation.value)
            )
            / 3.0,
        ),
    )
    pricing_alignment = min(
        1.0,
        max(
            0.0,
            (
                float(pricing_discipline.market_confirmation.value)
                + max(0.0, 1.0 - float(narrative_risk.deception_risk.value))
            )
            / 2.0,
        ),
    )
    timing_quality = min(
        1.0,
        max(
            0.0,
            (
                max(0.0, 1.0 - float(pricing_discipline.move_lateness.value))
                + float(pricing_discipline.reaction_vs_follow_through.value)
            )
            / 2.0,
        ),
    )
    risk_reward = min(
        1.0,
        max(0.0, (thesis_validity + pricing_alignment + timing_quality) / 3.0),
    )
    summary = min(1.0, max(0.0, (risk_reward + liquidity) / 2.0))

    def _cv(value: float) -> CausalValue:
        value = round(value, 4)
        return CausalValue(value=value, confidence=value, status=_status_from_score(value))

    return TradeReadinessState(
        summary=_cv(summary),
        thesis_validity=_cv(thesis_validity),
        pricing_alignment=_cv(pricing_alignment),
        timing_quality=_cv(timing_quality),
        liquidity_quality=_cv(liquidity),
        risk_reward_quality=_cv(risk_reward),
    )


def build_confidence_breakdown(
    *,
    source_assessment: SourceAssessmentState,
    cross_asset_confirmation: CrossAssetConfirmationState,
    pricing_discipline: PricingDisciplineState,
    trade_readiness: TradeReadinessState,
    narrative_risk: NarrativeRiskState,
    regime_confidence: float,
    analog_confidence: float,
    model_confidence: float,
) -> ConfidenceBreakdown:
    data_conf = min(
        1.0,
        max(
            0.0,
            (
                float(source_assessment.reliability_score.value)
                + float(source_assessment.freshness.value)
                + float(cross_asset_confirmation.aggregate_confirmation.value)
            )
            / 3.0,
        ),
    )
    state_conf = min(
        1.0,
        max(
            0.0,
            (
                regime_confidence
                + float(trade_readiness.thesis_validity.value)
                + max(0.0, 1.0 - float(narrative_risk.deception_risk.value))
            )
            / 3.0,
        ),
    )
    pricing_conf = min(
        1.0,
        max(
            0.0,
            (
                float(pricing_discipline.market_confirmation.value)
                + float(trade_readiness.pricing_alignment.value)
                + float(cross_asset_confirmation.aggregate_confirmation.value)
            )
            / 3.0,
        ),
    )
    analog_conf = min(1.0, max(0.0, analog_confidence))
    model_conf = min(1.0, max(0.0, model_confidence))
    uncapped = (
        0.20 * data_conf
        + 0.20 * state_conf
        + 0.20 * model_conf
        + 0.25 * pricing_conf
        + 0.15 * analog_conf
    )
    capped = uncapped
    cap_reasons: list[str] = []
    if data_conf < 0.35:
        capped = min(capped, 0.35)
        cap_reasons.append("weak_data")
    if pricing_conf < 0.30:
        capped = min(capped, 0.40)
        cap_reasons.append("low_pricing_confidence")
    if state_conf < 0.35:
        capped = min(capped, 0.45)
        cap_reasons.append("low_state_confidence")
    if cross_asset_confirmation.aggregate_confirmation.status == "conflicted":
        capped = min(capped, 0.45)
        cap_reasons.append("conflicted_cross_asset")
    if float(narrative_risk.deception_risk.value) > 0.70:
        capped = min(capped, 0.35)
        cap_reasons.append("high_deception_risk")

    def _cv(value: float, status: str | None = None) -> CausalValue:
        value = round(value, 4)
        return CausalValue(
            value=value,
            confidence=value,
            status=status or _status_from_score(value),
        )

    return ConfidenceBreakdown(
        data_confidence=_cv(data_conf),
        state_confidence=_cv(state_conf, "conflicted" if state_conf < 0.35 else None),
        model_confidence=_cv(model_conf),
        pricing_confidence=_cv(pricing_conf),
        analog_confidence=_cv(analog_conf),
        decision_confidence_uncapped=round(uncapped, 4),
        decision_confidence=round(max(0.0, min(1.0, capped)), 4),
        cap_reasons=cap_reasons,
    )


def resolve_trust_tier(tier: str | None) -> TrustTier:
    normalized = str(tier or "experimental").strip().lower()
    if normalized in {"experimental", "paper", "trusted"}:
        return normalized  # type: ignore[return-value]
    return "experimental"


def trust_tier_config(tier: TrustTier) -> dict[str, Any]:
    config = load_trust_tiers()
    return dict(config["tiers"][tier])


def _field_value(obj: Any, *keys: str, default: float = 0.0) -> float:
    current = obj
    for key in keys:
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _field_status(obj: Any, *keys: str, default: str = "missing") -> str:
    current = obj
    for key in keys:
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return str(current) if current is not None else default


def layer_freshness_ok(
    *,
    evidence_index: dict[str, dict[str, Any]],
    evidence_ids: list[str],
    layer: str,
    as_of: datetime,
) -> bool:
    freshness = load_evidence_freshness()["classes"]
    if layer not in {"pricing", "decision"}:
        return True
    if not evidence_ids:
        return False
    for evidence_id in evidence_ids:
        evidence = evidence_index.get(evidence_id)
        if not evidence:
            return False
        freshness_class = str(evidence.get("freshness_class") or "background")
        if freshness_class not in {"market", "decision_critical"}:
            return False
        timestamp = pd.to_datetime(evidence.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(timestamp):
            return False
        max_age_days = float(freshness[freshness_class]["max_age_days"])
        age_days = (as_of - timestamp.to_pydatetime()).total_seconds() / 86_400
        if age_days > max_age_days:
            return False
    return True


def apply_trust_tier_governance(
    *,
    requested_mode: str,
    trust_tier: TrustTier,
    confidence_breakdown: Any,
    narrative_risk: Any,
    cross_asset_confirmation: Any,
    critic_veto: bool,
    critic_reason_codes: list[str],
    pricing_evidence_ids: list[str],
    decision_evidence_ids: list[str],
    evidence_index: dict[str, dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    resolved_mode = requested_mode
    downgrade_reason_category: str | None = None
    pricing_fresh = layer_freshness_ok(
        evidence_index=evidence_index,
        evidence_ids=pricing_evidence_ids,
        layer="pricing",
        as_of=as_of,
    )
    decision_fresh = layer_freshness_ok(
        evidence_index=evidence_index,
        evidence_ids=decision_evidence_ids,
        layer="decision",
        as_of=as_of,
    )
    final_conf = _field_value(confidence_breakdown, "decision_confidence")
    if not decision_fresh or not pricing_fresh:
        resolved_mode = "research"
        downgrade_reason_category = "stale_evidence"
        final_conf = 0.0
    elif critic_veto:
        resolved_mode = "research"
        downgrade_reason_category = "critic_veto"
    else:
        policy = trust_tier_config(trust_tier)
        if _field_value(narrative_risk, "deception_risk", "value") > float(
            policy["max_deception_risk"]
        ):
            resolved_mode = "research"
            downgrade_reason_category = "high_deception_risk"
        elif (
            _field_status(cross_asset_confirmation, "aggregate_confirmation", "status")
            == "conflicted"
        ):
            resolved_mode = "research"
            downgrade_reason_category = "conflicted_cross_asset"
        elif _field_value(confidence_breakdown, "pricing_confidence", "value") < float(
            policy["min_pricing_confidence"]
        ):
            resolved_mode = "research"
            downgrade_reason_category = "low_pricing_confidence"
        elif _field_value(confidence_breakdown, "data_confidence", "value") < float(
            policy["min_data_confidence"]
        ):
            resolved_mode = "research"
            downgrade_reason_category = "weak_data"
        elif _field_value(confidence_breakdown, "state_confidence", "value") < float(
            policy["min_state_confidence"]
        ):
            resolved_mode = "research"
            downgrade_reason_category = "low_state_confidence"
        elif _field_value(confidence_breakdown, "model_confidence", "value") < float(
            policy["min_model_confidence"]
        ):
            resolved_mode = "research"
            downgrade_reason_category = "low_model_confidence"
        elif final_conf < float(policy["min_decision_confidence"]):
            resolved_mode = "research"
            downgrade_reason_category = "trust_tier_block"

    forced_mode_change = requested_mode != resolved_mode
    cap_reasons = (
        list(confidence_breakdown.cap_reasons)
        if hasattr(confidence_breakdown, "cap_reasons")
        else list((confidence_breakdown or {}).get("cap_reasons", []))
    )
    reason_codes = list(dict.fromkeys(critic_reason_codes + cap_reasons))
    if (
        forced_mode_change
        and downgrade_reason_category
        and downgrade_reason_category not in reason_codes
    ):
        reason_codes.append(downgrade_reason_category)
    return {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "downgrade_reason_category": downgrade_reason_category,
        "decision_confidence": round(final_conf, 4),
        "critic_reason_codes": reason_codes,
        "forced_mode_change": forced_mode_change,
        "pricing_fresh": pricing_fresh,
        "decision_fresh": decision_fresh,
        "trust_tier": trust_tier,
    }


def as_serializable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value

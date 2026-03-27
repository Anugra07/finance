from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ai_analyst.causal.types import ModelInterpretationPacket

FEATURE_FAMILY_MAP = {
    "ret_": "technical",
    "overnight_gap": "technical",
    "range_norm": "technical",
    "realized_vol": "technical",
    "volume_surprise": "technical",
    "share_turnover": "technical",
    "oil_supply_risk": "themes",
    "gas_supply_risk": "themes",
    "shipping_stress": "themes",
    "sanctions_pressure": "themes",
    "defense_escalation": "themes",
    "grid_stress": "themes",
    "cyber_infra_risk": "themes",
    "policy_relief": "themes",
    "commodity_shock_score": "mediators",
    "geo_novelty_score": "convergence",
    "regional_concentration_risk": "regime",
    "sector_context_shock": "channels",
    "stock_context_shock": "channels",
    "analog_match_score": "lags",
    "analog_failure_risk": "lags",
    "source_reliability_score": "source_quality",
    "source_claim_verifiability": "source_quality",
    "source_freshness_score": "source_quality",
    "narrative_deception_risk": "narrative_risk",
    "narrative_actionability_score": "narrative_risk",
    "narrative_novelty_score": "narrative_risk",
    "cross_asset_energy_confirmation": "cross_asset",
    "cross_asset_vol_confirmation": "cross_asset",
    "cross_asset_credit_confirmation": "cross_asset",
    "cross_asset_rates_confirmation": "cross_asset",
    "cross_asset_fx_confirmation": "cross_asset",
    "cross_asset_sector_flow_confirmation": "cross_asset",
    "cross_asset_peer_confirmation": "cross_asset",
    "cross_asset_aggregate_confirmation": "cross_asset",
    "pricing_discipline_market_confirmation": "pricing_discipline",
    "pricing_discipline_move_lateness": "pricing_discipline",
    "pricing_discipline_overextension": "pricing_discipline",
    "pricing_discipline_reaction_follow_through": "pricing_discipline",
    "trade_readiness_thesis_validity": "trade_readiness",
    "trade_readiness_pricing_alignment": "trade_readiness",
    "trade_readiness_timing_quality": "trade_readiness",
    "trade_readiness_liquidity_quality": "trade_readiness",
    "trade_readiness_risk_reward_quality": "trade_readiness",
    "event_dispersion_score": "convergence",
}


def _family_for_feature(name: str) -> str:
    for prefix, family in FEATURE_FAMILY_MAP.items():
        if name.startswith(prefix):
            return family
    return "technical"


def build_model_interpretation_packet(
    *,
    ticker: str,
    prediction_row: dict[str, object],
    feature_importance: pd.Series | None = None,
    pricing_disagreement_summary: dict[str, object] | None = None,
    analog_support_summary: list[dict[str, object]] | None = None,
) -> ModelInterpretationPacket:
    importance = (
        feature_importance.sort_values(ascending=False)
        if feature_importance is not None and not feature_importance.empty
        else pd.Series(dtype=float)
    )
    top_positive = [
        {"feature": feature, "shap": float(value), "value": None}
        for feature, value in importance.head(5).items()
    ]
    top_negative = [
        {"feature": feature, "shap": float(-abs(value)), "value": None}
        for feature, value in importance.tail(5).items()
    ]
    grouped: dict[str, float] = defaultdict(float)
    for feature, value in importance.items():
        grouped[_family_for_feature(str(feature))] += float(value)

    disagreement_flags: list[str] = []
    divergence = float((pricing_disagreement_summary or {}).get("divergence_score", 0.0) or 0.0)
    if divergence > 0.5:
        disagreement_flags.append("geo_signal_outpaces_market_confirmation")
    if float(prediction_row.get("confidence_score", 0.0) or 0.0) < 0.4:
        disagreement_flags.append("model_confidence_subdued")

    return ModelInterpretationPacket(
        ticker=ticker,
        model_score=float(prediction_row.get("prediction", 0.0) or 0.0),
        rank_pct=float(prediction_row.get("prob_outperform", 0.0) or 0.0),
        prediction_horizon_days=5,
        regime=str(prediction_row.get("regime") or "unknown"),
        top_positive_drivers=top_positive,
        top_negative_drivers=top_negative,
        driver_groups={key: round(value, 6) for key, value in grouped.items()},
        pricing_disagreement_summary=pricing_disagreement_summary or {},
        model_disagreement_flags=disagreement_flags,
        analog_support_summary=analog_support_summary or [],
    )

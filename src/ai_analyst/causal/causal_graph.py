from __future__ import annotations

import importlib.resources
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ai_analyst.causal.governance import (
    build_confidence_breakdown,
    build_cross_asset_confirmation_state,
    build_pricing_discipline_state,
    build_trade_readiness_state,
    summarize_narrative_risk,
    summarize_source_assessment,
)
from ai_analyst.causal.pricing_disagreement import build_pricing_disagreement_state
from ai_analyst.causal.types import (
    CausalChain,
    CausalState,
    CausalValue,
    CrossAssetConfirmationState,
    ExposureState,
    HorizonView,
    MediatorState,
    NarrativeRiskState,
    PricingDisagreementState,
    PricingDisciplineState,
    RegimeState,
    SourceAssessmentState,
    ThemeState,
    TradeReadinessState,
    TransmissionState,
)
from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path


class CausalGraphValidationError(ValueError):
    pass


def _load_yaml_asset(name: str) -> dict[str, Any]:
    with (
        importlib.resources.files("ai_analyst.causal")
        .joinpath(f"assets/{name}")
        .open(
            "r",
            encoding="utf-8",
        ) as handle
    ):
        return yaml.safe_load(handle)


def _validate_edges(edges: list[dict[str, Any]], *, required: list[str]) -> None:
    seen: set[tuple[str, ...]] = set()
    for edge in edges:
        missing = [field for field in required if field not in edge]
        if missing:
            raise CausalGraphValidationError(f"Missing edge fields: {missing}")
        key = tuple(str(edge.get(field)) for field in required[:4])
        if key in seen and bool(edge.get("deterministic_edge", False)):
            raise CausalGraphValidationError(f"Duplicate deterministic edge: {key}")
        seen.add(key)
        if int(edge["sign"]) not in {-1, 1}:
            raise CausalGraphValidationError(f"Invalid sign for edge: {edge}")
        if float(edge["lag_days_min"]) > float(edge["lag_days_max"]):
            raise CausalGraphValidationError(f"Invalid lag window for edge: {edge}")


class CausalGraphEngine:
    def __init__(self) -> None:
        self.transmission = _load_yaml_asset("transmission_edges.yaml")
        self.sector_mediator = _load_yaml_asset("sector_mediator_map.yaml")
        self.stock_exposure = _load_yaml_asset("stock_exposure_map.yaml")
        _validate_edges(
            self.transmission.get("edges", []),
            required=["source_kind", "source_key", "target_kind", "target_key"],
        )
        _validate_edges(
            self.sector_mediator.get("edges", []),
            required=["mediator", "sector", "sign", "weight"],
        )
        _validate_edges(
            self.stock_exposure.get("edges", []),
            required=["sector", "ticker", "sign", "weight"],
        )

    def build_chains(
        self,
        *,
        events: pd.DataFrame,
        theme_intensities: pd.DataFrame,
        sector_rankings: pd.DataFrame,
    ) -> list[CausalChain]:
        if theme_intensities.empty:
            return []

        event_lookup = (
            events.sort_values(["event_time", "market_relevance"], ascending=[False, False])
            .groupby("theme", sort=False)
            .head(1)
            .set_index("theme")
            if not events.empty and "theme" in events.columns
            else pd.DataFrame()
        )
        theme_scores = (
            theme_intensities[["theme", "intensity"]]
            .dropna(subset=["theme"])
            .drop_duplicates(subset=["theme"])
            .set_index("theme")
        )
        sector_scores = (
            sector_rankings[["sector", "sector_score"]]
            .dropna(subset=["sector"])
            .drop_duplicates(subset=["sector"])
            .set_index("sector")
            if not sector_rankings.empty
            else pd.DataFrame()
        )

        channel_edges = [
            edge for edge in self.transmission["edges"] if edge["source_kind"] == "theme"
        ]
        mediator_edges = [
            edge for edge in self.transmission["edges"] if edge["source_kind"] == "channel"
        ]
        chains: list[CausalChain] = []

        for theme, row in theme_scores.iterrows():
            base_score = float(row["intensity"] or 0.0)
            source_event_id = None
            evidence: list[dict[str, Any]] = []
            if not event_lookup.empty and theme in event_lookup.index:
                event_row = event_lookup.loc[theme]
                source_event_id = str(event_row.get("event_id") or "")
                evidence.append(
                    {
                        "event_id": source_event_id,
                        "topic": event_row.get("topic"),
                        "market_relevance": float(event_row.get("market_relevance", 0.0) or 0.0),
                    }
                )

            for channel_edge in channel_edges:
                if channel_edge["source_key"] != theme:
                    continue
                channel_score = base_score * float(channel_edge["weight"])
                for mediator_edge in mediator_edges:
                    if mediator_edge["source_key"] != channel_edge["target_key"]:
                        continue
                    mediator_score = channel_score * float(mediator_edge["weight"])
                    for sector_edge in self.sector_mediator["edges"]:
                        if sector_edge["mediator"] != mediator_edge["target_key"]:
                            continue
                        sector = str(sector_edge["sector"])
                        sector_rank_boost = 0.0
                        if not sector_scores.empty and sector in sector_scores.index:
                            sector_rank_boost = float(
                                sector_scores.loc[sector, "sector_score"] or 0.0
                            )
                        total_weight = mediator_score * float(sector_edge["weight"])
                        activation_confidence = min(
                            1.0,
                            max(0.0, abs(total_weight) / 5.0 + abs(sector_rank_boost) / 10.0),
                        )
                        chains.append(
                            CausalChain(
                                source_event_id=source_event_id or None,
                                theme=theme,
                                channel=str(channel_edge["target_key"]),
                                mediator=str(mediator_edge["target_key"]),
                                sector=sector,
                                ticker=None,
                                sign=int(sector_edge["sign"]),
                                weight=float(total_weight),
                                lag_days_min=int(
                                    min(
                                        channel_edge["lag_days_min"],
                                        mediator_edge["lag_days_min"],
                                        sector_edge["lag_days_min"],
                                    )
                                ),
                                lag_days_max=int(
                                    max(
                                        channel_edge["lag_days_max"],
                                        mediator_edge["lag_days_max"],
                                        sector_edge["lag_days_max"],
                                    )
                                ),
                                activation_confidence=round(activation_confidence, 4),
                                activation_status="supported"
                                if activation_confidence >= 0.55
                                else "weakly_supported",
                                deterministic_edge=bool(channel_edge["deterministic_edge"])
                                and bool(mediator_edge["deterministic_edge"]),
                                rationale=str(sector_edge["rationale"]),
                                evidence=evidence,
                            )
                        )
        return chains

    def build_state(
        self,
        *,
        as_of: datetime,
        theme_intensities: pd.DataFrame,
        sector_rankings: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        source_assessment_frame: pd.DataFrame | None = None,
        narrative_risk_frame: pd.DataFrame | None = None,
        pricing_disagreement: PricingDisagreementState,
        analog_matches: list[dict[str, Any]],
        chains: list[CausalChain],
        theme_regimes: pd.DataFrame,
    ) -> CausalState:
        macro = macro if macro is not None else pd.DataFrame()
        prices = prices if prices is not None else pd.DataFrame()
        source_assessment_frame = (
            source_assessment_frame if source_assessment_frame is not None else pd.DataFrame()
        )
        narrative_risk_frame = (
            narrative_risk_frame if narrative_risk_frame is not None else pd.DataFrame()
        )
        source_assessment: SourceAssessmentState = summarize_source_assessment(
            source_assessment_frame
        )
        narrative_risk: NarrativeRiskState = summarize_narrative_risk(narrative_risk_frame)
        cross_asset: CrossAssetConfirmationState = build_cross_asset_confirmation_state(
            macro=macro,
            prices=prices,
            sector_rankings=sector_rankings,
        )
        pricing_discipline: PricingDisciplineState = build_pricing_discipline_state(
            pricing_disagreement=pricing_disagreement,
            cross_asset_confirmation=cross_asset,
            prices=prices,
        )
        trade_readiness: TradeReadinessState = build_trade_readiness_state(
            source_assessment=source_assessment,
            narrative_risk=narrative_risk,
            cross_asset_confirmation=cross_asset,
            pricing_discipline=pricing_discipline,
            prices=prices,
        )
        uncertainty_map = {
            "regime": "missing" if theme_regimes.empty else "supported",
            "analogs": "missing" if not analog_matches else "supported",
            "pricing_disagreement": pricing_disagreement.divergence_score.status,
            "source_assessment": source_assessment.summary.status,
            "narrative_risk": narrative_risk.summary.status,
            "cross_asset_confirmation": cross_asset.aggregate_confirmation.status,
        }
        active_themes = []
        for _, row in theme_intensities.head(5).iterrows():
            confidence = min(
                1.0,
                max(
                    0.0,
                    float(row.get("avg_severity", 0.0) or 0.0)
                    * float(row.get("avg_novelty", 0.0) or 0.0)
                    + 0.25,
                ),
            )
            active_themes.append(
                CausalValue(
                    value=str(row["theme"]),
                    confidence=round(confidence, 4),
                    status="supported" if confidence >= 0.5 else "weakly_supported",
                    supporting_evidence=[
                        {
                            "intensity": float(row.get("intensity", 0.0) or 0.0),
                            "event_count": int(row.get("event_count", 0) or 0),
                        }
                    ],
                )
            )

        active_channels = []
        active_mediators = []
        for chain in chains[:5]:
            active_channels.append(
                CausalValue(
                    value=chain.channel,
                    confidence=chain.activation_confidence,
                    status=chain.activation_status,
                    supporting_evidence=chain.evidence,
                )
            )
            active_mediators.append(
                CausalValue(
                    value=chain.mediator,
                    confidence=chain.activation_confidence,
                    status=chain.activation_status,
                    supporting_evidence=chain.evidence,
                )
            )

        vulnerable_sectors = []
        for _, row in sector_rankings.head(10).iterrows():
            vulnerable_sectors.append(
                {
                    "sector": row.get("sector"),
                    "sector_score": float(row.get("sector_score", 0.0) or 0.0),
                    "context_shock": float(row.get("context_shock", 0.0) or 0.0),
                    "supporting_themes": row.get("supporting_themes") or [],
                }
            )

        if theme_regimes.empty:
            regime_label = "unknown"
            regime_confidence = 0.0
        else:
            top_regime = theme_regimes.sort_values("regime_score", ascending=False).iloc[0]
            regime_label = str(top_regime["regime_name"])
            regime_confidence = float(top_regime["regime_score"] or 0.0)

        horizons = [
            HorizonView(
                horizon="1-3 days",
                verdict="watch",
                confidence=min(1.0, max(0.0, pricing_disagreement.geo_signal_strength.confidence)),
                status=pricing_disagreement.geo_signal_strength.status,
                invalidation_triggers=["theme_intensity_fades", "market_response_reverses"],
                analog_summary=[
                    match for match in analog_matches if match.get("horizon") == "1-3 days"
                ][:2],
            ),
            HorizonView(
                horizon="1-3 weeks",
                verdict="monitor_transmission",
                confidence=min(
                    1.0, max(0.0, pricing_disagreement.mediator_confirmation.confidence)
                ),
                status=pricing_disagreement.mediator_confirmation.status,
                invalidation_triggers=[
                    "mediator_confirmation_breaks",
                    "sector_follow_through_absent",
                ],
                analog_summary=[
                    match for match in analog_matches if match.get("horizon") == "1-3 weeks"
                ][:2],
            ),
            HorizonView(
                horizon="1-3 months",
                verdict="conditional_follow_through",
                confidence=min(
                    1.0, max(0.0, pricing_disagreement.follow_through_status.confidence)
                ),
                status=pricing_disagreement.follow_through_status.status,
                invalidation_triggers=["regime_deescalates", "policy_relief_dominates"],
                analog_summary=[
                    match for match in analog_matches if match.get("horizon") == "1-3 months"
                ][:2],
            ),
        ]

        analog_confidence = 0.0
        if analog_matches:
            analog_confidence = float(
                max(float(match.get("similarity_score", 0.0) or 0.0) for match in analog_matches)
            )
        model_confidence = min(
            1.0,
            max(
                0.0,
                (
                    float(pricing_disagreement.mediator_confirmation.confidence)
                    + min(1.0, len(chains) / 10.0)
                )
                / 2.0,
            ),
        )
        confidence_breakdown = build_confidence_breakdown(
            source_assessment=source_assessment,
            cross_asset_confirmation=cross_asset,
            pricing_discipline=pricing_discipline,
            trade_readiness=trade_readiness,
            narrative_risk=narrative_risk,
            regime_confidence=regime_confidence,
            analog_confidence=analog_confidence,
            model_confidence=model_confidence,
        )

        return CausalState(
            as_of=as_of,
            regime=RegimeState(
                label=CausalValue(
                    value=regime_label,
                    confidence=round(regime_confidence, 4),
                    status="supported" if regime_confidence >= 0.55 else "weakly_supported",
                ),
                escalation_velocity=CausalValue(
                    value=float(theme_intensities["intensity"].head(3).sum())
                    if not theme_intensities.empty
                    else 0.0,
                    confidence=0.6 if not theme_intensities.empty else 0.0,
                    status="supported" if not theme_intensities.empty else "missing",
                ),
                tail_risk=CausalValue(
                    value=float(theme_intensities["intensity"].max())
                    if not theme_intensities.empty
                    else 0.0,
                    confidence=0.55 if not theme_intensities.empty else 0.0,
                    status="supported" if not theme_intensities.empty else "missing",
                ),
            ),
            themes=ThemeState(
                active_themes=active_themes,
                convergence_score=CausalValue(
                    value=float(theme_intensities["intensity"].head(3).sum())
                    if not theme_intensities.empty
                    else 0.0,
                    confidence=0.5 if len(active_themes) >= 2 else 0.25,
                    status="supported" if len(active_themes) >= 2 else "weakly_supported",
                ),
                uncertainty_map=uncertainty_map,
            ),
            transmission=TransmissionState(active_channels=active_channels),
            mediators=MediatorState(
                active_mediators=active_mediators,
                confirmation_score=CausalValue(
                    value=float(pricing_disagreement.mediator_confirmation.value or 0.0),
                    confidence=pricing_disagreement.mediator_confirmation.confidence,
                    status=pricing_disagreement.mediator_confirmation.status,
                ),
            ),
            exposures=ExposureState(vulnerable_sectors=vulnerable_sectors, vulnerable_stocks=[]),
            source_assessment=source_assessment,
            narrative_risk=narrative_risk,
            pricing_disagreement=pricing_disagreement,
            cross_asset_confirmation=cross_asset,
            pricing_discipline=pricing_discipline,
            trade_readiness=trade_readiness,
            confidence_breakdown=confidence_breakdown,
            horizon_views=horizons,
            missing_evidence=[key for key, value in uncertainty_map.items() if value == "missing"],
        )


def materialize_causal_state_outputs(
    settings: Settings,
    *,
    as_of: datetime,
    causal_state: CausalState,
    chains: list[CausalChain],
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path]]:
    state_paths: list[Path] = []
    chain_paths: list[Path] = []
    cross_asset_paths: list[Path] = []
    pricing_discipline_paths: list[Path] = []
    trade_readiness_paths: list[Path] = []
    loaded_at = datetime.now(tz=UTC)
    state_rows = []

    def _append_state(state_key: str, state_value: object, confidence: float, status: str) -> None:
        state_rows.append(
            {
                "as_of_date": as_of.date(),
                "state_key": state_key,
                "state_value": str(state_value),
                "state_confidence": float(confidence),
                "state_status": status,
                "graph_version": causal_state.version_metadata.graph_version,
                "causal_state_version": causal_state.version_metadata.causal_state_version,
                "reasoning_schema_version": causal_state.version_metadata.reasoning_schema_version,
                "transform_loaded_at": loaded_at,
            }
        )

    _append_state(
        "regime",
        causal_state.regime.label.value,
        causal_state.regime.label.confidence,
        causal_state.regime.label.status,
    )
    _append_state(
        "regime_score",
        causal_state.regime.label.confidence,
        causal_state.regime.label.confidence,
        causal_state.regime.label.status,
    )
    _append_state(
        "regime_is_escalating",
        1.0 if str(causal_state.regime.label.value) == "escalating" else 0.0,
        causal_state.regime.label.confidence,
        causal_state.regime.label.status,
    )
    _append_state(
        "theme_convergence_score",
        causal_state.themes.convergence_score.value
        if causal_state.themes.convergence_score
        else 0.0,
        causal_state.themes.convergence_score.confidence
        if causal_state.themes.convergence_score
        else 0.0,
        causal_state.themes.convergence_score.status
        if causal_state.themes.convergence_score
        else "missing",
    )
    _append_state(
        "pricing_geo_signal_strength",
        causal_state.pricing_disagreement.geo_signal_strength.value,
        causal_state.pricing_disagreement.geo_signal_strength.confidence,
        causal_state.pricing_disagreement.geo_signal_strength.status,
    )
    _append_state(
        "pricing_mediator_confirmation",
        causal_state.pricing_disagreement.mediator_confirmation.value,
        causal_state.pricing_disagreement.mediator_confirmation.confidence,
        causal_state.pricing_disagreement.mediator_confirmation.status,
    )
    _append_state(
        "pricing_market_response_strength",
        causal_state.pricing_disagreement.market_response_strength.value,
        causal_state.pricing_disagreement.market_response_strength.confidence,
        causal_state.pricing_disagreement.market_response_strength.status,
    )
    _append_state(
        "pricing_divergence",
        causal_state.pricing_disagreement.divergence_score.value,
        causal_state.pricing_disagreement.divergence_score.confidence,
        causal_state.pricing_disagreement.divergence_score.status,
    )
    _append_state(
        "pricing_crowdedness_proxy",
        causal_state.pricing_disagreement.crowdedness_proxy.value,
        causal_state.pricing_disagreement.crowdedness_proxy.confidence,
        causal_state.pricing_disagreement.crowdedness_proxy.status,
    )
    _append_state(
        "pricing_follow_through_status",
        causal_state.pricing_disagreement.follow_through_status.value,
        causal_state.pricing_disagreement.follow_through_status.confidence,
        causal_state.pricing_disagreement.follow_through_status.status,
    )
    _append_state(
        "source_reliability_score",
        causal_state.source_assessment.reliability_score.value,
        causal_state.source_assessment.reliability_score.confidence,
        causal_state.source_assessment.reliability_score.status,
    )
    _append_state(
        "source_claim_verifiability",
        causal_state.source_assessment.claim_verifiability.value,
        causal_state.source_assessment.claim_verifiability.confidence,
        causal_state.source_assessment.claim_verifiability.status,
    )
    _append_state(
        "source_freshness_score",
        causal_state.source_assessment.freshness.value,
        causal_state.source_assessment.freshness.confidence,
        causal_state.source_assessment.freshness.status,
    )
    _append_state(
        "narrative_deception_risk",
        causal_state.narrative_risk.deception_risk.value,
        causal_state.narrative_risk.deception_risk.confidence,
        causal_state.narrative_risk.deception_risk.status,
    )
    _append_state(
        "narrative_actionability_score",
        causal_state.narrative_risk.actionability_score.value,
        causal_state.narrative_risk.actionability_score.confidence,
        causal_state.narrative_risk.actionability_score.status,
    )
    _append_state(
        "narrative_novelty_score",
        causal_state.narrative_risk.novelty_score.value,
        causal_state.narrative_risk.novelty_score.confidence,
        causal_state.narrative_risk.novelty_score.status,
    )
    state_frame = pd.DataFrame(state_rows)
    out_path = warehouse_partition_path(
        settings,
        domain="causal/state_daily",
        partition_date=as_of.date(),
        stem=f"causal_state_daily_{as_of.date().isoformat()}",
    )
    write_parquet(state_frame, out_path)
    state_paths.append(out_path)

    cross_asset_frame = pd.DataFrame(
        [
            {
                "as_of_date": as_of.date(),
                "confirmation_key": key,
                "value": value.value,
                "confidence": value.confidence,
                "status": value.status,
                "transform_loaded_at": loaded_at,
            }
            for key, value in {
                "energy_confirmation": causal_state.cross_asset_confirmation.energy_confirmation,
                "vol_confirmation": causal_state.cross_asset_confirmation.vol_confirmation,
                "credit_confirmation": causal_state.cross_asset_confirmation.credit_confirmation,
                "rates_confirmation": causal_state.cross_asset_confirmation.rates_confirmation,
                "fx_confirmation": causal_state.cross_asset_confirmation.fx_confirmation,
                "sector_flow_confirmation": (
                    causal_state.cross_asset_confirmation.sector_flow_confirmation
                ),
                "peer_confirmation": causal_state.cross_asset_confirmation.peer_confirmation,
                "aggregate_confirmation": (
                    causal_state.cross_asset_confirmation.aggregate_confirmation
                ),
            }.items()
        ]
    )
    out_path = warehouse_partition_path(
        settings,
        domain="trust/cross_asset",
        partition_date=as_of.date(),
        stem=f"cross_asset_confirmation_daily_{as_of.date().isoformat()}",
    )
    write_parquet(cross_asset_frame, out_path)
    cross_asset_paths.append(out_path)

    pricing_frame = pd.DataFrame(
        [
            {
                "as_of_date": as_of.date(),
                "discipline_key": key,
                "value": value.value,
                "confidence": value.confidence,
                "status": value.status,
                "transform_loaded_at": loaded_at,
            }
            for key, value in {
                "summary": causal_state.pricing_discipline.summary,
                "market_confirmation": causal_state.pricing_discipline.market_confirmation,
                "move_lateness": causal_state.pricing_discipline.move_lateness,
                "overextension": causal_state.pricing_discipline.overextension,
                "reaction_vs_follow_through": (
                    causal_state.pricing_discipline.reaction_vs_follow_through
                ),
            }.items()
        ]
    )
    out_path = warehouse_partition_path(
        settings,
        domain="trust/pricing_discipline",
        partition_date=as_of.date(),
        stem=f"pricing_discipline_daily_{as_of.date().isoformat()}",
    )
    write_parquet(pricing_frame, out_path)
    pricing_discipline_paths.append(out_path)

    readiness_frame = pd.DataFrame(
        [
            {
                "as_of_date": as_of.date(),
                "readiness_key": key,
                "value": value.value,
                "confidence": value.confidence,
                "status": value.status,
                "transform_loaded_at": loaded_at,
            }
            for key, value in {
                "summary": causal_state.trade_readiness.summary,
                "thesis_validity": causal_state.trade_readiness.thesis_validity,
                "pricing_alignment": causal_state.trade_readiness.pricing_alignment,
                "timing_quality": causal_state.trade_readiness.timing_quality,
                "liquidity_quality": causal_state.trade_readiness.liquidity_quality,
                "risk_reward_quality": causal_state.trade_readiness.risk_reward_quality,
            }.items()
        ]
    )
    out_path = warehouse_partition_path(
        settings,
        domain="trust/trade_readiness",
        partition_date=as_of.date(),
        stem=f"trade_readiness_daily_{as_of.date().isoformat()}",
    )
    write_parquet(readiness_frame, out_path)
    trade_readiness_paths.append(out_path)

    if chains:
        chain_frame = pd.DataFrame(
            [
                {
                    "as_of_date": as_of.date(),
                    "source_event_id": chain.source_event_id,
                    "theme": chain.theme,
                    "channel": chain.channel,
                    "mediator": chain.mediator,
                    "sector": chain.sector,
                    "ticker": chain.ticker,
                    "sign": chain.sign,
                    "weight": chain.weight,
                    "lag_days_min": chain.lag_days_min,
                    "lag_days_max": chain.lag_days_max,
                    "activation_confidence": chain.activation_confidence,
                    "activation_status": chain.activation_status,
                    "deterministic_edge": chain.deterministic_edge,
                    "rationale": chain.rationale,
                    "graph_version": chain.version_metadata.graph_version,
                    "transform_loaded_at": loaded_at,
                }
                for chain in chains
            ]
        )
        out_path = warehouse_partition_path(
            settings,
            domain="causal/chains",
            partition_date=as_of.date(),
            stem=f"causal_chain_activations_{as_of.date().isoformat()}",
        )
        write_parquet(chain_frame, out_path)
        chain_paths.append(out_path)
    return (
        state_paths,
        chain_paths,
        cross_asset_paths,
        pricing_discipline_paths,
        trade_readiness_paths,
    )


def build_and_materialize_causal_state(
    settings: Settings,
    *,
    as_of: datetime,
) -> tuple[
    CausalState | None,
    list[CausalChain],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
    list[Path],
]:
    conn = connect(settings)
    try:
        events = conn.execute(
            """
            SELECT *
            FROM normalized_events
            WHERE event_time <= ?
              AND event_time >= ? - INTERVAL 30 DAY
            ORDER BY event_time DESC, market_relevance DESC
            """,
            [as_of, as_of],
        ).df()
        theme_intensities = conn.execute(
            """
            SELECT *
            FROM theme_intensity_daily
            WHERE date = (
                SELECT MAX(date) FROM theme_intensity_daily WHERE date <= CAST(? AS DATE)
            )
            ORDER BY intensity DESC, theme
            """,
            [as_of],
        ).df()
        sector_rankings = conn.execute(
            """
            SELECT *
            FROM sector_rankings
            WHERE as_of_date = (
                SELECT MAX(as_of_date) FROM sector_rankings WHERE as_of_date <= CAST(? AS DATE)
            )
            ORDER BY rank_desc
            """,
            [as_of],
        ).df()
        prices = conn.execute(
            """
            SELECT *
            FROM prices
            WHERE known_at <= ?
            ORDER BY ticker, date
            """,
            [as_of],
        ).df()
        analogs = conn.execute(
            """
            SELECT *
            FROM historical_analogs
            WHERE as_of_date = (
                SELECT MAX(as_of_date) FROM historical_analogs WHERE as_of_date <= CAST(? AS DATE)
            )
            ORDER BY analog_type, similarity_score DESC
            """,
            [as_of],
        ).df()
        theme_regimes = conn.execute(
            """
            SELECT *
            FROM theme_regimes
            WHERE as_of_date = (
                SELECT MAX(as_of_date) FROM theme_regimes WHERE as_of_date <= CAST(? AS DATE)
            )
            ORDER BY regime_score DESC, theme
            """,
            [as_of],
        ).df()
        source_assessment = conn.execute(
            """
            SELECT *
            FROM event_source_assessment
            WHERE event_id IN (
                SELECT event_id
                FROM normalized_events
                WHERE event_time <= ?
                  AND event_time >= ? - INTERVAL 30 DAY
            )
            """,
            [as_of, as_of],
        ).df()
        narrative_risk = conn.execute(
            """
            SELECT *
            FROM event_narrative_risk
            WHERE event_id IN (
                SELECT event_id
                FROM normalized_events
                WHERE event_time <= ?
                  AND event_time >= ? - INTERVAL 30 DAY
            )
            """,
            [as_of, as_of],
        ).df()
        macro = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY series_id, observation_date
                        ORDER BY realtime_start DESC, known_at DESC
                    ) AS row_num
                FROM macro_vintages
                WHERE known_at <= ?
            )
            SELECT * EXCLUDE (row_num)
            FROM ranked
            WHERE row_num = 1
            ORDER BY series_id, observation_date
            """,
            [as_of],
        ).df()
    finally:
        conn.close()

    if theme_intensities.empty:
        return None, [], [], [], [], [], []
    pricing_disagreement = build_pricing_disagreement_state(
        theme_intensities=theme_intensities,
        sector_rankings=sector_rankings,
        prices=prices,
    )
    engine = CausalGraphEngine()
    chains = engine.build_chains(
        events=events,
        theme_intensities=theme_intensities,
        sector_rankings=sector_rankings,
    )
    causal_state = engine.build_state(
        as_of=as_of,
        theme_intensities=theme_intensities,
        sector_rankings=sector_rankings,
        macro=macro,
        prices=prices,
        source_assessment_frame=source_assessment,
        narrative_risk_frame=narrative_risk,
        pricing_disagreement=pricing_disagreement,
        analog_matches=analogs.to_dict(orient="records"),
        chains=chains,
        theme_regimes=theme_regimes,
    )
    state_paths, chain_paths, cross_asset_paths, pricing_paths, readiness_paths = (
        materialize_causal_state_outputs(
        settings,
        as_of=as_of,
        causal_state=causal_state,
        chains=chains,
    )
    )
    return (
        causal_state,
        chains,
        state_paths,
        chain_paths,
        cross_asset_paths,
        pricing_paths,
        readiness_paths,
    )

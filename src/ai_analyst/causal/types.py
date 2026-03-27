from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ai_analyst.causal.versioning import VersionMetadata, default_version_metadata
from ai_analyst.core.models import TrustTier

EvidenceStatus = Literal[
    "supported",
    "weakly_supported",
    "unsupported",
    "conflicted",
    "missing",
]


@dataclass(slots=True)
class CausalValue:
    value: Any
    confidence: float
    status: EvidenceStatus
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class RegimeState:
    label: CausalValue
    escalation_velocity: CausalValue
    tail_risk: CausalValue


@dataclass(slots=True)
class ThemeState:
    active_themes: list[CausalValue] = field(default_factory=list)
    convergence_score: CausalValue | None = None
    uncertainty_map: dict[str, EvidenceStatus] = field(default_factory=dict)


@dataclass(slots=True)
class TransmissionState:
    active_channels: list[CausalValue] = field(default_factory=list)
    dependency_markers: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MediatorState:
    active_mediators: list[CausalValue] = field(default_factory=list)
    confirmation_score: CausalValue | None = None


@dataclass(slots=True)
class ExposureState:
    vulnerable_sectors: list[dict[str, Any]] = field(default_factory=list)
    vulnerable_stocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SourceAssessmentState:
    summary: CausalValue
    reliability_score: CausalValue
    claim_verifiability: CausalValue
    source_grade: CausalValue
    freshness: CausalValue
    provenance_quality: CausalValue


@dataclass(slots=True)
class NarrativeRiskState:
    summary: CausalValue
    deception_risk: CausalValue
    signaling_vs_capability: CausalValue
    novelty_score: CausalValue
    actionability_score: CausalValue
    repetition_decay: CausalValue
    propaganda_risk: CausalValue


@dataclass(slots=True)
class PricingDisagreementState:
    geo_signal_strength: CausalValue
    mediator_confirmation: CausalValue
    market_response_strength: CausalValue
    divergence_score: CausalValue
    crowdedness_proxy: CausalValue
    follow_through_status: CausalValue


@dataclass(slots=True)
class CrossAssetConfirmationState:
    energy_confirmation: CausalValue
    vol_confirmation: CausalValue
    credit_confirmation: CausalValue
    rates_confirmation: CausalValue
    fx_confirmation: CausalValue
    sector_flow_confirmation: CausalValue
    peer_confirmation: CausalValue
    aggregate_confirmation: CausalValue


@dataclass(slots=True)
class PricingDisciplineState:
    summary: CausalValue
    market_confirmation: CausalValue
    move_lateness: CausalValue
    overextension: CausalValue
    reaction_vs_follow_through: CausalValue


@dataclass(slots=True)
class TradeReadinessState:
    summary: CausalValue
    thesis_validity: CausalValue
    pricing_alignment: CausalValue
    timing_quality: CausalValue
    liquidity_quality: CausalValue
    risk_reward_quality: CausalValue


@dataclass(slots=True)
class ConfidenceBreakdown:
    data_confidence: CausalValue
    state_confidence: CausalValue
    model_confidence: CausalValue
    pricing_confidence: CausalValue
    analog_confidence: CausalValue
    decision_confidence_uncapped: float
    decision_confidence: float
    cap_reasons: list[str] = field(default_factory=list)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class HorizonView:
    horizon: str
    verdict: str
    confidence: float
    status: EvidenceStatus
    invalidation_triggers: list[str] = field(default_factory=list)
    analog_summary: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CausalChain:
    source_event_id: str | None
    theme: str
    channel: str
    mediator: str
    sector: str
    ticker: str | None
    sign: int
    weight: float
    lag_days_min: int
    lag_days_max: int
    activation_confidence: float
    activation_status: EvidenceStatus
    deterministic_edge: bool
    rationale: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class AnalogMatch:
    horizon: str
    analog_key: str
    analog_start: str
    analog_end: str
    similarity_score: float
    analogy_strength: str
    misleading_analogy: bool
    important_differences: list[str] = field(default_factory=list)
    non_analog_reasons: list[str] = field(default_factory=list)
    analogy_failure_risk: float = 0.0
    forward_outcomes: dict[str, Any] = field(default_factory=dict)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class CausalState:
    as_of: datetime
    regime: RegimeState
    themes: ThemeState
    transmission: TransmissionState
    mediators: MediatorState
    exposures: ExposureState
    source_assessment: SourceAssessmentState
    narrative_risk: NarrativeRiskState
    pricing_disagreement: PricingDisagreementState
    cross_asset_confirmation: CrossAssetConfirmationState
    pricing_discipline: PricingDisciplineState
    trade_readiness: TradeReadinessState
    confidence_breakdown: ConfidenceBreakdown
    horizon_views: list[HorizonView] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class ModelInterpretationPacket:
    ticker: str
    model_score: float
    rank_pct: float
    prediction_horizon_days: int
    regime: str
    top_positive_drivers: list[dict[str, Any]] = field(default_factory=list)
    top_negative_drivers: list[dict[str, Any]] = field(default_factory=list)
    driver_groups: dict[str, float] = field(default_factory=dict)
    pricing_disagreement_summary: dict[str, Any] = field(default_factory=dict)
    model_disagreement_flags: list[str] = field(default_factory=list)
    analog_support_summary: list[dict[str, Any]] = field(default_factory=list)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class ResearchAnswer:
    ticker: str
    as_of: datetime
    active_chains: list[dict[str, Any]]
    analogs_by_horizon: dict[str, list[dict[str, Any]]]
    uncertainty_map: dict[str, EvidenceStatus]
    unknowns: list[str]
    missing_evidence: list[str]
    competing_hypotheses: list[dict[str, Any]]
    causal_state: dict[str, Any]
    fact_layer: dict[str, Any] = field(default_factory=dict)
    interpretation_layer: dict[str, Any] = field(default_factory=dict)
    pricing_layer: dict[str, Any] = field(default_factory=dict)
    decision_layer: dict[str, Any] = field(default_factory=dict)
    falsification_layer: dict[str, Any] = field(default_factory=dict)
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    trust_tier: TrustTier = "experimental"
    requested_mode: str = "research"
    resolved_mode: str = "research"
    downgrade_reason_category: str | None = None
    critic_outcome: dict[str, Any] = field(default_factory=dict)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)


@dataclass(slots=True)
class DecisionForecast:
    ticker: str
    as_of: datetime
    horizon_verdicts: list[dict[str, Any]]
    conviction: float
    abstain: bool
    invalidation_triggers: list[str]
    key_risks: list[str]
    ranking_context: dict[str, Any]
    model_disagreement_flags: list[str]
    causal_state: dict[str, Any]
    fact_layer: dict[str, Any] = field(default_factory=dict)
    interpretation_layer: dict[str, Any] = field(default_factory=dict)
    pricing_layer: dict[str, Any] = field(default_factory=dict)
    decision_layer: dict[str, Any] = field(default_factory=dict)
    falsification_layer: dict[str, Any] = field(default_factory=dict)
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    trust_tier: TrustTier = "experimental"
    requested_mode: str = "decision"
    resolved_mode: str = "decision"
    downgrade_reason_category: str | None = None
    critic_outcome: dict[str, Any] = field(default_factory=dict)
    version_metadata: VersionMetadata = field(default_factory=default_version_metadata)

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

FreshnessClass = Literal["background", "context", "market", "decision_critical"]
TrustTier = Literal["experimental", "paper", "trusted"]
MarketScope = Literal["US", "IN"]


@dataclass(slots=True)
class PriceBar:
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    known_at: datetime
    market_code: str = "US"
    currency: str = "USD"
    exchange_code: str | None = None


@dataclass(slots=True)
class CorporateAction:
    ticker: str
    date: date
    split_factor: float
    dividend_cash: float
    known_at: datetime


@dataclass(slots=True)
class EventRecord:
    event_time: datetime
    ingest_time: datetime
    source: str
    topic: str
    severity: float
    novelty: float
    confidence: float
    event_id: str | None = None
    event_family: str | None = None
    region: str | None = None
    affected_commodities: list[str] = field(default_factory=list)
    affected_sectors: list[str] = field(default_factory=list)
    affected_entities: list[str] = field(default_factory=list)
    geography: str | None = None
    theme: str | None = None
    duration_hours: float | None = None
    market_relevance: float | None = None
    raw_ref: str | None = None


@dataclass(slots=True)
class ThemeIntensity:
    as_of: datetime
    theme: str
    intensity: float
    source: str


@dataclass(slots=True)
class EvidenceRef:
    evidence_id: str
    source_type: str
    source_ref: str
    timestamp: datetime
    reliability: float
    freshness_class: FreshnessClass
    content_hash: str


@dataclass(slots=True)
class ContextPack:
    ticker: str
    as_of: datetime
    market_snapshot: dict[str, Any]
    macro_snapshot: dict[str, Any]
    top_events: list[dict[str, Any]]
    shap_drivers: list[dict[str, Any]] = field(default_factory=list)
    analogs: list[dict[str, Any]] = field(default_factory=list)
    filing_excerpts: list[dict[str, Any]] = field(default_factory=list)
    freshness_flags: dict[str, Any] = field(default_factory=dict)
    sector_rankings: list[dict[str, Any]] = field(default_factory=list)
    solution_ideas: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "research"
    causal_state: dict[str, Any] = field(default_factory=dict)
    causal_chains: list[dict[str, Any]] = field(default_factory=list)
    analog_matches: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    model_interpretation: dict[str, Any] = field(default_factory=dict)
    uncertainty_map: dict[str, Any] = field(default_factory=dict)
    competing_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    evidence_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_assessment: dict[str, Any] = field(default_factory=dict)
    narrative_risk: dict[str, Any] = field(default_factory=dict)
    cross_asset_confirmation: dict[str, Any] = field(default_factory=dict)
    pricing_discipline: dict[str, Any] = field(default_factory=dict)
    trade_readiness: dict[str, Any] = field(default_factory=dict)
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    trust_tier: TrustTier = "experimental"
    market_scope: MarketScope = "US"
    account_profile: dict[str, Any] = field(default_factory=dict)
    orderability: dict[str, Any] = field(default_factory=dict)
    monthly_model_interpretation: dict[str, Any] = field(default_factory=dict)
    monthly_benchmark_context: dict[str, Any] = field(default_factory=dict)
    version_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AccountProfile:
    capital: float
    base_currency: str
    market_scope: MarketScope
    max_names: int = 3
    fees_model: str = "zerodha_delivery_v1"
    slippage_model: str = "adv_proxy_v1"


@dataclass(slots=True)
class OrderabilityState:
    max_affordable_shares: int
    estimated_cash_use: float
    estimated_total_entry_cost: float
    estimated_total_exit_cost: float
    expected_net_edge_after_costs: float
    orderability_status: str
    residual_cash: float = 0.0


@dataclass(slots=True)
class ShortlistCandidate:
    ticker: str
    exchange_code: str
    currency: str
    current_price: float
    max_affordable_shares: int
    estimated_cash_use: float
    monthly_rank_score: float
    timing_gate_status: str
    decision_mode: str
    conviction: float
    thesis_summary: str
    pricing_status: str
    falsification_triggers: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    fact_layer: dict[str, Any] = field(default_factory=dict)
    interpretation_layer: dict[str, Any] = field(default_factory=dict)
    pricing_layer: dict[str, Any] = field(default_factory=dict)
    decision_layer: dict[str, Any] = field(default_factory=dict)
    falsification_layer: dict[str, Any] = field(default_factory=dict)
    cost_tradability: dict[str, Any] = field(default_factory=dict)
    downgrade_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ShortlistDecision:
    system_mode: str
    market_scope: MarketScope
    capital: float
    base_currency: str
    target_horizon_days: int
    market_summary: dict[str, Any]
    shortlist: list[ShortlistCandidate]
    primary_candidate: ShortlistCandidate | None = None
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    downgrade_reasons: list[str] = field(default_factory=list)
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime

import pandas as pd

from ai_analyst.causal.causal_graph import CausalGraphEngine
from ai_analyst.causal.governance import (
    as_serializable,
    build_confidence_breakdown,
    build_cross_asset_confirmation_state,
    build_pricing_discipline_state,
    build_trade_readiness_state,
    resolve_trust_tier,
    summarize_narrative_risk,
    summarize_source_assessment,
)
from ai_analyst.causal.pricing_disagreement import build_pricing_disagreement_state
from ai_analyst.config import Settings
from ai_analyst.core.contracts import MacroContextSource
from ai_analyst.core.models import ContextPack, EvidenceRef
from ai_analyst.events.context import LocalMacroContextSource
from ai_analyst.warehouse.snapshot_builder import SnapshotBuilder


class ContextPackBuilder:
    def __init__(
        self,
        settings: Settings,
        *,
        macro_context_source: MacroContextSource | None = None,
    ) -> None:
        self.settings = settings
        self.snapshot_builder = SnapshotBuilder(settings)
        self.macro_context_source = macro_context_source or LocalMacroContextSource(settings)

    @staticmethod
    def _latest_timestamp(frame, column: str) -> pd.Timestamp | None:
        if frame.empty or column not in frame.columns:
            return None
        values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
        if values.empty:
            return None
        return values.max()

    def _build_freshness_flags(
        self,
        *,
        as_of: datetime,
        prices,
        event_feed,
        submissions,
        theme_intensities,
    ) -> dict[str, object]:
        price_latest = self._latest_timestamp(prices, "known_at")
        event_latest = self._latest_timestamp(event_feed, "event_time")
        filing_latest = self._latest_timestamp(submissions, "acceptance_datetime")
        if filing_latest is None:
            filing_latest = self._latest_timestamp(submissions, "filing_date")

        theme_latest = self._latest_timestamp(theme_intensities, "latest_event_time")
        if (
            theme_latest is None
            and not theme_intensities.empty
            and "date" in theme_intensities.columns
        ):
            theme_dates = pd.to_datetime(
                theme_intensities["date"],
                utc=True,
                errors="coerce",
            ).dropna()
            if not theme_dates.empty:
                theme_latest = theme_dates.max()

        def _age_days(latest: pd.Timestamp | None) -> float | None:
            if latest is None:
                return None
            return round((as_of - latest.to_pydatetime()).total_seconds() / 86_400, 2)

        price_age_days = _age_days(price_latest)
        event_age_days = _age_days(event_latest)
        filing_age_days = _age_days(filing_latest)
        theme_age_days = _age_days(theme_latest)

        return {
            "macro_snapshot_built_at": as_of.isoformat(),
            "price_rows": len(prices),
            "event_rows": len(event_feed),
            "filing_rows": len(submissions),
            "price_latest_known_at": price_latest.isoformat() if price_latest is not None else None,
            "price_age_days": price_age_days,
            "price_is_stale": price_age_days is None or price_age_days > 2.0,
            "event_latest_time": event_latest.isoformat() if event_latest is not None else None,
            "event_age_days": event_age_days,
            "event_is_stale": event_age_days is None or event_age_days > 2.0,
            "filing_latest_time": filing_latest.isoformat() if filing_latest is not None else None,
            "filing_age_days": filing_age_days,
            "theme_latest_time": theme_latest.isoformat() if theme_latest is not None else None,
            "theme_age_days": theme_age_days,
            "theme_is_stale": theme_age_days is None or theme_age_days > 7.0,
        }

    @staticmethod
    def _group_analogs_by_horizon(analogs) -> dict[str, list[dict[str, object]]]:
        if analogs.empty:
            return {}
        grouped: dict[str, list[dict[str, object]]] = {}
        for analog_type, group in analogs.groupby("analog_type", sort=False):
            grouped[str(analog_type)] = group.head(3).to_dict(orient="records")
        return grouped

    @staticmethod
    def _competing_hypotheses(
        theme_intensities,
        pricing_disagreement,
    ) -> list[dict[str, object]]:
        if theme_intensities.empty:
            return [
                {
                    "hypothesis": "insufficient_context",
                    "confidence": 0.0,
                    "status": "missing",
                    "why": "No theme intensities were available for the as-of snapshot.",
                }
            ]
        top_theme = str(
            theme_intensities.sort_values("intensity", ascending=False).iloc[0]["theme"]
        )
        divergence = float(pricing_disagreement.divergence_score.value or 0.0)
        return [
            {
                "hypothesis": f"{top_theme}_dominant_transmission",
                "confidence": float(pricing_disagreement.geo_signal_strength.confidence),
                "status": pricing_disagreement.geo_signal_strength.status,
                "why": "Top theme intensity and sector response are aligned.",
            },
            {
                "hypothesis": "market_underreacting",
                "confidence": float(pricing_disagreement.divergence_score.confidence),
                "status": pricing_disagreement.divergence_score.status,
                "why": "Geo signal exceeds current mediator and market confirmation.",
            },
            {
                "hypothesis": "false_positive_theme_spike",
                "confidence": max(0.0, round(1.0 - min(1.0, divergence), 4)),
                "status": "conflicted" if divergence > 0.5 else "weakly_supported",
                "why": "If follow-through stays absent, the current theme spike may fade quickly.",
            },
        ]

    @staticmethod
    def _hash_payload(payload: dict[str, object]) -> str:
        rendered = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _coerce_timestamp(value: object, *, fallback: datetime) -> datetime:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return fallback
        return timestamp.to_pydatetime()

    def _build_evidence_index(
        self,
        *,
        ticker: str,
        as_of: datetime,
        snapshot,
        event_feed: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        evidence_index: dict[str, dict[str, object]] = {}
        if not snapshot.evidence_catalog.empty:
            for row in snapshot.evidence_catalog.to_dict(orient="records"):
                evidence_index[str(row["evidence_id"])] = {
                    "evidence_id": row["evidence_id"],
                    "source_type": row.get("source_type"),
                    "source_ref": row.get("source_ref"),
                    "timestamp": row.get("timestamp"),
                    "reliability": float(row.get("reliability", 0.0) or 0.0),
                    "freshness_class": row.get("freshness_class"),
                    "content_hash": row.get("content_hash"),
                    "content_preview": row.get("content_preview"),
                }
        for _, row in snapshot.prices.sort_values("date").tail(5).iterrows():
            payload = {
                "ticker": row.get("ticker"),
                "date": row.get("date"),
                "adj_close": row.get("adj_close"),
            }
            evidence = EvidenceRef(
                evidence_id=f"price::{row.get('ticker')}::{str(row.get('date'))}",
                source_type="price",
                source_ref=str(row.get("ticker")),
                timestamp=self._coerce_timestamp(row.get("known_at"), fallback=as_of),
                reliability=0.95,
                freshness_class="decision_critical",
                content_hash=self._hash_payload(payload),
            )
            evidence_index[evidence.evidence_id] = asdict(evidence)
        if not snapshot.theme_intensities.empty:
            for _, row in snapshot.theme_intensities.head(10).iterrows():
                evidence = EvidenceRef(
                    evidence_id=f"theme::{row.get('theme')}::{snapshot.as_of.date().isoformat()}",
                    source_type="theme_intensity",
                    source_ref=str(row.get("theme")),
                    timestamp=as_of,
                    reliability=0.85,
                    freshness_class="context",
                    content_hash=self._hash_payload(row.to_dict()),
                )
                evidence_index[evidence.evidence_id] = asdict(evidence)
        for _, row in event_feed.head(10).iterrows():
            event_id = str(row.get("event_id") or "")
            catalog_id = f"evt::{event_id}" if event_id else None
            if catalog_id and catalog_id in evidence_index:
                continue
            evidence = EvidenceRef(
                evidence_id=catalog_id or f"event::{ticker}::{len(evidence_index)}",
                source_type=str(row.get("source") or "event"),
                source_ref=str(row.get("raw_ref") or row.get("topic") or ""),
                timestamp=self._coerce_timestamp(row.get("event_time"), fallback=as_of),
                reliability=float(row.get("confidence", 0.0) or 0.0),
                freshness_class="market"
                if float(row.get("market_relevance", 0.0) or 0.0) >= 0.55
                else "context",
                content_hash=self._hash_payload(row.to_dict()),
            )
            evidence_index[evidence.evidence_id] = asdict(evidence)
        return evidence_index

    @staticmethod
    def _state_frame_to_lookup(
        frame: pd.DataFrame,
        *,
        key_col: str,
        value_col: str,
        confidence_col: str,
        status_col: str,
    ) -> dict[str, dict[str, object]]:
        if frame.empty:
            return {}
        lookup: dict[str, dict[str, object]] = {}
        for _, row in frame.iterrows():
            lookup[str(row[key_col])] = {
                "value": float(row[value_col]) if pd.notna(row[value_col]) else 0.0,
                "confidence": float(row[confidence_col]) if pd.notna(row[confidence_col]) else 0.0,
                "status": str(row[status_col]) if pd.notna(row[status_col]) else "missing",
            }
        return lookup

    def build(self, *, ticker: str, as_of: datetime, mode: str = "research") -> ContextPack:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        else:
            as_of = as_of.astimezone(UTC)

        snapshot = self.snapshot_builder.build(as_of=as_of, tickers=[ticker])
        theme_intensities = snapshot.theme_intensities
        if theme_intensities.empty:
            theme_intensities = self.macro_context_source.get_theme_intensities(as_of=as_of)
        event_feed = self.macro_context_source.get_event_feed(ticker=ticker, as_of=as_of)
        sector_rankings = snapshot.sector_rankings.to_dict(orient="records")
        solution_ideas = []
        if isinstance(self.macro_context_source, LocalMacroContextSource):
            if not sector_rankings:
                sector_rankings = self.macro_context_source.get_sector_rankings(
                    as_of=as_of
                ).to_dict(orient="records")
            solution_ideas = self.macro_context_source.get_solution_ideas(as_of=as_of).to_dict(
                orient="records"
            )

        market_snapshot = snapshot.prices.to_dict(orient="records")
        macro_snapshot = {
            "macro_rows": snapshot.macro.to_dict(orient="records"),
            "theme_intensities": theme_intensities.to_dict(orient="records"),
        }
        freshness_flags = self._build_freshness_flags(
            as_of=as_of,
            prices=snapshot.prices,
            event_feed=event_feed,
            submissions=snapshot.submissions,
            theme_intensities=theme_intensities,
        )
        source_assessment_state = summarize_source_assessment(snapshot.event_source_assessment)
        narrative_risk_state = summarize_narrative_risk(snapshot.event_narrative_risk)
        pricing_disagreement = build_pricing_disagreement_state(
            theme_intensities=theme_intensities,
            sector_rankings=snapshot.sector_rankings,
            prices=snapshot.prices,
        )
        cross_asset_confirmation = build_cross_asset_confirmation_state(
            macro=snapshot.macro,
            prices=snapshot.prices,
            sector_rankings=snapshot.sector_rankings,
        )
        pricing_discipline = build_pricing_discipline_state(
            pricing_disagreement=pricing_disagreement,
            cross_asset_confirmation=cross_asset_confirmation,
            prices=snapshot.prices,
        )
        trade_readiness = build_trade_readiness_state(
            source_assessment=source_assessment_state,
            narrative_risk=narrative_risk_state,
            cross_asset_confirmation=cross_asset_confirmation,
            pricing_discipline=pricing_discipline,
            prices=snapshot.prices,
        )
        analog_matches = self._group_analogs_by_horizon(snapshot.historical_analogs)
        evidence_index = self._build_evidence_index(
            ticker=ticker,
            as_of=as_of,
            snapshot=snapshot,
            event_feed=event_feed,
        )
        graph_engine = CausalGraphEngine()
        causal_chains = snapshot.causal_chains.to_dict(orient="records")
        causal_state_payload = snapshot.causal_state.to_dict(orient="records")
        if not causal_state_payload:
            built_chains = graph_engine.build_chains(
                events=snapshot.events,
                theme_intensities=theme_intensities,
                sector_rankings=snapshot.sector_rankings,
            )
            built_state = graph_engine.build_state(
                as_of=as_of,
                theme_intensities=theme_intensities,
                sector_rankings=snapshot.sector_rankings,
                macro=snapshot.macro,
                prices=snapshot.prices,
                source_assessment_frame=snapshot.event_source_assessment,
                narrative_risk_frame=snapshot.event_narrative_risk,
                pricing_disagreement=pricing_disagreement,
                analog_matches=snapshot.historical_analogs.to_dict(orient="records"),
                chains=built_chains,
                theme_regimes=snapshot.theme_regimes,
            )
            causal_chains = [asdict(chain) for chain in built_chains]
            causal_state_payload = asdict(built_state)
        else:
            causal_state_payload = {
                "rows": causal_state_payload,
                "cross_asset_confirmation": self._state_frame_to_lookup(
                    snapshot.cross_asset_confirmation,
                    key_col="confirmation_key",
                    value_col="value",
                    confidence_col="confidence",
                    status_col="status",
                ),
                "pricing_discipline": self._state_frame_to_lookup(
                    snapshot.pricing_discipline,
                    key_col="discipline_key",
                    value_col="value",
                    confidence_col="confidence",
                    status_col="status",
                ),
                "trade_readiness": self._state_frame_to_lookup(
                    snapshot.trade_readiness,
                    key_col="readiness_key",
                    value_col="value",
                    confidence_col="confidence",
                    status_col="status",
                ),
            }
        competing_hypotheses = self._competing_hypotheses(
            theme_intensities,
            pricing_disagreement,
        )
        uncertainty_map = (
            dict(causal_state_payload.get("themes", {}).get("uncertainty_map", {}))
            if isinstance(causal_state_payload, dict)
            else {}
        )
        missing_evidence = (
            list(causal_state_payload.get("missing_evidence", []))
            if isinstance(causal_state_payload, dict)
            else []
        )
        analog_confidence = 0.0
        if not snapshot.historical_analogs.empty:
            analog_confidence = float(
                pd.to_numeric(
                    snapshot.historical_analogs["similarity_score"],
                    errors="coerce",
                ).fillna(0.0).max()
            )
        confidence_breakdown = build_confidence_breakdown(
            source_assessment=source_assessment_state,
            cross_asset_confirmation=cross_asset_confirmation,
            pricing_discipline=pricing_discipline,
            trade_readiness=trade_readiness,
            narrative_risk=narrative_risk_state,
            regime_confidence=float(
                causal_state_payload.get("regime", {}).get("label", {}).get("confidence", 0.0)
                if isinstance(causal_state_payload, dict)
                else 0.0
            ),
            analog_confidence=analog_confidence,
            model_confidence=0.35,
        )
        return ContextPack(
            ticker=ticker,
            as_of=as_of,
            market_snapshot={"prices": market_snapshot},
            macro_snapshot=macro_snapshot,
            top_events=event_feed.head(10).to_dict(orient="records"),
            analogs=snapshot.historical_analogs.head(10).to_dict(orient="records"),
            filing_excerpts=snapshot.submissions.head(5).to_dict(orient="records"),
            freshness_flags=freshness_flags,
            sector_rankings=sector_rankings,
            solution_ideas=solution_ideas,
            mode=mode,
            causal_state=causal_state_payload if isinstance(causal_state_payload, dict) else {},
            causal_chains=causal_chains,
            analog_matches=analog_matches,
            model_interpretation={
                "pricing_disagreement": asdict(pricing_disagreement),
                "top_chain_count": len(causal_chains),
            },
            uncertainty_map=uncertainty_map,
            competing_hypotheses=competing_hypotheses,
            missing_evidence=missing_evidence,
            evidence_index=evidence_index,
            source_assessment=as_serializable(source_assessment_state),
            narrative_risk=as_serializable(narrative_risk_state),
            cross_asset_confirmation=as_serializable(cross_asset_confirmation),
            pricing_discipline=as_serializable(pricing_discipline),
            trade_readiness=as_serializable(trade_readiness),
            confidence_breakdown=as_serializable(confidence_breakdown),
            trust_tier=resolve_trust_tier(self.settings.trust_tier),
            version_metadata=causal_state_payload.get("version_metadata", {})
            if isinstance(causal_state_payload, dict)
            else {},
        )

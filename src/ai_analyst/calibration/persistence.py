from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.core.models import ContextPack
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.layout import warehouse_partition_path


def _forecast_id(ticker: str, as_of: datetime, horizon: str) -> str:
    payload = f"{ticker}|{as_of.isoformat()}|{horizon}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def _evidence_hash(payload: dict[str, object]) -> str:
    rendered = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:24]


def _jsonish(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _direction_from_verdict(verdict: str) -> str:
    lowered = str(verdict or "").strip().lower()
    if lowered in {"outperform", "bullish", "positive"}:
        return "up"
    if lowered in {"underperform", "bearish", "negative"}:
        return "down"
    return "neutral"


def persist_decision_forecast(
    settings: Settings,
    *,
    context_pack: ContextPack,
    decision_output: dict[str, object],
) -> tuple[list[Path], list[Path]]:
    as_of = context_pack.as_of.astimezone(UTC)
    trade_date = as_of.date()
    loaded_at = datetime.now(tz=UTC)
    critic = decision_output.get("critic") or {}
    research_packet = decision_output.get("research_packet") or {}
    confidence_breakdown = decision_output.get("confidence_breakdown") or {}
    evidence_hash = _evidence_hash(
        {
            "causal_state": context_pack.causal_state,
            "research_packet": research_packet,
            "decision_output": decision_output,
        }
    )
    override_applied = (
        bool(critic.get("force_abstain"))
        or abs(float(critic.get("confidence_adjustment", 0.0) or 0.0)) >= 0.10
    )
    override_reason = ", ".join(
        [
            *[
                str(item).strip()
                for item in (decision_output.get("key_risks") or [])
                if str(item).strip()
            ][:2],
            *[
                str(item).strip()
                for item in (critic.get("missing_evidence") or [])
                if str(item).strip()
            ][:2],
        ]
    ).strip(", ")

    outcome_rows: list[dict[str, object]] = []
    override_rows: list[dict[str, object]] = []
    horizon_verdicts = decision_output.get("horizon_verdicts") or []
    for row in horizon_verdicts:
        horizon = str(row.get("horizon") or "")
        forecast_id = _forecast_id(context_pack.ticker, as_of, horizon)
        outcome_rows.append(
            {
                "forecast_id": forecast_id,
                "ticker": context_pack.ticker,
                "as_of": as_of,
                "mode": context_pack.mode,
                "requested_mode": str(decision_output.get("requested_mode") or context_pack.mode),
                "resolved_mode": str(decision_output.get("resolved_mode") or context_pack.mode),
                "trust_tier": str(decision_output.get("trust_tier") or context_pack.trust_tier),
                "horizon": horizon,
                "direction": _direction_from_verdict(str(row.get("verdict") or "")),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "decision_confidence_uncapped": float(
                    confidence_breakdown.get("decision_confidence_uncapped", 0.0) or 0.0
                ),
                "data_confidence": float(
                    (confidence_breakdown.get("data_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "state_confidence": float(
                    (confidence_breakdown.get("state_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "model_confidence": float(
                    (confidence_breakdown.get("model_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "pricing_confidence": float(
                    (confidence_breakdown.get("pricing_confidence") or {}).get("value", 0.0)
                    or 0.0
                ),
                "analog_confidence": float(
                    (confidence_breakdown.get("analog_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "abstain": bool(decision_output.get("abstain")),
                "downgrade_reason_category": decision_output.get("downgrade_reason_category"),
                "invalidation_triggers": decision_output.get("invalidation_triggers") or [],
                "critic_reason_codes": (decision_output.get("critic_outcome") or {}).get(
                    "critic_reason_codes",
                    [],
                ),
                "evidence_hash": evidence_hash,
                "freshness_summary": _jsonish(context_pack.freshness_flags),
                "transform_loaded_at": loaded_at,
            }
        )
        override_rows.append(
            {
                "forecast_id": forecast_id,
                "ticker": context_pack.ticker,
                "as_of": as_of,
                "override_applied": override_applied,
                "override_reason": override_reason or None,
                "requested_mode": str(decision_output.get("requested_mode") or context_pack.mode),
                "resolved_mode": str(decision_output.get("resolved_mode") or context_pack.mode),
                "trust_tier": str(decision_output.get("trust_tier") or context_pack.trust_tier),
                "downgrade_reason_category": decision_output.get("downgrade_reason_category"),
                "critic_reason_codes": (decision_output.get("critic_outcome") or {}).get(
                    "critic_reason_codes",
                    [],
                ),
                "evidence_used": critic.get("missing_evidence") or [],
                "final_direction": _direction_from_verdict(str(row.get("verdict") or "")),
                "transform_loaded_at": loaded_at,
            }
        )

    if not outcome_rows:
        forecast_id = _forecast_id(context_pack.ticker, as_of, "no_horizon")
        outcome_rows.append(
            {
                "forecast_id": forecast_id,
                "ticker": context_pack.ticker,
                "as_of": as_of,
                "mode": context_pack.mode,
                "requested_mode": str(decision_output.get("requested_mode") or context_pack.mode),
                "resolved_mode": str(decision_output.get("resolved_mode") or context_pack.mode),
                "trust_tier": str(decision_output.get("trust_tier") or context_pack.trust_tier),
                "horizon": "no_horizon",
                "direction": "neutral",
                "confidence": 0.0,
                "decision_confidence_uncapped": float(
                    confidence_breakdown.get("decision_confidence_uncapped", 0.0) or 0.0
                ),
                "data_confidence": float(
                    (confidence_breakdown.get("data_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "state_confidence": float(
                    (confidence_breakdown.get("state_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "model_confidence": float(
                    (confidence_breakdown.get("model_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "pricing_confidence": float(
                    (confidence_breakdown.get("pricing_confidence") or {}).get("value", 0.0)
                    or 0.0
                ),
                "analog_confidence": float(
                    (confidence_breakdown.get("analog_confidence") or {}).get("value", 0.0) or 0.0
                ),
                "abstain": True,
                "downgrade_reason_category": decision_output.get("downgrade_reason_category"),
                "invalidation_triggers": [],
                "critic_reason_codes": (decision_output.get("critic_outcome") or {}).get(
                    "critic_reason_codes",
                    [],
                ),
                "evidence_hash": evidence_hash,
                "freshness_summary": _jsonish(context_pack.freshness_flags),
                "transform_loaded_at": loaded_at,
            }
        )
        override_rows.append(
            {
                "forecast_id": forecast_id,
                "ticker": context_pack.ticker,
                "as_of": as_of,
                "override_applied": override_applied,
                "override_reason": override_reason or None,
                "requested_mode": str(decision_output.get("requested_mode") or context_pack.mode),
                "resolved_mode": str(decision_output.get("resolved_mode") or context_pack.mode),
                "trust_tier": str(decision_output.get("trust_tier") or context_pack.trust_tier),
                "downgrade_reason_category": decision_output.get("downgrade_reason_category"),
                "critic_reason_codes": (decision_output.get("critic_outcome") or {}).get(
                    "critic_reason_codes",
                    [],
                ),
                "evidence_used": critic.get("missing_evidence") or [],
                "final_direction": "neutral",
                "transform_loaded_at": loaded_at,
            }
        )

    outcome_frame = pd.DataFrame(outcome_rows)
    override_frame = pd.DataFrame(override_rows)
    outcome_path = warehouse_partition_path(
        settings,
        domain="forecast/outcomes",
        partition_date=trade_date,
        stem=f"forecast_outcomes_{context_pack.ticker}_{trade_date.isoformat()}",
    )
    override_path = warehouse_partition_path(
        settings,
        domain="forecast/override_log",
        partition_date=trade_date,
        stem=f"llm_override_log_{context_pack.ticker}_{trade_date.isoformat()}",
    )
    write_parquet(outcome_frame, outcome_path)
    write_parquet(override_frame, override_path)
    return [outcome_path], [override_path]

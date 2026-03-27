from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ai_analyst.causal.entity_normalization import build_event_relations, normalize_event_entities
from ai_analyst.causal.governance import build_event_governance_rows
from ai_analyst.events.ontology import infer_theme


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _to_timestamp(value: Any, *, fallback: datetime) -> datetime:
    if value is None or value == "":
        return fallback
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return fallback
    return parsed.to_pydatetime()


def _event_id(source: str, topic: str, event_time: datetime, raw_ref: str | None) -> str:
    payload = f"{source}|{topic}|{event_time.isoformat()}|{raw_ref or ''}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def normalize_event_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    ingest_time: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    ingest_time = ingest_time or datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    narrative_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    loaded_at = datetime.now(tz=UTC)

    for record in frame.to_dict(orient="records"):
        topic = str(
            record.get("topic") or record.get("title") or record.get("name") or "event"
        ).strip()
        event_time = _to_timestamp(
            record.get("event_time") or record.get("issued_at"),
            fallback=ingest_time,
        )
        event_id = str(
            record.get("event_id") or _event_id(source, topic, event_time, record.get("raw_ref"))
        )
        event_family = record.get("event_family")
        theme = infer_theme(str(event_family) if event_family else None, record.get("theme"))
        affected_entities = _ensure_list(record.get("affected_entities"))
        affected_sectors = _ensure_list(record.get("affected_sectors"))
        affected_commodities = _ensure_list(record.get("affected_commodities"))

        normalized_record = {
            "event_id": event_id,
            "event_time": event_time,
            "ingest_time": _to_timestamp(record.get("ingest_time"), fallback=ingest_time),
            "source": source,
            "topic": topic,
            "event_family": event_family,
            "theme": theme,
            "region": record.get("region"),
            "geography": record.get("geography") or record.get("region"),
            "severity": float(record.get("severity", 0.0) or 0.0),
            "confidence": float(record.get("confidence", 0.0) or 0.0),
            "novelty": float(record.get("novelty", 0.0) or 0.0),
            "duration_hours": float(record.get("duration_hours", 0.0) or 0.0),
            "market_relevance": float(record.get("market_relevance", 0.0) or 0.0),
            "affected_commodities": affected_commodities,
            "affected_sectors": affected_sectors,
            "affected_entities": affected_entities,
            "raw_ref": record.get("raw_ref"),
            "transform_loaded_at": loaded_at,
        }
        rows.append(normalized_record)

        normalized_entities = normalize_event_entities(normalized_record)
        for entity in normalized_entities:
            entity_rows.append(
                {
                    "event_id": event_id,
                    "entity_type": entity.entity_type,
                    "entity_id": entity.entity_id,
                    "entity_name": entity.entity_name,
                    "entity_role": "affected",
                    "transform_loaded_at": loaded_at,
                }
            )
        relation_rows.extend(
            build_event_relations(
                event_id=event_id,
                record=normalized_record,
                normalized_entities=normalized_entities,
            )
        )
        source_row, narrative_row, evidence_row = build_event_governance_rows(
            source=source,
            record=normalized_record,
        )
        source_rows.append(source_row)
        narrative_rows.append(narrative_row)
        evidence_rows.append(evidence_row)

    return (
        pd.DataFrame(rows),
        pd.DataFrame(entity_rows),
        pd.DataFrame(relation_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(narrative_rows),
        pd.DataFrame(evidence_rows),
    )

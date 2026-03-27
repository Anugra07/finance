from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.events.normalization import normalize_event_frame
from ai_analyst.utils.dates import utc_now
from ai_analyst.utils.http import get_json
from ai_analyst.utils.io import read_json, write_json, write_parquet
from ai_analyst.warehouse.layout import (
    raw_snapshot_path,
    snapshot_time_from_path,
    warehouse_partition_path,
)

logger = logging.getLogger(__name__)


def _iter_raw_json(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _parse_epoch_ms(value: object, *, fallback: datetime) -> datetime:
    if value in (None, ""):
        return fallback
    try:
        numeric = float(value)
        if numeric <= 0:
            return fallback
        # World Monitor mixes second and millisecond epochs across feeds.
        if numeric >= 1.0e11:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=UTC)
    except (TypeError, ValueError, OSError):
        return fallback


def _severity_from_label(label: str | None) -> float:
    mapping = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.55,
        "low": 0.3,
    }
    if not label:
        return 0.5
    normalized = label.lower()
    for key, value in mapping.items():
        if key in normalized:
            return value
    return 0.5


class WorldMonitorClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()

    def _url(self, endpoint: str) -> str:
        return f"{self.settings.worldmonitor_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _safe_get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = get_json(self.session, self._url(endpoint), params=params)
        except requests.RequestException as exc:
            logger.warning("World Monitor request failed for %s: %s", endpoint, exc)
            return fallback or {}
        if not isinstance(response, dict):
            logger.warning("World Monitor returned non-dict payload for %s", endpoint)
            return fallback or {}
        return response

    def collect_snapshot(
        self,
        *,
        snapshot_at: datetime | None = None,
        max_items: int = 50,
    ) -> Path:
        snapshot_at = snapshot_at or utc_now()
        payload = {
            "collected_at": snapshot_at.isoformat(),
            "base_url": self.settings.worldmonitor_base_url,
            "market_sector_summary": self._safe_get(
                "/api/market/v1/get-sector-summary",
                params={"period": "1d"},
                fallback={"sectors": []},
            ),
            "market_etf_flows": self._safe_get(
                "/api/market/v1/list-etf-flows",
                fallback={"summary": {}, "etfs": []},
            ),
            "sanctions_pressure": self._safe_get(
                "/api/sanctions/v1/list-sanctions-pressure",
                params={"max_items": max_items},
                fallback={"entries": [], "countries": [], "programs": []},
            ),
            "navigational_warnings": self._safe_get(
                "/api/maritime/v1/list-navigational-warnings",
                params={"page_size": max_items},
                fallback={"warnings": []},
            ),
            "cyber_threats": self._safe_get(
                "/api/cyber/v1/list-cyber-threats",
                params={
                    "page_size": max_items,
                    "start": int((snapshot_at - timedelta(days=7)).timestamp() * 1000),
                    "end": int(snapshot_at.timestamp() * 1000),
                },
                fallback={"threats": [], "pagination": {}},
            ),
        }
        path = raw_snapshot_path(
            self.settings,
            source="worldmonitor",
            stem="context",
            snapshot_at=snapshot_at,
        )
        write_json(path, payload)
        return path


def _payload_to_event_frame(payload: dict[str, Any], *, snapshot_at: datetime) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    sanctions = payload.get("sanctions_pressure", {}) or {}
    for country in sanctions.get("countries", []) or []:
        event_time = _parse_epoch_ms(sanctions.get("datasetDate"), fallback=snapshot_at)
        entry_count = float(country.get("entryCount", 0) or 0)
        new_entry_count = float(country.get("newEntryCount", 0) or 0)
        severity = min(1.0, 0.15 + (new_entry_count / max(entry_count, 1.0)))
        rows.append(
            {
                "event_time": event_time,
                "ingest_time": snapshot_at,
                "topic": (
                    "Sanctions pressure: "
                    f"{country.get('countryName') or country.get('countryCode')}"
                ),
                "event_family": "sanctions",
                "theme": "sanctions_pressure",
                "region": country.get("countryCode"),
                "geography": country.get("countryName"),
                "severity": severity,
                "confidence": 0.85,
                "novelty": min(1.0, new_entry_count / max(entry_count, 1.0)),
                "duration_hours": 24.0 * 30.0,
                "market_relevance": min(1.0, 0.25 + (entry_count / 1000.0)),
                "affected_commodities": [],
                "affected_sectors": ["Energy", "Industrials", "Materials", "Financials"],
                "affected_entities": (
                    [country.get("countryName")] if country.get("countryName") else []
                ),
                "raw_ref": "sanctions_pressure.countries",
            }
        )

    warnings = payload.get("navigational_warnings", {}) or {}
    for warning in warnings.get("warnings", []) or []:
        issued_at = _parse_epoch_ms(warning.get("issuedAt"), fallback=snapshot_at)
        expires_at = _parse_epoch_ms(
            warning.get("expiresAt"),
            fallback=issued_at + timedelta(hours=24),
        )
        duration_hours = max((expires_at - issued_at).total_seconds() / 3600.0, 1.0)
        rows.append(
            {
                "event_time": issued_at,
                "ingest_time": snapshot_at,
                "topic": warning.get("title") or "Navigational warning",
                "event_family": "shipping_disruption",
                "theme": "shipping_stress",
                "region": warning.get("area"),
                "geography": warning.get("area"),
                "severity": 0.7,
                "confidence": 0.80,
                "novelty": 0.65 if (snapshot_at - issued_at) <= timedelta(days=2) else 0.25,
                "duration_hours": duration_hours,
                "market_relevance": 0.75,
                "affected_commodities": ["oil", "lng"],
                "affected_sectors": ["Energy", "Industrials", "Consumer Discretionary"],
                "affected_entities": [warning.get("authority")] if warning.get("authority") else [],
                "raw_ref": warning.get("id"),
            }
        )

    cyber = payload.get("cyber_threats", {}) or {}
    for threat in cyber.get("threats", []) or []:
        first_seen = _parse_epoch_ms(threat.get("firstSeenAt"), fallback=snapshot_at)
        last_seen = _parse_epoch_ms(threat.get("lastSeenAt"), fallback=first_seen)
        rows.append(
            {
                "event_time": last_seen,
                "ingest_time": snapshot_at,
                "topic": threat.get("indicator") or threat.get("id") or "Cyber threat",
                "event_family": "cyber_attack",
                "theme": "cyber_infra_risk",
                "region": threat.get("country"),
                "geography": threat.get("country"),
                "severity": _severity_from_label(threat.get("severity")),
                "confidence": 0.70,
                "novelty": min(
                    1.0,
                    max((last_seen - first_seen).total_seconds() / 86400.0, 0.0) / 7.0,
                ),
                "duration_hours": max((last_seen - first_seen).total_seconds() / 3600.0, 1.0),
                "market_relevance": 0.55,
                "affected_commodities": [],
                "affected_sectors": [
                    "Information Technology",
                    "Industrials",
                    "Communication Services",
                ],
                "affected_entities": [tag for tag in threat.get("tags", []) or [] if tag],
                "raw_ref": threat.get("id"),
            }
        )

    return pd.DataFrame(rows)


def transform_worldmonitor(
    settings: Settings,
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path], list[Path]]:
    normalized_paths: list[Path] = []
    entity_paths: list[Path] = []
    relation_paths: list[Path] = []
    source_assessment_paths: list[Path] = []
    narrative_risk_paths: list[Path] = []
    evidence_catalog_paths: list[Path] = []

    for path in _iter_raw_json(settings.raw_root / "worldmonitor"):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        raw_frame = _payload_to_event_frame(payload, snapshot_at=snapshot_at)
        normalized, entities, relations, source_assessment, narrative_risk, evidence_catalog = (
            normalize_event_frame(
            raw_frame,
            source="worldmonitor",
            ingest_time=snapshot_at,
            )
        )

        if not normalized.empty:
            out_path = warehouse_partition_path(
                settings,
                domain="events/normalized",
                partition_date=snapshot_at.date(),
                stem=path.stem,
            )
            write_parquet(normalized, out_path)
            normalized_paths.append(out_path)
        if not entities.empty:
            out_path = warehouse_partition_path(
                settings,
                domain="events/entities",
                partition_date=snapshot_at.date(),
                stem=f"{path.stem}_entities",
            )
            write_parquet(entities, out_path)
            entity_paths.append(out_path)
        if not relations.empty:
            out_path = warehouse_partition_path(
                settings,
                domain="events/relations",
                partition_date=snapshot_at.date(),
                stem=f"{path.stem}_relations",
            )
            write_parquet(relations, out_path)
            relation_paths.append(out_path)
        if not source_assessment.empty:
            out_path = warehouse_partition_path(
                settings,
                domain="events/source_assessment",
                partition_date=snapshot_at.date(),
                stem=f"{path.stem}_source_assessment",
            )
            write_parquet(source_assessment, out_path)
            source_assessment_paths.append(out_path)
        if not narrative_risk.empty:
            out_path = warehouse_partition_path(
                settings,
                domain="events/narrative_risk",
                partition_date=snapshot_at.date(),
                stem=f"{path.stem}_narrative_risk",
            )
            write_parquet(narrative_risk, out_path)
            narrative_risk_paths.append(out_path)
        if not evidence_catalog.empty:
            out_path = warehouse_partition_path(
                settings,
                domain="events/evidence_catalog",
                partition_date=snapshot_at.date(),
                stem=f"{path.stem}_evidence_catalog",
            )
            write_parquet(evidence_catalog, out_path)
            evidence_catalog_paths.append(out_path)

    return (
        normalized_paths,
        entity_paths,
        relation_paths,
        source_assessment_paths,
        narrative_risk_paths,
        evidence_catalog_paths,
    )

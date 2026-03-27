from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.utils.dates import utc_now
from ai_analyst.utils.http import get_json
from ai_analyst.utils.io import read_json, write_json, write_parquet
from ai_analyst.warehouse.database import connect
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


def _normalize_cik(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits.zfill(10) if digits else ""


def _normalize_accession(value: object) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits or None


def _to_date(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _to_float(value: object) -> float | None:
    try:
        if value in (None, "", "."):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_relation_ciks(
    settings: Settings,
    *,
    relation_name: str,
    limit: int | None = None,
) -> list[str]:
    conn = connect(settings)
    try:
        frame = conn.execute(
            f"""
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_at DESC) AS row_num
                FROM {relation_name}
                WHERE cik IS NOT NULL
                  AND TRIM(CAST(cik AS VARCHAR)) <> ''
            )
            SELECT DISTINCT cik
            FROM ranked
            WHERE row_num = 1
            ORDER BY cik
            """
        ).df()
    finally:
        conn.close()

    ciks = [_normalize_cik(value) for value in frame.get("cik", pd.Series(dtype="object"))]
    ciks = [cik for cik in ciks if cik]
    if limit is not None:
        return ciks[:limit]
    return ciks


def latest_universe_ciks(settings: Settings, *, limit: int | None = None) -> list[str]:
    return _latest_relation_ciks(settings, relation_name="universe_membership", limit=limit)


def latest_v1_universe_ciks(settings: Settings, *, limit: int | None = None) -> list[str]:
    return _latest_relation_ciks(settings, relation_name="v1_universe", limit=limit)


class SecClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.require_sec_identity(),
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )
        self._min_interval_seconds = 1.0 / max(self.settings.sec_rate_limit_per_second, 1.0)
        self._last_request_monotonic = 0.0

    def _url(self, path: str) -> str:
        return f"{self.settings.sec_data_base_url.rstrip('/')}/{path.lstrip('/')}"

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_monotonic
        if elapsed < self._min_interval_seconds:
            time.sleep(self._min_interval_seconds - elapsed)
        self._last_request_monotonic = time.monotonic()

    def _get_json(self, path: str) -> Any:
        self._respect_rate_limit()
        return get_json(self.session, self._url(path))

    def _collect_endpoint(
        self,
        *,
        ciks: list[str],
        source: str,
        endpoint_builder: Callable[[str], Any],
        snapshot_at: datetime,
    ) -> list[Path]:
        outputs: list[Path] = []
        for raw_cik in ciks:
            cik = _normalize_cik(raw_cik)
            if not cik:
                continue
            try:
                payload = endpoint_builder(cik)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                logger.warning("SEC request failed for CIK %s (%s): %s", cik, status, exc)
                continue
            if isinstance(payload, dict):
                payload["cik"] = cik
                payload["collected_at"] = snapshot_at.isoformat()
            out_path = raw_snapshot_path(
                self.settings,
                source=source,
                stem=f"cik{cik}",
                snapshot_at=snapshot_at,
            )
            write_json(out_path, payload)
            outputs.append(out_path)
        return outputs

    def collect_submissions(
        self,
        ciks: list[str],
        *,
        snapshot_at: datetime | None = None,
    ) -> list[Path]:
        snapshot_at = snapshot_at or utc_now()
        return self._collect_endpoint(
            ciks=ciks,
            source="sec_submissions",
            endpoint_builder=lambda cik: self._get_json(f"/submissions/CIK{cik}.json"),
            snapshot_at=snapshot_at,
        )

    def collect_companyfacts(
        self,
        ciks: list[str],
        *,
        snapshot_at: datetime | None = None,
    ) -> list[Path]:
        snapshot_at = snapshot_at or utc_now()
        return self._collect_endpoint(
            ciks=ciks,
            source="sec_companyfacts",
            endpoint_builder=lambda cik: self._get_json(f"/api/xbrl/companyfacts/CIK{cik}.json"),
            snapshot_at=snapshot_at,
        )


def _recent_submissions_frame(
    payload: dict[str, Any],
    *,
    snapshot_at: datetime,
    source_snapshot: str,
) -> pd.DataFrame:
    recent = (payload.get("filings") or {}).get("recent") or {}
    if not isinstance(recent, dict) or not recent:
        return pd.DataFrame()

    frame = pd.DataFrame(recent)
    if frame.empty:
        return frame

    transform_loaded_at = pd.Timestamp.now(tz=UTC)
    out = pd.DataFrame(
        {
            "cik": _normalize_cik(payload.get("cik")),
            "company_name": payload.get("name"),
            "accession_number": frame.get("accessionNumber", pd.Series(dtype="object")).map(
                _normalize_accession
            ),
            "form": frame.get("form"),
            "filing_date": pd.to_datetime(frame.get("filingDate"), errors="coerce").dt.date,
            "acceptance_datetime": pd.to_datetime(
                frame.get("acceptanceDateTime"), utc=True, errors="coerce"
            ),
            "primary_document": frame.get("primaryDocument"),
            "primary_doc_description": frame.get("primaryDocDescription"),
            "snapshot_at": snapshot_at,
            "source_snapshot": source_snapshot,
            "transform_loaded_at": transform_loaded_at,
        }
    )
    return out.dropna(subset=["accession_number"])


def transform_submissions(settings: Settings) -> tuple[list[Path], list[Path]]:
    submissions_outputs: list[Path] = []
    filing_index_outputs: list[Path] = []

    for path in _iter_raw_json(settings.raw_root / "sec_submissions"):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        frame = _recent_submissions_frame(
            payload,
            snapshot_at=snapshot_at,
            source_snapshot=str(path),
        )
        if frame.empty:
            continue

        submissions_path = warehouse_partition_path(
            settings,
            domain="edgar/submissions",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(frame, submissions_path)
        submissions_outputs.append(submissions_path)

        filing_index = frame[
            [
                "cik",
                "accession_number",
                "form",
                "filing_date",
                "acceptance_datetime",
                "primary_document",
                "primary_doc_description",
                "snapshot_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ].copy()
        filing_index_path = warehouse_partition_path(
            settings,
            domain="edgar/filing_index",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(filing_index, filing_index_path)
        filing_index_outputs.append(filing_index_path)

    return submissions_outputs, filing_index_outputs


def _companyfacts_frame(
    payload: dict[str, Any],
    *,
    snapshot_at: datetime,
    source_snapshot: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    facts = payload.get("facts") or {}
    entity_name = payload.get("entityName") or payload.get("name")
    cik = _normalize_cik(payload.get("cik"))
    transform_loaded_at = pd.Timestamp.now(tz=UTC)

    if not isinstance(facts, dict):
        return pd.DataFrame()

    for taxonomy, metrics in facts.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_payload in metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            units = metric_payload.get("units") or {}
            if not isinstance(units, dict):
                continue
            for unit, observations in units.items():
                if not isinstance(observations, list):
                    continue
                for observation in observations:
                    if not isinstance(observation, dict):
                        continue
                    rows.append(
                        {
                            "cik": cik,
                            "entity_name": entity_name,
                            "taxonomy": taxonomy,
                            "metric_name": metric_name,
                            "unit": unit,
                            "value": _to_float(observation.get("val")),
                            "period_end": _to_date(observation.get("end")),
                            "filing_date": _to_date(observation.get("filed")),
                            "frame": observation.get("frame"),
                            "fy": pd.to_numeric(observation.get("fy"), errors="coerce"),
                            "fp": observation.get("fp"),
                            "form": observation.get("form"),
                            "snapshot_at": snapshot_at,
                            "source_snapshot": source_snapshot,
                            "transform_loaded_at": transform_loaded_at,
                        }
                    )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if "fy" in frame.columns:
        frame["fy"] = frame["fy"].astype("Int64")
    return frame


def transform_companyfacts(settings: Settings) -> list[Path]:
    outputs: list[Path] = []
    for path in _iter_raw_json(settings.raw_root / "sec_companyfacts"):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        frame = _companyfacts_frame(
            payload,
            snapshot_at=snapshot_at,
            source_snapshot=str(path),
        )
        if frame.empty:
            continue
        out_path = warehouse_partition_path(
            settings,
            domain="edgar/companyfacts",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(frame, out_path)
        outputs.append(out_path)
    return outputs

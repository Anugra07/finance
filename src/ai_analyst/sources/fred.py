from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.utils.dates import utc_now
from ai_analyst.utils.http import get_json
from ai_analyst.utils.io import read_json, write_json, write_parquet
from ai_analyst.warehouse.layout import (
    raw_snapshot_path,
    snapshot_time_from_path,
    warehouse_partition_path,
)

EARLIEST_REALTIME = "1776-07-04"
LATEST_REALTIME = "9999-12-31"
MAX_VINTAGE_DATES_PER_REQUEST = 200


def _iter_raw_json(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _to_date(value: object) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed if not pd.isna(parsed) else pd.NaT


def _to_value(value: object) -> float | None:
    if value in (None, "", "."):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FredClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()

    def _url(self, endpoint: str) -> str:
        return f"{self.settings.fred_api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _fetch_observations(
        self,
        *,
        series_id: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        vintage_dates: list[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.settings.require_fred(),
            "file_type": "json",
        }
        if realtime_start:
            params["realtime_start"] = realtime_start
        if realtime_end:
            params["realtime_end"] = realtime_end
        if vintage_dates:
            params["vintage_dates"] = ",".join(vintage_dates)
        return get_json(self.session, self._url("/series/observations"), params=params)

    def _fetch_vintage_dates(self, *, series_id: str) -> list[str]:
        payload = get_json(
            self.session,
            self._url("/series/vintagedates"),
            params={
                "series_id": series_id,
                "api_key": self.settings.require_fred(),
                "file_type": "json",
                "realtime_start": EARLIEST_REALTIME,
                "realtime_end": LATEST_REALTIME,
            },
        )
        return [str(value) for value in payload.get("vintage_dates", []) or [] if value]

    def collect_current_series(
        self,
        series_ids: list[str],
        *,
        snapshot_at: datetime | None = None,
    ) -> list[Path]:
        snapshot_at = snapshot_at or utc_now()
        outputs: list[Path] = []
        for series_id in series_ids:
            payload = self._fetch_observations(series_id=series_id)
            payload["series_id"] = series_id
            payload["collected_at"] = snapshot_at.isoformat()
            path = raw_snapshot_path(
                self.settings,
                source="fred_current",
                stem=series_id.lower(),
                snapshot_at=snapshot_at,
            )
            write_json(path, payload)
            outputs.append(path)
        return outputs

    def collect_vintages(
        self,
        series_ids: list[str],
        *,
        snapshot_at: datetime | None = None,
    ) -> list[Path]:
        snapshot_at = snapshot_at or utc_now()
        outputs: list[Path] = []
        for series_id in series_ids:
            vintage_dates = self._fetch_vintage_dates(series_id=series_id)
            if not vintage_dates:
                continue
            for chunk_no, start in enumerate(
                range(0, len(vintage_dates), MAX_VINTAGE_DATES_PER_REQUEST),
                start=1,
            ):
                chunk = vintage_dates[start : start + MAX_VINTAGE_DATES_PER_REQUEST]
                payload = self._fetch_observations(
                    series_id=series_id,
                    vintage_dates=chunk,
                )
                payload["series_id"] = series_id
                payload["vintage_dates"] = chunk
                payload["chunk_no"] = chunk_no
                payload["collected_at"] = snapshot_at.isoformat()
                path = raw_snapshot_path(
                    self.settings,
                    source="alfred_vintages",
                    stem=f"{series_id.lower()}_chunk{chunk_no:03d}",
                    snapshot_at=snapshot_at,
                )
                write_json(path, payload)
                outputs.append(path)
        return outputs


def _observations_frame(
    payload: dict[str, Any],
    *,
    snapshot_at: datetime,
    known_at_from_realtime: bool,
) -> pd.DataFrame:
    series_id = str(payload.get("series_id") or payload.get("seriess_id") or "").strip()
    rows: list[dict[str, object]] = []

    for observation in payload.get("observations", []) or []:
        observation_date = _to_date(observation.get("date"))
        realtime_start = _to_date(observation.get("realtime_start"))
        realtime_end = _to_date(observation.get("realtime_end"))
        if pd.isna(observation_date):
            continue
        if known_at_from_realtime and not pd.isna(realtime_start):
            known_at = pd.Timestamp(realtime_start).tz_localize(UTC)
        else:
            known_at = snapshot_at
        rows.append(
            {
                "series_id": series_id,
                "observation_date": pd.Timestamp(observation_date).date(),
                "value": _to_value(observation.get("value")),
                "realtime_start": (
                    pd.Timestamp(realtime_start).date() if not pd.isna(realtime_start) else None
                ),
                "realtime_end": (
                    pd.Timestamp(realtime_end).date() if not pd.isna(realtime_end) else None
                ),
                "known_at": known_at,
                "source_snapshot": "",
                "transform_loaded_at": utc_now(),
            }
        )

    return pd.DataFrame(rows)


def transform_current(settings: Settings) -> list[Path]:
    outputs: list[Path] = []
    for path in _iter_raw_json(settings.raw_root / "fred_current"):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        frame = _observations_frame(
            payload,
            snapshot_at=snapshot_at,
            known_at_from_realtime=False,
        )
        if frame.empty:
            continue
        frame["source_snapshot"] = str(path)
        out_path = warehouse_partition_path(
            settings,
            domain="macro/current",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(frame, out_path)
        outputs.append(out_path)
    return outputs


def transform_vintages(settings: Settings) -> list[Path]:
    outputs: list[Path] = []
    for path in _iter_raw_json(settings.raw_root / "alfred_vintages"):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        frame = _observations_frame(
            payload,
            snapshot_at=snapshot_at,
            known_at_from_realtime=True,
        )
        if frame.empty:
            continue
        frame["source_snapshot"] = str(path)
        out_path = warehouse_partition_path(
            settings,
            domain="macro/vintages",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(frame, out_path)
        outputs.append(out_path)
    return outputs

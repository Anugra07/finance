from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

from ai_analyst.config import Settings
from ai_analyst.utils.io import ensure_dir


def normalize_snapshot_time(snapshot_at: datetime | None = None) -> datetime:
    snapshot_at = snapshot_at or datetime.now(tz=UTC)
    if snapshot_at.tzinfo is None:
        return snapshot_at.replace(tzinfo=UTC)
    return snapshot_at.astimezone(UTC)


def raw_snapshot_dir(settings: Settings, source: str, snapshot_at: datetime | None = None) -> Path:
    snapshot_at = normalize_snapshot_time(snapshot_at)
    return ensure_dir(settings.raw_root / source / snapshot_at.date().isoformat())


def raw_snapshot_path(
    settings: Settings,
    *,
    source: str,
    stem: str,
    snapshot_at: datetime | None = None,
) -> Path:
    snapshot_at = normalize_snapshot_time(snapshot_at)
    timestamp = snapshot_at.strftime("%Y%m%dT%H%M%SZ")
    return raw_snapshot_dir(settings, source, snapshot_at) / f"{stem}_{timestamp}.json"


def warehouse_partition_dir(
    settings: Settings,
    *,
    domain: str,
    partition_date: date,
) -> Path:
    return ensure_dir(
        settings.warehouse_root
        / domain
        / f"year={partition_date.year:04d}"
        / f"month={partition_date.month:02d}"
        / f"day={partition_date.day:02d}"
    )


def warehouse_partition_path(
    settings: Settings,
    *,
    domain: str,
    partition_date: date,
    stem: str,
) -> Path:
    return (
        warehouse_partition_dir(settings, domain=domain, partition_date=partition_date)
        / f"{stem}.parquet"
    )


def snapshot_time_from_path(path: Path) -> datetime:
    match = re.search(r"_(\d{8}T\d{6}Z)\.json$", path.name)
    if not match:
        raise ValueError(f"Could not parse snapshot timestamp from {path}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)

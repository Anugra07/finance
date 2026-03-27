from __future__ import annotations

import json
from pathlib import Path

from ai_analyst.config import Settings


def resolve_latest_report_path(settings: Settings) -> Path | None:
    latest_pointer = settings.reports_path / "nightly_latest.json"
    if latest_pointer.exists():
        return latest_pointer

    dated_reports = sorted(settings.reports_path.glob("nightly_ranked_report_*.json"))
    if dated_reports:
        return dated_reports[-1]

    generic_reports = sorted(settings.reports_path.glob("*.json"))
    return generic_reports[-1] if generic_reports else None


def persist_ranked_report(
    settings: Settings,
    *,
    report: dict[str, object],
    dated_path: Path,
) -> None:
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, default=str)
    dated_path.write_text(payload, encoding="utf-8")
    (settings.reports_path / "nightly_latest.json").write_text(payload, encoding="utf-8")

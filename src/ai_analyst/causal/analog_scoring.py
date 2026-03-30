from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ai_analyst.causal.versioning import ANALOG_MODEL_VERSION
from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path

HORIZONS = ("1-3 days", "1-3 weeks", "1-3 months")


@dataclass(slots=True)
class AnalogScoringArtifacts:
    analogs: pd.DataFrame
    paths: list[Path]


def _theme_vector(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    return {
        str(row["theme"]): float(row.get("intensity", 0.0) or 0.0)
        for _, row in frame.iterrows()
        if row.get("theme")
    }


def _score_theme_overlap(anchor: dict[str, float], other: dict[str, float]) -> float:
    if not anchor or not other:
        return 0.0
    keys = sorted(set(anchor) | set(other))
    anchor_total = sum(abs(anchor.get(key, 0.0)) for key in keys) or 1.0
    distance = sum(abs(anchor.get(key, 0.0) - other.get(key, 0.0)) for key in keys)
    return max(0.0, 1.0 - distance / anchor_total)


def _top_theme(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    row = frame.sort_values("intensity", ascending=False).iloc[0]
    return str(row["theme"]) if row.get("theme") is not None else None


def _top_geographies(events: pd.DataFrame, trade_date: date) -> set[str]:
    if events.empty:
        return set()
    frame = events.copy()
    frame["event_date"] = pd.to_datetime(frame["event_time"], utc=True).dt.date
    subset = frame.loc[frame["event_date"] == trade_date]
    if subset.empty or "geography" not in subset.columns:
        return set()
    return set(subset["geography"].dropna().astype(str).head(5).tolist())


def _top_routes(events: pd.DataFrame, trade_date: date) -> set[str]:
    if events.empty:
        return set()
    frame = events.copy()
    frame["event_date"] = pd.to_datetime(frame["event_time"], utc=True).dt.date
    subset = frame.loc[frame["event_date"] == trade_date]
    if subset.empty:
        return set()
    if "geography" in subset.columns:
        return set(
            subset["geography"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .head(5)
            .tolist()
        )
    return set()


def _similarity_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    union = len(left | right)
    return overlap / union if union else 0.0


def _regime_lookup(theme_regimes: pd.DataFrame) -> dict[date, str]:
    if theme_regimes.empty:
        return {}
    frame = theme_regimes.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.date
    frame = frame.sort_values(["as_of_date", "regime_score"], ascending=[True, False])
    return {
        trade_date: str(group.iloc[0]["regime_name"])
        for trade_date, group in frame.groupby("as_of_date", sort=False)
    }


def _horizon_weights(horizon: str) -> dict[str, float]:
    if horizon == "1-3 days":
        return {
            "theme": 0.35,
            "region": 0.25,
            "route": 0.20,
            "regime": 0.10,
            "sector": 0.10,
        }
    if horizon == "1-3 months":
        return {
            "theme": 0.25,
            "region": 0.10,
            "route": 0.10,
            "regime": 0.30,
            "sector": 0.25,
        }
    return {
        "theme": 0.30,
        "region": 0.15,
        "route": 0.15,
        "regime": 0.20,
        "sector": 0.20,
    }


def build_horizon_analog_matches(
    *,
    as_of_date: date,
    theme_daily: pd.DataFrame,
    sector_rankings: pd.DataFrame,
    events: pd.DataFrame | None = None,
    theme_regimes: pd.DataFrame | None = None,
    limit: int = 3,
) -> list[dict[str, object]]:
    if theme_daily.empty:
        return []
    events_frame = events if events is not None else pd.DataFrame()
    regimes_frame = theme_regimes if theme_regimes is not None else pd.DataFrame()

    frame = theme_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    anchor = frame.loc[frame["date"] == as_of_date]
    if anchor.empty:
        latest_date = max(frame["date"])
        anchor = frame.loc[frame["date"] == latest_date]
        as_of_date = latest_date

    anchor_vector = _theme_vector(anchor)
    anchor_top_theme = _top_theme(anchor)
    anchor_geographies = _top_geographies(events_frame, as_of_date)
    anchor_routes = _top_routes(events_frame, as_of_date)
    sector_frame = sector_rankings.copy()
    sector_frame["as_of_date"] = (
        pd.to_datetime(sector_frame["as_of_date"]).dt.date if not sector_frame.empty else []
    )
    anchor_top_sector = None
    if not sector_frame.empty:
        current_sector = sector_frame.loc[sector_frame["as_of_date"] == as_of_date]
        if not current_sector.empty:
            anchor_top_sector = str(
                current_sector.sort_values("sector_score", ascending=False).iloc[0]["sector"]
            )
    regime_lookup = _regime_lookup(regimes_frame)
    anchor_regime = regime_lookup.get(as_of_date)

    matches: list[dict[str, object]] = []
    for other_date, group in frame.groupby("date"):
        if other_date == as_of_date:
            continue
        other_vector = _theme_vector(group)
        theme_overlap = _score_theme_overlap(anchor_vector, other_vector)
        region_similarity = _similarity_ratio(
            anchor_geographies,
            _top_geographies(events_frame, other_date),
        )
        route_similarity = _similarity_ratio(
            anchor_routes,
            _top_routes(events_frame, other_date),
        )
        regime_similarity = (
            1.0 if anchor_regime and regime_lookup.get(other_date) == anchor_regime else 0.0
        )
        top_theme_match = 1.0 if _top_theme(group) == anchor_top_theme and anchor_top_theme else 0.0
        top_sector_match = 0.0
        if anchor_top_sector and not sector_frame.empty:
            other_sector = sector_frame.loc[sector_frame["as_of_date"] == other_date]
            if not other_sector.empty:
                top_sector_match = (
                    1.0
                    if str(
                        other_sector.sort_values("sector_score", ascending=False).iloc[0]["sector"]
                    )
                    == anchor_top_sector
                    else 0.0
                )
        important_differences: list[str] = []
        if top_theme_match == 0.0:
            important_differences.append("top_theme_differs")
        if top_sector_match == 0.0:
            important_differences.append("top_sector_differs")
        for horizon in HORIZONS:
            weights = _horizon_weights(horizon)
            similarity = round(
                theme_overlap * weights["theme"]
                + region_similarity * weights["region"]
                + route_similarity * weights["route"]
                + regime_similarity * weights["regime"]
                + top_sector_match * weights["sector"],
                4,
            )
            misleading = similarity < 0.45
            failure_risk = round(max(0.0, 1.0 - similarity), 4)
            analogy_strength = (
                "strong" if similarity >= 0.75 else "moderate" if similarity >= 0.55 else "weak"
            )
            non_analog_reasons: list[str] = []
            if misleading:
                non_analog_reasons.append("weak_similarity")
            if not regime_similarity:
                non_analog_reasons.append("regime_differs")
            if not route_similarity and horizon == "1-3 days":
                non_analog_reasons.append("route_structure_differs")
            matches.append(
                {
                    "as_of_date": as_of_date,
                    "analog_type": horizon,
                    "anchor_key": f"theme_state:{as_of_date.isoformat()}",
                    "analog_key": f"theme_state:{other_date.isoformat()}",
                    "analog_start": other_date,
                    "analog_end": other_date,
                    "similarity_score": similarity,
                    "thesis": f"{horizon} analog built from theme overlap and sector transmission.",
                    "raw_ref": f"analog_model_version={ANALOG_MODEL_VERSION}",
                    "important_differences": important_differences,
                    "analogy_strength": analogy_strength,
                    "misleading_analogy": misleading,
                    "non_analog_reasons": non_analog_reasons,
                    "analogy_failure_risk": failure_risk,
                    "forward_outcomes": {
                        "top_theme": _top_theme(group),
                        "theme_overlap": theme_overlap,
                        "region_similarity": region_similarity,
                        "route_similarity": route_similarity,
                        "regime_similarity": regime_similarity,
                    },
                    "transform_loaded_at": datetime.now(tz=UTC),
                }
            )
    matches = sorted(
        matches, key=lambda item: (item["analog_type"], item["similarity_score"]), reverse=True
    )
    limited: list[dict[str, object]] = []
    for horizon in HORIZONS:
        limited.extend(
            sorted(
                [row for row in matches if row["analog_type"] == horizon],
                key=lambda item: item["similarity_score"],
                reverse=True,
            )[:limit]
        )
    return limited


def materialize_historical_analogs(
    settings: Settings,
    *,
    as_of_date: date | None = None,
) -> AnalogScoringArtifacts:
    conn = connect(settings)
    try:
        theme_daily = conn.execute("SELECT * FROM theme_intensity_daily").df()
        sector_rankings = conn.execute("SELECT * FROM sector_rankings").df()
        events = conn.execute("SELECT * FROM normalized_events").df()
        theme_regimes = conn.execute("SELECT * FROM theme_regimes").df()
    finally:
        conn.close()

    if theme_daily.empty:
        return AnalogScoringArtifacts(analogs=pd.DataFrame(), paths=[])
    resolved_date = as_of_date or pd.to_datetime(theme_daily["date"]).dt.date.max()
    frame = pd.DataFrame(
        build_horizon_analog_matches(
            as_of_date=resolved_date,
            theme_daily=theme_daily,
            sector_rankings=sector_rankings,
            events=events,
            theme_regimes=theme_regimes,
        )
    )
    paths: list[Path] = []
    if frame.empty:
        return AnalogScoringArtifacts(analogs=frame, paths=paths)
    out_path = warehouse_partition_path(
        settings,
        domain="analogs/historical",
        partition_date=resolved_date,
        stem=f"historical_analogs_{resolved_date.isoformat()}",
    )
    write_parquet(frame, out_path)
    paths.append(out_path)
    return AnalogScoringArtifacts(analogs=frame, paths=paths)

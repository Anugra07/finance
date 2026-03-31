from __future__ import annotations

from datetime import datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.events.context import LocalMacroContextSource
from ai_analyst.warehouse.database import connect


def _stock_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "ticker": row["ticker"],
                "sector": row.get("sector"),
                "market_code": row.get("market_code"),
                "horizon_days": int(row.get("horizon_days", 5) or 5),
                "expected_excess_alpha_rank": float(row["prediction"]),
                "probability_of_outperforming": float(row["prob_outperform"]),
                "confidence_score": float(row["confidence_score"]),
                "observed_excess_alpha_5d": float(
                    row.get("observed_excess_alpha", row.get("excess_alpha_5d", 0.0)) or 0.0
                ),
                "split_no": int(row.get("split_no", 0)),
            }
        )
    return rows


def _prediction_sector_rows(source: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in source.iterrows():
        rows.append(
            {
                "sector": row["sector"],
                "mean_prediction": float(row["mean_prediction"]),
                "stock_count": int(row["stock_count"]),
                "avg_confidence": float(row["avg_confidence"]),
            }
        )
    return rows


def _theme_rows(theme_frame: pd.DataFrame, *, limit: int = 10) -> list[dict[str, object]]:
    if theme_frame.empty:
        return []
    rows: list[dict[str, object]] = []
    ordered = theme_frame.sort_values(["intensity", "theme"], ascending=[False, True]).head(limit)
    for _, row in ordered.iterrows():
        latest_event_time = row.get("latest_event_time")
        rows.append(
            {
                "theme": row.get("theme"),
                "intensity": float(row.get("intensity", 0.0)),
                "event_count": int(row.get("event_count", 0) or 0),
                "avg_severity": float(row.get("avg_severity", 0.0) or 0.0),
                "avg_novelty": float(row.get("avg_novelty", 0.0) or 0.0),
                "latest_event_time": (
                    pd.to_datetime(latest_event_time).isoformat()
                    if pd.notna(latest_event_time)
                    else None
                ),
            }
        )
    return rows


def _sector_opportunity_rows(source: pd.DataFrame, *, limit: int = 5) -> list[dict[str, object]]:
    if source.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, row in source.head(limit).iterrows():
        supporting_themes = row.get("supporting_themes")
        if isinstance(supporting_themes, pd.Series):
            supporting_themes = supporting_themes.tolist()
        rows.append(
            {
                "sector": row.get("sector"),
                "sector_score": float(row.get("sector_score", 0.0) or 0.0),
                "context_shock": float(row.get("context_shock", 0.0) or 0.0),
                "finance_score": float(row.get("finance_score", 0.0) or 0.0),
                "top_theme": row.get("top_theme"),
                "supporting_themes": (
                    supporting_themes if isinstance(supporting_themes, list) else []
                ),
                "solution_bucket": row.get("solution_bucket"),
            }
        )
    return rows


def _solution_rows(source: pd.DataFrame, *, limit: int = 10) -> list[dict[str, object]]:
    if source.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, row in source.head(limit).iterrows():
        rows.append(
            {
                "theme": row.get("theme"),
                "solution_type": row.get("solution_type"),
                "label": row.get("label"),
                "beneficiary_sector": row.get("beneficiary_sector"),
                "hedge_role": row.get("hedge_role"),
                "rationale": row.get("rationale"),
                "theme_intensity": float(row.get("intensity", 0.0) or 0.0),
            }
        )
    return rows


def _analog_rows(source: pd.DataFrame, *, limit: int = 6) -> list[dict[str, object]]:
    if source.empty:
        return []
    rows: list[dict[str, object]] = []
    ordered = source.sort_values(
        ["analog_type", "similarity_score"],
        ascending=[True, False],
    ).head(limit)
    for _, row in ordered.iterrows():
        rows.append(
            {
                "horizon": row.get("analog_type"),
                "analog_key": row.get("analog_key"),
                "similarity_score": float(row.get("similarity_score", 0.0) or 0.0),
                "analogy_strength": row.get("analogy_strength"),
                "misleading_analogy": bool(row.get("misleading_analogy")),
                "important_differences": row.get("important_differences") or [],
                "analogy_failure_risk": float(row.get("analogy_failure_risk", 0.0) or 0.0),
            }
        )
    return rows


def _calibration_summary(settings: Settings) -> dict[str, object]:
    conn = connect(settings)
    try:
        metrics = conn.execute(
            """
            WITH latest_date AS (
                SELECT MAX(as_of_date) AS as_of_date
                FROM forecast_calibration_metrics
            )
            SELECT m.*
            FROM forecast_calibration_metrics m
            INNER JOIN latest_date d ON m.as_of_date = d.as_of_date
            ORDER BY metric_name, metric_bucket
            """
        ).df()
        analogs = conn.execute(
            """
            WITH latest_date AS (
                SELECT MAX(as_of_date) AS as_of_date
                FROM historical_analogs
            )
            SELECT *
            FROM historical_analogs
            WHERE as_of_date = (SELECT as_of_date FROM latest_date)
            ORDER BY analog_type, similarity_score DESC
            """
        ).df()
        causal_state = conn.execute(
            """
            WITH latest_date AS (
                SELECT MAX(as_of_date) AS as_of_date
                FROM causal_state_daily
            )
            SELECT *
            FROM causal_state_daily
            WHERE as_of_date = (SELECT as_of_date FROM latest_date)
            ORDER BY state_key
            """
        ).df()
    finally:
        conn.close()

    summary = {
        "metrics": [],
        "top_analogs": _analog_rows(analogs),
        "pricing_disagreement": {},
    }
    if not metrics.empty:
        summary["metrics"] = metrics.to_dict(orient="records")
    if not causal_state.empty:
        state_lookup = {
            str(row["state_key"]): row["state_value"] for _, row in causal_state.iterrows()
        }
        summary["pricing_disagreement"] = {
            "pricing_geo_signal_strength": state_lookup.get("pricing_geo_signal_strength"),
            "pricing_mediator_confirmation": state_lookup.get("pricing_mediator_confirmation"),
            "pricing_market_response_strength": state_lookup.get(
                "pricing_market_response_strength"
            ),
            "pricing_divergence": state_lookup.get("pricing_divergence"),
        }
    return summary


def _geo_condition_summary(
    theme_frame: pd.DataFrame,
    recent_events: pd.DataFrame,
) -> dict[str, object]:
    event_lookup: dict[str, list[str]] = {}
    if not recent_events.empty and "theme" in recent_events.columns:
        deduped = recent_events.dropna(subset=["theme", "topic"]).copy()
        deduped["topic"] = deduped["topic"].astype(str)
        for theme, group in deduped.groupby("theme", sort=False):
            event_lookup[str(theme)] = group["topic"].drop_duplicates().head(2).tolist()

    top_condition_drivers: list[dict[str, object]] = []
    for item in _theme_rows(theme_frame, limit=5):
        top_condition_drivers.append(
            {
                **item,
                "supporting_topics": event_lookup.get(str(item.get("theme")), []),
            }
        )

    latest_event_time = None
    if not recent_events.empty and "event_time" in recent_events.columns:
        latest = pd.to_datetime(recent_events["event_time"], utc=True, errors="coerce").max()
        if pd.notna(latest):
            latest_event_time = latest.isoformat()

    geographies: list[str] = []
    if not recent_events.empty and "geography" in recent_events.columns:
        geographies = (
            recent_events["geography"].dropna().astype(str).drop_duplicates().head(5).tolist()
        )

    return {
        "top_condition_drivers": top_condition_drivers,
        "recent_event_count": int(len(recent_events)),
        "latest_event_time": latest_event_time,
        "active_geographies": geographies,
    }


def build_ranked_report(
    predictions: pd.DataFrame,
    *,
    as_of: datetime,
    limit: int = 25,
    settings: Settings | None = None,
) -> dict[str, object]:
    as_of_ts = pd.to_datetime(as_of)

    top_stocks: list[dict[str, object]] = []
    top_sectors: list[dict[str, object]] = []
    bottom_sectors: list[dict[str, object]] = []

    if not predictions.empty:
        frame = predictions.loc[pd.to_datetime(predictions["date"]) == as_of_ts].copy()
        if frame.empty:
            frame = predictions.sort_values("date").groupby("ticker", group_keys=False).tail(1)
        frame = frame.sort_values("prediction", ascending=False)
        sector_frame = (
            predictions.groupby("sector", as_index=False)
            .agg(
                mean_prediction=("prediction", "mean"),
                stock_count=("ticker", "nunique"),
                avg_confidence=("confidence_score", "mean"),
            )
            .sort_values(["mean_prediction", "sector"], ascending=[False, True])
        )
        top_stocks = _stock_rows(frame.head(limit))
        top_sectors = _prediction_sector_rows(sector_frame.head(5))
        bottom_sectors = _prediction_sector_rows(
            sector_frame.tail(5).sort_values(["mean_prediction", "sector"], ascending=[True, True])
        )

    report: dict[str, object] = {
        "as_of": as_of_ts.isoformat(),
        "top_stocks": top_stocks,
        "top_sectors": top_sectors,
        "bottom_sectors": bottom_sectors,
    }

    if settings is None:
        return report

    context = LocalMacroContextSource(settings)
    theme_intensities = context.get_theme_intensities(as_of=as_of_ts.to_pydatetime())
    recent_events = context.get_recent_events(as_of=as_of_ts.to_pydatetime())
    geo_sector_rankings = context.get_sector_rankings(as_of=as_of_ts.to_pydatetime(), limit=15)
    solution_ideas = context.get_solution_ideas(as_of=as_of_ts.to_pydatetime(), limit=10)

    booming_sectors = _sector_opportunity_rows(
        geo_sector_rankings.sort_values(["sector_score", "sector"], ascending=[False, True]),
        limit=5,
    )
    pressured_sectors = _sector_opportunity_rows(
        geo_sector_rankings.sort_values(["sector_score", "sector"], ascending=[True, True]),
        limit=5,
    )

    highlighted_sectors = {row["sector"] for row in top_sectors} | {
        row["sector"] for row in booming_sectors
    }
    stock_ranking_within_top_sectors = [
        row for row in top_stocks if row.get("sector") in highlighted_sectors
    ][:limit]

    report.update(
        {
            "geo_energy_condition_summary": _geo_condition_summary(
                theme_intensities,
                recent_events,
            ),
            "theme_intensity_dashboard": _theme_rows(theme_intensities),
            "sector_boom_stress_rankings": {
                "geo_beneficiaries": booming_sectors,
                "geo_pressured": pressured_sectors,
                "predicted_top_sectors": top_sectors,
                "predicted_bottom_sectors": bottom_sectors,
            },
            "beneficiary_hedge_solution_ideas": _solution_rows(solution_ideas),
            "stock_ranking_within_top_sectors": stock_ranking_within_top_sectors,
            "calibration_and_analog_summary": _calibration_summary(settings),
        }
    )
    return report

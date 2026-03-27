from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.llm.context_pack import ContextPackBuilder
from ai_analyst.llm.forecast import OllamaClient
from ai_analyst.llm.reasoning import run_decision_mode, run_research_mode
from ai_analyst.reporting.io import resolve_latest_report_path
from ai_analyst.reporting.nightly import build_ranked_report
from ai_analyst.warehouse.database import connect


def analyst_health_payload(settings: Settings) -> dict[str, object]:
    conn = connect(settings)
    try:
        freshness = {
            "prices": conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0],
            "macro": conn.execute("SELECT COUNT(*) FROM macro_vintages").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM normalized_events").fetchone()[0],
            "analogs": conn.execute("SELECT COUNT(*) FROM historical_analogs").fetchone()[0],
            "causal_state": conn.execute("SELECT COUNT(*) FROM causal_state_daily").fetchone()[0],
        }
    finally:
        conn.close()
    return {
        "status": "ok",
        "ollama_host": settings.ollama_host,
        "forecast_model": settings.ollama_forecast_model,
        "critic_model": settings.ollama_critic_model,
        "warehouse_counts": freshness,
    }


def analyst_context_payload(
    settings: Settings,
    *,
    ticker: str,
    as_of: datetime,
    mode: str = "research",
) -> dict[str, object]:
    pack = ContextPackBuilder(settings).build(ticker=ticker.upper(), as_of=as_of, mode=mode)
    return asdict(pack)


def analyst_research_payload(
    settings: Settings,
    *,
    ticker: str,
    as_of: datetime,
    model: str | None = None,
) -> dict[str, object]:
    pack = ContextPackBuilder(settings).build(ticker=ticker.upper(), as_of=as_of, mode="research")
    client = OllamaClient(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)
    return run_research_mode(pack, model=model or settings.ollama_forecast_model, client=client)


def analyst_decision_payload(
    settings: Settings,
    *,
    ticker: str,
    as_of: datetime,
    model: str | None = None,
    critic_model: str | None = None,
) -> dict[str, object]:
    pack = ContextPackBuilder(settings).build(ticker=ticker.upper(), as_of=as_of, mode="decision")
    client = OllamaClient(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)
    return run_decision_mode(
        pack,
        model=model or settings.ollama_forecast_model,
        critic_model=critic_model or settings.ollama_critic_model,
        client=client,
    )


def analyst_brief_payload(settings: Settings, *, as_of: datetime) -> dict[str, object]:
    report_path = resolve_latest_report_path(settings)
    if report_path is not None:
        return json.loads(report_path.read_text(encoding="utf-8"))
    return build_ranked_report(pd.DataFrame(), as_of=as_of.astimezone(UTC), settings=settings)

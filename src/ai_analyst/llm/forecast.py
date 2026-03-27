from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import requests

from ai_analyst.core.models import ContextPack

FORECAST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "thesis": {"type": "string"},
        "drivers": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "horizon": {"type": "string", "enum": ["5 trading days"]},
        "verdict": {
            "type": "string",
            "enum": ["outperform", "neutral", "underperform", "low_conviction"],
        },
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary",
        "thesis",
        "drivers",
        "risks",
        "confidence",
        "horizon",
        "verdict",
        "citations",
    ],
}

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strongest_counterarguments": {"type": "array", "items": {"type": "string"}},
        "missing_data_checks": {"type": "array", "items": {"type": "string"}},
        "disconfirming_evidence": {"type": "array", "items": {"type": "string"}},
        "confidence_adjustment": {"type": "number", "minimum": -1, "maximum": 0},
    },
    "required": [
        "strongest_counterarguments",
        "missing_data_checks",
        "disconfirming_evidence",
        "confidence_adjustment",
    ],
}

ALLOWED_VERDICTS = {"outperform", "neutral", "underperform", "low_conviction"}
ALLOWED_CITATIONS = {
    "market_snapshot.prices",
    "macro_snapshot.theme_intensities",
    "macro_snapshot.macro_rows",
    "top_events",
    "filing_excerpts",
    "freshness_flags",
    "sector_rankings",
    "solution_ideas",
    "context_pack",
}


class OllamaClient:
    def __init__(self, *, host: str = "http://localhost:11434", timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.session.post(
            f"{self.host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": response_schema,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response") or "").strip()
        if not text:
            raise ValueError("Ollama returned an empty response body.")
        return json.loads(text)


def _compact_context(context_pack: ContextPack) -> str:
    return json.dumps(asdict(context_pack), indent=2, default=str)


def _normalize_probability(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1.0 and numeric <= 100.0:
        numeric = numeric / 100.0
    return round(min(1.0, max(0.0, numeric)), 4)


def _normalize_critic_adjustment(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if abs(numeric) > 1.0 and abs(numeric) <= 100.0:
        numeric = numeric / 100.0
    numeric = -abs(numeric)
    return round(min(0.0, max(-1.0, numeric)), 4)


def _ensure_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_verdict(value: object) -> str:
    lowered = str(value or "").strip().lower().replace(" ", "_")
    if lowered in ALLOWED_VERDICTS:
        return lowered
    alias_map = {
        "buy": "outperform",
        "strong_buy": "outperform",
        "sell": "underperform",
        "strong_sell": "underperform",
        "hold": "neutral",
    }
    return alias_map.get(lowered, "low_conviction")


def _normalize_citations(value: object) -> list[str]:
    citations = _ensure_string_list(value)
    normalized: list[str] = []
    for citation in citations:
        lowered = citation.strip().lower()
        matched = next(
            (allowed for allowed in ALLOWED_CITATIONS if allowed.lower() in lowered),
            None,
        )
        if matched and matched not in normalized:
            normalized.append(matched)
    return normalized or ["context_pack"]


def normalize_forecast_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(payload.get("summary") or "").strip(),
        "thesis": str(payload.get("thesis") or "").strip(),
        "drivers": _ensure_string_list(payload.get("drivers")),
        "risks": _ensure_string_list(payload.get("risks")),
        "confidence": _normalize_probability(payload.get("confidence")),
        "horizon": "5 trading days",
        "verdict": _normalize_verdict(payload.get("verdict")),
        "citations": _normalize_citations(payload.get("citations")),
    }


def normalize_critic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "strongest_counterarguments": _ensure_string_list(
            payload.get("strongest_counterarguments")
        ),
        "missing_data_checks": _ensure_string_list(payload.get("missing_data_checks")),
        "disconfirming_evidence": _ensure_string_list(payload.get("disconfirming_evidence")),
        "confidence_adjustment": _normalize_critic_adjustment(payload.get("confidence_adjustment")),
    }


def build_forecast_prompt(context_pack: ContextPack) -> str:
    return (
        "You are a local research analyst. Use only the provided context pack. "
        "Do not invent facts. Produce a concise 5-day forecast as strict JSON.\n\n"
        "Required rules:\n"
        "- cite only themes, events, filings, rankings, or freshness flags present in the pack\n"
        "- confidence must be between 0 and 1\n"
        "- horizon must be exactly '5 trading days'\n"
        "- verdict must be one of outperform, neutral, underperform, low_conviction\n"
        "- citations should be short strings naming the context fields you used\n\n"
        f"Context pack:\n{_compact_context(context_pack)}"
    )


def build_critic_prompt(context_pack: ContextPack, forecast: dict[str, Any]) -> str:
    return (
        "You are the critic pass for a local research analyst. Use only the "
        "provided context pack and forecast JSON. Produce strict JSON listing the "
        "strongest ways the forecast could be wrong.\n\n"
        "Required rules:\n"
        "- focus on disconfirming evidence and missing data\n"
        "- confidence_adjustment must be between -1 and 0 because this is a critic pass\n"
        "- do not introduce facts not present in the context pack\n\n"
        f"Context pack:\n{_compact_context(context_pack)}\n\n"
        f"Forecast JSON:\n{json.dumps(forecast, indent=2, default=str)}"
    )


def run_forecast_pass(
    context_pack: ContextPack,
    *,
    model: str,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    client = client or OllamaClient()
    payload = client.generate_json(
        model=model,
        prompt=build_forecast_prompt(context_pack),
        response_schema=FORECAST_SCHEMA,
    )
    return normalize_forecast_payload(payload)


def run_critic_pass(
    context_pack: ContextPack,
    forecast: dict[str, Any],
    *,
    model: str,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    client = client or OllamaClient()
    payload = client.generate_json(
        model=model,
        prompt=build_critic_prompt(context_pack, forecast),
        response_schema=CRITIC_SCHEMA,
    )
    return normalize_critic_payload(payload)


def merge_forecast_and_critic(
    context_pack: ContextPack,
    forecast: dict[str, Any],
    critic: dict[str, Any],
) -> dict[str, Any]:
    base_confidence = float(forecast.get("confidence", 0.0) or 0.0)
    confidence_adjustment = float(critic.get("confidence_adjustment", 0.0) or 0.0)
    final_confidence = round(
        min(1.0, max(0.0, base_confidence + confidence_adjustment)),
        4,
    )
    abstain_reasons: list[str] = []
    if context_pack.freshness_flags.get("price_is_stale"):
        abstain_reasons.append("price_data_stale")
    if context_pack.freshness_flags.get("theme_is_stale"):
        abstain_reasons.append("theme_context_stale")
    if abstain_reasons:
        forecast = {**forecast, "verdict": "low_conviction"}
        final_confidence = min(final_confidence, 0.35)
    return {
        "ticker": context_pack.ticker,
        "as_of": context_pack.as_of.isoformat(),
        "forecast": forecast,
        "critic": critic,
        "final_confidence": final_confidence,
        "freshness_flags": context_pack.freshness_flags,
        "abstain_reasons": abstain_reasons,
    }


def run_two_pass_forecast(
    context_pack: ContextPack,
    *,
    forecast_model: str = "llama3.2",
    critic_model: str | None = None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    client = client or OllamaClient()
    forecast = run_forecast_pass(
        context_pack,
        model=forecast_model,
        client=client,
    )
    critic = run_critic_pass(
        context_pack,
        forecast,
        model=critic_model or forecast_model,
        client=client,
    )
    return merge_forecast_and_critic(context_pack, forecast, critic)

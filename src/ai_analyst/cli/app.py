from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import typer

from ai_analyst.config import get_settings
from ai_analyst.logging import configure_logging
from ai_analyst.utils.dates import parse_iso_datetime
from ai_analyst.utils.io import ensure_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
collect_app = typer.Typer(no_args_is_help=True)
transform_app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True)
features_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
geo_app = typer.Typer(no_args_is_help=True)
analyst_app = typer.Typer(no_args_is_help=True)
portfolio_app = typer.Typer(no_args_is_help=True)

app.add_typer(collect_app, name="collect")
app.add_typer(transform_app, name="transform")
app.add_typer(db_app, name="db")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(geo_app, name="geo")
app.add_typer(analyst_app, name="analyst")
app.add_typer(portfolio_app, name="portfolio")


def _csv_items(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _optional_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@app.callback()
def callback() -> None:
    configure_logging()


@app.command()
def bootstrap() -> None:
    """Create the local folder structure and refresh an empty DuckDB file."""
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    for path in [
        settings.raw_root,
        settings.warehouse_root,
        settings.reports_path,
        settings.mlruns_path,
        settings.workspace / "docs",
        settings.workspace / "notebooks",
        settings.workspace / "tests",
    ]:
        ensure_dir(path)
    refresh_views(settings)
    typer.echo(f"Bootstrapped workspace at {settings.workspace}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host for the local analyst API."),
    port: int = typer.Option(8181, help="Bind port for the local analyst API."),
) -> None:
    """Serve minimal local analyst HTTP endpoints for future UI consumers."""
    from ai_analyst.api.server import serve as serve_api

    settings = get_settings()
    typer.echo(f"Serving analyst API on http://{host}:{port}")
    serve_api(settings, host=host, port=port)


@app.command()
def snapshot(
    as_of: str = typer.Option(..., help="UTC timestamp, for example 2024-01-31T23:59:59Z"),
    tickers: str = typer.Option("", help="Comma-separated tickers."),
) -> None:
    """Build a point-in-time snapshot bundle and print it as JSON."""
    from ai_analyst.warehouse.snapshot_builder import SnapshotBuilder

    settings = get_settings()
    bundle = SnapshotBuilder(settings).build(
        as_of=parse_iso_datetime(as_of),
        tickers=_csv_items(tickers),
    )
    payload = {
        "as_of": bundle.as_of.isoformat(),
        "macro_rows": len(bundle.macro),
        "price_rows": len(bundle.prices),
        "companyfacts_rows": len(bundle.companyfacts),
        "submissions_rows": len(bundle.submissions),
        "universe_rows": len(bundle.universe),
        "event_rows": len(bundle.events),
        "event_relation_rows": len(bundle.event_relations),
        "event_source_assessment_rows": len(bundle.event_source_assessment),
        "event_narrative_risk_rows": len(bundle.event_narrative_risk),
        "evidence_catalog_rows": len(bundle.evidence_catalog),
        "theme_intensity_rows": len(bundle.theme_intensities),
        "theme_regime_rows": len(bundle.theme_regimes),
        "sector_ranking_rows": len(bundle.sector_rankings),
        "analog_rows": len(bundle.historical_analogs),
        "cross_asset_confirmation_rows": len(bundle.cross_asset_confirmation),
        "pricing_discipline_rows": len(bundle.pricing_discipline),
        "trade_readiness_rows": len(bundle.trade_readiness),
        "causal_state_rows": len(bundle.causal_state),
        "causal_chain_rows": len(bundle.causal_chains),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("context-pack")
def context_pack(
    ticker: str = typer.Option(..., help="Ticker symbol, for example AAPL."),
    as_of: str = typer.Option(..., help="UTC timestamp, for example 2024-01-31T23:59:59Z"),
    mode: str = typer.Option("research", help="Context mode: research or decision."),
    trust_tier: str | None = typer.Option(
        None,
        help="Optional trust tier override: experimental, paper, or trusted.",
    ),
) -> None:
    """Build a ContextPack and print it as JSON."""
    from ai_analyst.llm.context_pack import ContextPackBuilder

    settings = get_settings()
    if trust_tier:
        settings.trust_tier = trust_tier
    pack = ContextPackBuilder(settings).build(
        ticker=ticker.upper(),
        as_of=parse_iso_datetime(as_of),
        mode=mode,
    )
    typer.echo(json.dumps(asdict(pack), indent=2, default=str))


@analyst_app.command("forecast")
def analyst_forecast(
    ticker: str = typer.Option(..., help="Ticker symbol, for example AAPL."),
    as_of: str = typer.Option(..., help="UTC timestamp, for example 2024-01-31T23:59:59Z"),
    model: str | None = typer.Option(
        None,
        help="Forecast model name available in Ollama. Defaults to config/.env.",
    ),
    critic_model: str | None = typer.Option(
        None,
        help="Optional critic model override. Defaults to the forecast model.",
    ),
    ollama_host: str | None = typer.Option(
        None,
        help="Base URL for the local Ollama server. Defaults to config/.env.",
    ),
    timeout: float | None = typer.Option(
        None,
        help="HTTP timeout in seconds for each LLM pass. Defaults to config/.env.",
    ),
    persist: bool = typer.Option(
        True,
        "--persist/--no-persist",
        help="Persist forecast outcomes, override logs, and calibration metrics.",
    ),
    trust_tier: str | None = typer.Option(
        None,
        help="Optional trust tier override: experimental, paper, or trusted.",
    ),
) -> None:
    """Run the v3 decision-mode analyst forecast from the ContextPack."""
    from ai_analyst.llm.context_pack import ContextPackBuilder
    from ai_analyst.llm.forecast import OllamaClient
    from ai_analyst.llm.reasoning import run_decision_mode
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    if trust_tier:
        settings.trust_tier = trust_tier
    resolved_model = model or settings.ollama_forecast_model
    resolved_critic_model = critic_model or settings.ollama_critic_model
    resolved_host = ollama_host or settings.ollama_host
    resolved_timeout = timeout or settings.ollama_timeout_seconds
    pack = ContextPackBuilder(settings).build(
        ticker=ticker.upper(),
        as_of=parse_iso_datetime(as_of),
        mode="decision",
    )
    client = OllamaClient(host=resolved_host, timeout=resolved_timeout)
    output = run_decision_mode(
        pack,
        model=resolved_model,
        critic_model=resolved_critic_model,
        client=client,
    )
    if persist:
        from ai_analyst.calibration.metrics import materialize_calibration_metrics
        from ai_analyst.calibration.persistence import persist_decision_forecast

        outcome_paths, override_paths = persist_decision_forecast(
            settings,
            context_pack=pack,
            decision_output=output,
        )
        refresh_views(settings)
        calibration_paths = materialize_calibration_metrics(settings)
        refresh_views(settings)
        output["persistence"] = {
            "forecast_outcome_paths": [str(path) for path in outcome_paths],
            "override_log_paths": [str(path) for path in override_paths],
            "calibration_metric_paths": [str(path) for path in calibration_paths],
        }
    typer.echo(json.dumps(output, indent=2, default=str))


@analyst_app.command("research")
def analyst_research(
    ticker: str = typer.Option(..., help="Ticker symbol, for example AAPL."),
    as_of: str = typer.Option(..., help="UTC timestamp, for example 2024-01-31T23:59:59Z"),
    model: str | None = typer.Option(
        None,
        help="Research model name available in Ollama. Defaults to config/.env.",
    ),
    ollama_host: str | None = typer.Option(
        None,
        help="Base URL for the local Ollama server. Defaults to config/.env.",
    ),
    timeout: float | None = typer.Option(
        None,
        help="HTTP timeout in seconds for each LLM pass. Defaults to config/.env.",
    ),
    trust_tier: str | None = typer.Option(
        None,
        help="Optional trust tier override: experimental, paper, or trusted.",
    ),
) -> None:
    """Run the v3 research-mode analyst with specialist role prompts."""
    from ai_analyst.llm.context_pack import ContextPackBuilder
    from ai_analyst.llm.forecast import OllamaClient
    from ai_analyst.llm.reasoning import run_research_mode

    settings = get_settings()
    if trust_tier:
        settings.trust_tier = trust_tier
    resolved_model = model or settings.ollama_forecast_model
    resolved_host = ollama_host or settings.ollama_host
    resolved_timeout = timeout or settings.ollama_timeout_seconds
    pack = ContextPackBuilder(settings).build(
        ticker=ticker.upper(),
        as_of=parse_iso_datetime(as_of),
        mode="research",
    )
    client = OllamaClient(host=resolved_host, timeout=resolved_timeout)
    output = run_research_mode(pack, model=resolved_model, client=client)
    typer.echo(json.dumps(output, indent=2, default=str))


@analyst_app.command("ollama-status")
def analyst_ollama_status(
    ollama_host: str | None = typer.Option(
        None,
        help="Base URL for the local Ollama server. Defaults to config/.env.",
    ),
) -> None:
    """Show local Ollama connectivity and installed models."""
    import requests

    settings = get_settings()
    host = ollama_host or settings.ollama_host
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    payload = response.json()
    typer.echo(
        json.dumps(
            {
                "host": host,
                "configured_forecast_model": settings.ollama_forecast_model,
                "configured_critic_model": settings.ollama_critic_model,
                "models": payload.get("models", []),
            },
            indent=2,
            default=str,
        )
    )


@analyst_app.command("trace")
def analyst_trace(
    ticker: str = typer.Option(..., help="Ticker symbol, for example AAPL."),
    as_of: str = typer.Option(..., help="UTC timestamp, for example 2024-01-31T23:59:59Z"),
    trust_tier: str | None = typer.Option(
        None,
        help="Optional trust tier override: experimental, paper, or trusted.",
    ),
) -> None:
    """Show the full causal, evidence, and downgrade trace for one ticker/date."""
    from ai_analyst.llm.context_pack import ContextPackBuilder

    settings = get_settings()
    if trust_tier:
        settings.trust_tier = trust_tier
    pack = ContextPackBuilder(settings).build(
        ticker=ticker.upper(),
        as_of=parse_iso_datetime(as_of),
        mode="decision",
    )
    payload = {
        "lineage": pack.version_metadata,
        "trust_tier": pack.trust_tier,
        "evidence_freshness_summary": {
            key: value.get("freshness_class")
            for key, value in list(pack.evidence_index.items())[:10]
        },
        "causal_trace": pack.causal_chains[:10],
        "analog_trace": pack.analog_matches,
        "pricing_trace": {
            "cross_asset_confirmation": pack.cross_asset_confirmation,
            "pricing_discipline": pack.pricing_discipline,
            "confidence_breakdown": pack.confidence_breakdown,
        },
        "downgrade_gate_outcome": {
            "requested_mode": "decision",
            "trust_tier": pack.trust_tier,
            "missing_evidence": pack.missing_evidence,
            "confidence_breakdown": pack.confidence_breakdown,
        },
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


@db_app.command("calibration")
def refresh_calibration_metrics() -> None:
    """Recompute calibration metrics from persisted forecasts and realized labels."""
    from ai_analyst.calibration.metrics import materialize_calibration_metrics
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    paths = materialize_calibration_metrics(settings)
    refresh_views(settings)
    typer.echo(
        json.dumps(
            {
                "metric_file_count": len(paths),
                "paths": [str(path) for path in paths],
            },
            indent=2,
        )
    )


@collect_app.command("fred-current")
def collect_fred_current(
    series: str = typer.Option("", help="Comma-separated FRED series ids."),
) -> None:
    from ai_analyst.sources.fred import FredClient

    settings = get_settings()
    outputs = FredClient(settings).collect_current_series(
        _csv_items(series) or settings.macro_series
    )
    typer.echo(f"Saved {len(outputs)} FRED current snapshots.")


@collect_app.command("fred-vintages")
def collect_fred_vintages(
    series: str = typer.Option("", help="Comma-separated FRED series ids."),
) -> None:
    from ai_analyst.sources.fred import FredClient

    settings = get_settings()
    outputs = FredClient(settings).collect_vintages(_csv_items(series) or settings.macro_series)
    typer.echo(f"Saved {len(outputs)} ALFRED vintage snapshots.")


@collect_app.command("sec-submissions")
def collect_sec_submissions(cik: str = typer.Option(..., help="Comma-separated CIKs.")) -> None:
    from ai_analyst.sources.sec import SecClient

    settings = get_settings()
    outputs = SecClient(settings).collect_submissions(_csv_items(cik))
    typer.echo(f"Saved {len(outputs)} SEC submissions snapshots.")


@collect_app.command("sec-companyfacts")
def collect_sec_companyfacts(cik: str = typer.Option(..., help="Comma-separated CIKs.")) -> None:
    from ai_analyst.sources.sec import SecClient

    settings = get_settings()
    outputs = SecClient(settings).collect_companyfacts(_csv_items(cik))
    typer.echo(f"Saved {len(outputs)} SEC companyfacts snapshots.")


@collect_app.command("sec-universe")
def collect_sec_universe(
    limit: int | None = typer.Option(
        None,
        help="Optional max number of latest-universe CIKs to collect.",
    ),
) -> None:
    from ai_analyst.sources.sec import SecClient, latest_universe_ciks

    settings = get_settings()
    ciks = latest_universe_ciks(settings, limit=limit)
    if not ciks:
        raise typer.BadParameter(
            "No CIKs found in universe_membership. Collect and transform the universe first."
        )
    client = SecClient(settings)
    submissions = client.collect_submissions(ciks)
    companyfacts = client.collect_companyfacts(ciks)
    typer.echo(
        json.dumps(
            {
                "cik_count": len(ciks),
                "submissions_snapshots": len(submissions),
                "companyfacts_snapshots": len(companyfacts),
            },
            indent=2,
        )
    )


@collect_app.command("sec-v1")
def collect_sec_v1(
    limit: int | None = typer.Option(
        None,
        help="Optional max number of latest V1-universe CIKs to collect.",
    ),
) -> None:
    from ai_analyst.sources.sec import SecClient, latest_v1_universe_ciks

    settings = get_settings()
    ciks = latest_v1_universe_ciks(settings, limit=limit)
    if not ciks:
        raise typer.BadParameter(
            "No CIKs found in v1_universe. Build features or materialize the V1 universe first."
        )
    client = SecClient(settings)
    submissions = client.collect_submissions(ciks)
    companyfacts = client.collect_companyfacts(ciks)
    typer.echo(
        json.dumps(
            {
                "cik_count": len(ciks),
                "submissions_snapshots": len(submissions),
                "companyfacts_snapshots": len(companyfacts),
            },
            indent=2,
        )
    )


@collect_app.command("prices")
def collect_prices(
    tickers: str = typer.Option(..., help="Comma-separated tickers."),
    start_date: str | None = typer.Option(None, help="Optional ISO start date."),
    end_date: str | None = typer.Option(None, help="Optional ISO end date."),
) -> None:
    from ai_analyst.sources.tiingo import TiingoPriceSource

    settings = get_settings()
    source = TiingoPriceSource(settings)
    outputs = source.collect_raw(
        _csv_items(tickers),
        start_date=_optional_date(start_date),
        end_date=_optional_date(end_date),
    )
    typer.echo(f"Saved {len(outputs)} Tiingo price snapshots.")


@collect_app.command("prices-universe")
def collect_prices_universe(
    limit: int | None = typer.Option(
        None,
        help="Optional max number of latest-universe tickers to collect.",
    ),
    start_date: str | None = typer.Option(None, help="Optional ISO start date."),
    end_date: str | None = typer.Option(None, help="Optional ISO end date."),
    include_benchmark: bool = typer.Option(
        True,
        help="Include SPY benchmark history alongside the universe tickers.",
    ),
) -> None:
    from ai_analyst.sources.tiingo import TiingoPriceSource
    from ai_analyst.sources.universe import latest_sp500_tickers

    settings = get_settings()
    tickers = latest_sp500_tickers(
        settings,
        limit=limit,
        include_benchmark=include_benchmark,
    )
    if not tickers:
        raise typer.BadParameter(
            "No tickers found in universe_membership. Collect and transform the universe first."
        )
    source = TiingoPriceSource(settings)
    outputs = source.collect_raw(
        tickers,
        start_date=_optional_date(start_date),
        end_date=_optional_date(end_date),
    )
    typer.echo(
        json.dumps(
            {
                "ticker_count": len(tickers),
                "snapshots_saved": len(outputs),
            },
            indent=2,
        )
    )


@collect_app.command("universe")
def collect_universe() -> None:
    from ai_analyst.sources.universe import collect_sp500_constituents

    settings = get_settings()
    output = collect_sp500_constituents(settings)
    typer.echo(f"Saved universe snapshot to {output}")


@collect_app.command("worldmonitor")
def collect_worldmonitor(
    max_items: int = typer.Option(50, help="Maximum items per endpoint."),
) -> None:
    from ai_analyst.sources.worldmonitor import WorldMonitorClient

    settings = get_settings()
    output = WorldMonitorClient(settings).collect_snapshot(max_items=max_items)
    typer.echo(f"Saved World Monitor context snapshot to {output}")


@collect_app.command("gpr")
def collect_gpr() -> None:
    """Download the Caldara-Iacoviello Geopolitical Risk Index (daily + monthly)."""
    from ai_analyst.sources.gpr import collect_gpr as _collect_gpr

    settings = get_settings()
    daily_path, monthly_path = _collect_gpr(settings)
    typer.echo(f"Saved GPR daily to {daily_path}")
    typer.echo(f"Saved GPR monthly to {monthly_path}")


@transform_app.command("macro")
def transform_macro() -> None:
    from ai_analyst.sources.fred import transform_current, transform_vintages

    settings = get_settings()
    current_outputs = transform_current(settings)
    vintage_outputs = transform_vintages(settings)
    typer.echo(
        "Materialized "
        f"{len(current_outputs)} current macro files and "
        f"{len(vintage_outputs)} vintage files."
    )


@transform_app.command("sec")
def transform_sec() -> None:
    from ai_analyst.sources.sec import transform_companyfacts, transform_submissions

    settings = get_settings()
    submissions_outputs, filing_index_outputs = transform_submissions(settings)
    companyfacts_outputs = transform_companyfacts(settings)
    typer.echo(
        "Materialized "
        f"{len(submissions_outputs)} submissions files, "
        f"{len(filing_index_outputs)} filing index files, and "
        f"{len(companyfacts_outputs)} companyfacts files."
    )


@transform_app.command("prices")
def transform_price_data() -> None:
    from ai_analyst.sources.tiingo import transform_prices

    settings = get_settings()
    price_outputs, action_outputs = transform_prices(settings)
    typer.echo(
        f"Materialized {len(price_outputs)} price files and {len(action_outputs)} action files."
    )


@transform_app.command("universe")
def transform_universe() -> None:
    from ai_analyst.sources.universe import transform_sp500_constituents

    settings = get_settings()
    outputs = transform_sp500_constituents(settings)
    typer.echo(f"Materialized {len(outputs)} universe files.")


@transform_app.command("worldmonitor")
def transform_worldmonitor_context() -> None:
    from ai_analyst.sources.worldmonitor import transform_worldmonitor

    settings = get_settings()
    (
        normalized_outputs,
        entity_outputs,
        relation_outputs,
        source_assessment_outputs,
        narrative_risk_outputs,
        evidence_catalog_outputs,
    ) = transform_worldmonitor(settings)
    typer.echo(
        "Materialized "
        f"{len(normalized_outputs)} normalized event files, "
        f"{len(entity_outputs)} event-entity files, and "
        f"{len(relation_outputs)} event-relation files, "
        f"{len(source_assessment_outputs)} source-assessment files, "
        f"{len(narrative_risk_outputs)} narrative-risk files, and "
        f"{len(evidence_catalog_outputs)} evidence-catalog files."
    )


@transform_app.command("all")
def transform_all(
    skip_worldmonitor: bool = typer.Option(
        False,
        "--skip-worldmonitor",
        help="Skip World Monitor event transforms if local World Monitor data is unavailable.",
    ),
) -> None:
    from ai_analyst.sources.fred import transform_current, transform_vintages
    from ai_analyst.sources.sec import transform_companyfacts, transform_submissions
    from ai_analyst.sources.tiingo import transform_prices
    from ai_analyst.sources.universe import transform_sp500_constituents
    from ai_analyst.sources.worldmonitor import transform_worldmonitor

    settings = get_settings()
    transform_current(settings)
    transform_vintages(settings)
    transform_submissions(settings)
    transform_companyfacts(settings)
    transform_prices(settings)
    transform_sp500_constituents(settings)
    if not skip_worldmonitor:
        transform_worldmonitor(settings)
    typer.echo(
        "Finished raw-to-parquet transforms for macro, SEC, prices, "
        f"universe, and {'World Monitor' if not skip_worldmonitor else 'no World Monitor data'}."
    )


@db_app.command("refresh")
def refresh_database() -> None:
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    refresh_views(settings)
    typer.echo(f"Refreshed DuckDB views at {settings.duckdb_file}")


@db_app.command("freshness")
def check_data_freshness() -> None:
    """Check data freshness across all warehouse domains."""
    from ai_analyst.monitoring import check_freshness, freshness_summary

    settings = get_settings()
    results = check_freshness(settings)
    typer.echo(json.dumps(freshness_summary(results), indent=2))


@features_app.command("build")
def build_features() -> None:
    from ai_analyst.features.engineering import (
        materialize_features_and_labels,
        materialize_v1_universe,
    )
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    materialize_v1_universe(settings)
    refresh_views(settings)
    feature_paths, label_paths = materialize_features_and_labels(settings)
    refresh_views(settings)
    typer.echo(
        "Materialized "
        f"{len(feature_paths)} feature partitions and "
        f"{len(label_paths)} label partitions."
    )


@train_app.command("baseline")
def train_v1_baseline() -> None:
    from ai_analyst.modeling.train import train_baseline

    settings = get_settings()
    artifacts = train_baseline(settings)
    typer.echo(
        json.dumps(
            {
                "metric_rows": len(artifacts.metrics),
                "prediction_rows": len(artifacts.predictions),
                "report_path": str(artifacts.report_path) if artifacts.report_path else None,
                "ablation_paths": [str(path) for path in (artifacts.ablation_paths or [])],
            },
            indent=2,
        )
    )


@portfolio_app.command("rebalance")
def portfolio_rebalance(
    report_path: str | None = typer.Option(
        None,
        help="Optional explicit nightly report path. Defaults to the latest report in reports/.",
    ),
) -> None:
    """Build a constrained heuristic rebalance plan from the latest ranked report."""
    from ai_analyst.portfolio.allocator import build_rebalance_plan

    settings = get_settings()
    payload = build_rebalance_plan(
        settings,
        report_path=Path(report_path) if report_path else None,
    )
    typer.echo(json.dumps(payload, indent=2, default=str))


@geo_app.command("seed-defaults")
def geo_seed_defaults() -> None:
    from ai_analyst.events.exposures import seed_geo_energy_reference_data
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    exposure_paths, solution_paths, entity_paths = seed_geo_energy_reference_data(settings)
    refresh_views(settings)
    typer.echo(
        "Seeded "
        f"{len(exposure_paths)} exposure files and "
        f"{len(solution_paths)} solution mapping files plus "
        f"{len(entity_paths)} entity and dependency files."
    )


@geo_app.command("build-context")
def geo_build_context(
    as_of: str | None = typer.Option(None, help="Optional ISO date, for example 2024-01-31."),
) -> None:
    from ai_analyst.causal.analog_scoring import materialize_historical_analogs
    from ai_analyst.causal.causal_graph import build_and_materialize_causal_state
    from ai_analyst.causal.regime_engine import materialize_theme_regimes
    from ai_analyst.events.sector_opportunity import materialize_sector_rankings
    from ai_analyst.events.theme_intensity import materialize_theme_intensity_tables
    from ai_analyst.warehouse.database import refresh_views

    settings = get_settings()
    resolved_date = date.fromisoformat(as_of) if as_of else None
    hourly_paths, daily_paths = materialize_theme_intensity_tables(settings)
    refresh_views(settings)
    regime_paths = materialize_theme_regimes(settings)
    refresh_views(settings)
    ranking_paths = materialize_sector_rankings(
        settings,
        as_of=resolved_date,
    )
    refresh_views(settings)
    analog_artifacts = materialize_historical_analogs(
        settings,
        as_of_date=resolved_date,
    )
    refresh_views(settings)
    resolved_as_of_dt = (
        parse_iso_datetime(f"{resolved_date.isoformat()}T20:00:00Z")
        if resolved_date
        else parse_iso_datetime(f"{date.today().isoformat()}T20:00:00Z")
    )
    (
        _,
        _,
        causal_state_paths,
        causal_chain_paths,
        cross_asset_paths,
        pricing_paths,
        readiness_paths,
    ) = build_and_materialize_causal_state(
        settings,
        as_of=resolved_as_of_dt,
    )
    refresh_views(settings)
    typer.echo(
        "Materialized "
        f"{len(hourly_paths)} hourly theme files, "
        f"{len(daily_paths)} daily theme files, and "
        f"{len(regime_paths)} regime files, "
        f"{len(ranking_paths)} sector ranking files, and "
        f"{len(analog_artifacts.paths)} analog files plus "
        f"{len(causal_state_paths)} causal-state files and "
        f"{len(causal_chain_paths)} causal-chain files, "
        f"{len(cross_asset_paths)} cross-asset files, "
        f"{len(pricing_paths)} pricing-discipline files, and "
        f"{len(readiness_paths)} trade-readiness files."
    )


@geo_app.command("validate-graph")
def geo_validate_graph() -> None:
    """Validate the YAML-backed causal graph assets."""
    from ai_analyst.causal.causal_graph import CausalGraphEngine
    from ai_analyst.causal.governance import (
        load_evidence_freshness,
        load_narrative_rules,
        load_source_profiles,
        load_trust_tiers,
    )

    engine = CausalGraphEngine()
    source_profiles = load_source_profiles()
    narrative_rules = load_narrative_rules()
    trust_tiers = load_trust_tiers()
    evidence_freshness = load_evidence_freshness()
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "transmission_edges": len(engine.transmission.get("edges", [])),
                "sector_mediator_edges": len(engine.sector_mediator.get("edges", [])),
                "stock_exposure_edges": len(engine.stock_exposure.get("edges", [])),
                "source_profiles": len(source_profiles.get("profiles", {})),
                "narrative_rules": len(narrative_rules.get("rules", {})),
                "trust_tiers": len(trust_tiers.get("tiers", {})),
                "freshness_classes": len(evidence_freshness.get("classes", {})),
            },
            indent=2,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

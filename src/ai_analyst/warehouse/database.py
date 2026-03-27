from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ai_analyst.config import Settings
from ai_analyst.utils.io import ensure_dir
from ai_analyst.warehouse.schema import CANONICAL_SCHEMAS, empty_relation_sql

logger = logging.getLogger(__name__)


RELATION_DOMAIN_MAP: dict[str, str] = {
    "macro_observations": "macro/current",
    "macro_vintages": "macro/vintages",
    "edgar_submissions": "edgar/submissions",
    "edgar_companyfacts": "edgar/companyfacts",
    "filing_index": "edgar/filing_index",
    "prices": "prices/daily",
    "price_actions": "prices/actions",
    "universe_membership": "universe/sp500_current",
    "v1_universe": "universe/v1_top150",
    "feature_matrix": "features/daily",
    "label_matrix": "labels/daily",
    "normalized_events": "events/normalized",
    "event_entities": "events/entities",
    "event_relations": "events/relations",
    "event_source_assessment": "events/source_assessment",
    "event_narrative_risk": "events/narrative_risk",
    "evidence_catalog": "events/evidence_catalog",
    "dependency_markers": "entities/dependencies",
    "theme_intensity_hourly": "themes/hourly",
    "theme_intensity_daily": "themes/daily",
    "sector_theme_exposure": "exposures/sector",
    "industry_theme_exposure": "exposures/industry",
    "stock_theme_exposure": "exposures/stock",
    "sector_rankings": "rankings/sector",
    "theme_regimes": "regimes/themes",
    "historical_analogs": "analogs/historical",
    "cross_asset_confirmation_daily": "trust/cross_asset",
    "pricing_discipline_daily": "trust/pricing_discipline",
    "trade_readiness_daily": "trust/trade_readiness",
    "causal_state_daily": "causal/state_daily",
    "causal_chain_activations": "causal/chains",
    "solution_mappings": "solutions/default",
    "entity_company": "entities/company",
    "entity_country": "entities/country",
    "entity_region": "entities/region",
    "entity_commodity": "entities/commodity",
    "entity_sector": "entities/sector",
    "entity_route": "entities/route",
    "entity_event_type": "entities/event_type",
    "entity_policy_action": "entities/policy_action",
    "forecast_outcomes": "forecast/outcomes",
    "forecast_calibration_metrics": "forecast/calibration_metrics",
    "llm_override_log": "forecast/override_log",
    "feature_family_ablation_metrics": "forecast/feature_family_ablation",
}


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    ensure_dir(settings.duckdb_file.parent)
    conn = duckdb.connect(str(settings.duckdb_file))
    conn.execute("PRAGMA threads=4")
    return conn


def _valid_parquet_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    valid: list[Path] = []
    for path in sorted(root.rglob("*.parquet")):
        try:
            pq.ParquetFile(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping unreadable Parquet file %s: %s", path, exc)
            continue
        valid.append(path)
    return valid


def _read_parquet_sql(files: list[Path]) -> str:
    rendered = ", ".join(repr(str(path)) for path in files)
    return f"SELECT * FROM read_parquet([{rendered}], union_by_name=true)"


def _drop_relation_if_exists(
    conn: duckdb.DuckDBPyConnection,
    relation_name: str,
) -> None:
    row = conn.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name = ?
        """,
        [relation_name],
    ).fetchone()
    if not row:
        return
    table_type = str(row[0]).upper()
    if table_type == "VIEW":
        conn.execute(f"DROP VIEW {relation_name}")
    else:
        conn.execute(f"DROP TABLE {relation_name}")


def refresh_views(settings: Settings) -> None:
    conn = connect(settings)
    try:
        for relation_name, schema in CANONICAL_SCHEMAS.items():
            domain = RELATION_DOMAIN_MAP.get(relation_name)
            files = _valid_parquet_files(settings.warehouse_root / domain) if domain else []
            query = _read_parquet_sql(files) if files else empty_relation_sql(schema)
            _drop_relation_if_exists(conn, relation_name)
            conn.execute(f"CREATE VIEW {relation_name} AS {query}")
    finally:
        conn.close()

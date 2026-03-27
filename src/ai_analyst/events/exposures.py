from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ai_analyst.causal.entity_normalization import default_entity_reference_frames
from ai_analyst.config import Settings
from ai_analyst.events.ontology import (
    default_industry_theme_exposures,
    default_sector_theme_exposures,
    default_solution_mappings,
    default_stock_theme_exposures,
)
from ai_analyst.events.theme_intensity import theme_intensity_wide
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path


def seed_geo_energy_reference_data(
    settings: Settings,
) -> tuple[list[Path], list[Path], list[Path]]:
    sector_exposures = default_sector_theme_exposures()
    industry_exposures = default_industry_theme_exposures()
    stock_exposures = default_stock_theme_exposures()
    solution_mappings = default_solution_mappings()
    entity_frames = default_entity_reference_frames()
    as_of = datetime.now(tz=UTC).date()

    sector_path = warehouse_partition_path(
        settings,
        domain="exposures/sector",
        partition_date=as_of,
        stem=f"sector_theme_exposure_{as_of.isoformat()}",
    )
    industry_path = warehouse_partition_path(
        settings,
        domain="exposures/industry",
        partition_date=as_of,
        stem=f"industry_theme_exposure_{as_of.isoformat()}",
    )
    stock_path = warehouse_partition_path(
        settings,
        domain="exposures/stock",
        partition_date=as_of,
        stem=f"stock_theme_exposure_{as_of.isoformat()}",
    )
    solution_path = warehouse_partition_path(
        settings,
        domain="solutions/default",
        partition_date=as_of,
        stem=f"solution_mappings_{as_of.isoformat()}",
    )
    write_parquet(sector_exposures, sector_path)
    write_parquet(industry_exposures, industry_path)
    write_parquet(stock_exposures, stock_path)
    write_parquet(solution_mappings, solution_path)

    entity_paths: list[Path] = []
    domain_map = {
        "entity_company": "entities/company",
        "entity_country": "entities/country",
        "entity_region": "entities/region",
        "entity_commodity": "entities/commodity",
        "entity_sector": "entities/sector",
        "entity_route": "entities/route",
        "entity_event_type": "entities/event_type",
        "entity_policy_action": "entities/policy_action",
        "dependency_markers": "entities/dependencies",
    }
    for relation_name, frame in entity_frames.items():
        out_path = warehouse_partition_path(
            settings,
            domain=domain_map[relation_name],
            partition_date=as_of,
            stem=f"{relation_name}_{as_of.isoformat()}",
        )
        write_parquet(frame, out_path)
        entity_paths.append(out_path)

    return [sector_path, industry_path, stock_path], [solution_path], entity_paths


def compute_sector_context_shocks(
    theme_daily: pd.DataFrame,
    sector_exposures: pd.DataFrame,
) -> pd.DataFrame:
    if theme_daily.empty or sector_exposures.empty:
        return pd.DataFrame()

    exposure_frame = sector_exposures.copy()
    exposure_frame["exposure"] = pd.to_numeric(
        exposure_frame["exposure"],
        errors="coerce",
    ).fillna(0.0)
    wide = theme_intensity_wide(theme_daily)
    if wide.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for record in wide.to_dict(orient="records"):
        as_of_date = pd.to_datetime(record["date"]).date()
        theme_values = {key: float(value or 0.0) for key, value in record.items() if key != "date"}
        for sector, group in exposure_frame.groupby("sector", sort=True):
            contributions = {
                row["theme"]: float(row["exposure"]) * theme_values.get(str(row["theme"]), 0.0)
                for _, row in group.iterrows()
            }
            ranked_themes = sorted(
                contributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            supporting_themes = [theme for theme, value in ranked_themes if abs(value) > 0][:3]
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "sector": sector,
                    "context_shock": float(sum(contributions.values())),
                    "top_theme": ranked_themes[0][0] if ranked_themes else None,
                    "supporting_themes": supporting_themes,
                }
            )
    return pd.DataFrame(rows)


def compute_stock_context_shocks(
    *,
    universe: pd.DataFrame,
    sector_shocks: pd.DataFrame,
    stock_exposures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if universe.empty or sector_shocks.empty:
        return pd.DataFrame()

    frame = (
        universe[["ticker", "sector"]]
        .drop_duplicates()
        .merge(
            sector_shocks[["as_of_date", "sector", "context_shock"]],
            on="sector",
            how="left",
        )
    )
    frame["stock_context_shock"] = frame["context_shock"].fillna(0.0)

    if stock_exposures is not None and not stock_exposures.empty:
        stock_adj = (
            stock_exposures.groupby("ticker", as_index=False)["exposure"]
            .sum()
            .rename(columns={"exposure": "stock_exposure_adjustment"})
        )
        frame = frame.merge(stock_adj, on="ticker", how="left")
        frame["stock_context_shock"] = frame["stock_context_shock"] + frame[
            "stock_exposure_adjustment"
        ].fillna(0.0)
        frame = frame.drop(columns=["stock_exposure_adjustment"])

    return frame


def load_reference_frames(
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = connect(settings)
    try:
        sector_exposures = conn.execute("SELECT * FROM sector_theme_exposure").df()
        stock_exposures = conn.execute("SELECT * FROM stock_theme_exposure").df()
        solution_mappings = conn.execute("SELECT * FROM solution_mappings").df()
    finally:
        conn.close()
    return sector_exposures, stock_exposures, solution_mappings

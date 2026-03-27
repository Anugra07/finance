from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path


def build_theme_regimes(theme_daily: pd.DataFrame) -> pd.DataFrame:
    if theme_daily.empty:
        return pd.DataFrame()

    frame = theme_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, object]] = []
    for trade_date, group in frame.groupby(frame["date"].dt.date):
        intensity_total = float(group["intensity"].sum() or 0.0)
        top_theme = group.sort_values("intensity", ascending=False).iloc[0]
        regime_name = "calm"
        regime_score = 0.25
        top_theme_name = str(top_theme["theme"])
        if intensity_total >= 2.5:
            regime_name = "escalating"
            regime_score = min(1.0, intensity_total / 6.0)
        elif top_theme_name in {"oil_supply_risk", "gas_supply_risk", "shipping_stress"}:
            regime_name = "energy_supply_stress"
            regime_score = min(1.0, float(top_theme["intensity"] or 0.0) / 4.0)
        elif top_theme_name in {"defense_escalation", "sanctions_pressure"}:
            regime_name = "geo_elevated"
            regime_score = min(1.0, float(top_theme["intensity"] or 0.0) / 4.0)
        rows.append(
            {
                "as_of_date": trade_date,
                "theme": top_theme_name,
                "regime_name": regime_name,
                "regime_score": round(regime_score, 4),
                "source": "causal_regime_engine_v1",
                "transform_loaded_at": datetime.now(tz=UTC),
            }
        )
    return pd.DataFrame(rows)


def materialize_theme_regimes(settings: Settings) -> list[Path]:
    conn = connect(settings)
    try:
        theme_daily = conn.execute("SELECT * FROM theme_intensity_daily").df()
    finally:
        conn.close()

    frame = build_theme_regimes(theme_daily)
    paths: list[Path] = []
    if frame.empty:
        return paths
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    for trade_date, group in frame.groupby(frame["as_of_date"].dt.date):
        out_path = warehouse_partition_path(
            settings,
            domain="regimes/themes",
            partition_date=trade_date,
            stem=f"theme_regimes_{trade_date.isoformat()}",
        )
        write_parquet(group, out_path)
        paths.append(out_path)
    return paths

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path

if TYPE_CHECKING:
    from ai_analyst.regime.changepoint import ChangepointDetector
    from ai_analyst.regime.hmm import HMMRegimeDetector


def build_theme_regimes(
    theme_daily: pd.DataFrame,
    *,
    hmm_detector: HMMRegimeDetector | None = None,
    changepoint_detector: ChangepointDetector | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build theme regime labels. Backward-compatible: no extra args = threshold logic only.

    When hmm_detector and prices are provided, HMM-based regime labels and probabilities
    are added alongside the threshold-based columns.
    """
    if theme_daily.empty:
        return pd.DataFrame()

    frame = theme_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    # Optional HMM regime data keyed by date
    hmm_by_date: dict[object, dict[str, object]] = {}
    if hmm_detector is not None and prices is not None and not prices.empty:
        try:
            result = hmm_detector.predict(prices["ret_1d"], prices["realized_vol_20d"])
            for idx, state in result.states.items():
                d = idx.date() if hasattr(idx, "date") and callable(idx.date) else idx
                label = result.state_labels.get(int(state), "unknown")
                probs = {col: float(result.state_probabilities.loc[idx, col]) for col in result.state_probabilities.columns}
                hmm_by_date[d] = {
                    "hmm_state": int(state),
                    "hmm_label": label,
                    "hmm_bull_prob": probs.get("bull", 0.0),
                    "hmm_bear_prob": probs.get("bear", 0.0),
                    "hmm_neutral_prob": probs.get("neutral", 0.0),
                }
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("HMM prediction failed: %s", exc)

    # Optional changepoint dates
    changepoint_dates: set[object] = set()
    if changepoint_detector is not None and prices is not None and not prices.empty:
        try:
            cp_dates = changepoint_detector.detect(prices["ret_1d"])
            changepoint_dates = set(cp_dates)
        except Exception:
            pass

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

        row: dict[str, object] = {
            "as_of_date": trade_date,
            "theme": top_theme_name,
            "regime_name": regime_name,
            "regime_score": round(regime_score, 4),
            "source": "causal_regime_engine_v1",
            "transform_loaded_at": datetime.now(tz=UTC),
        }

        # Augment with HMM data if available
        hmm = hmm_by_date.get(trade_date)
        if hmm:
            row["hmm_state"] = hmm["hmm_state"]
            row["hmm_label"] = hmm["hmm_label"]
            row["hmm_bull_prob"] = hmm["hmm_bull_prob"]
            row["hmm_bear_prob"] = hmm["hmm_bear_prob"]
            row["hmm_neutral_prob"] = hmm["hmm_neutral_prob"]

            # Override regime_name if HMM says bear and changepoint detected
            if hmm["hmm_label"] == "bear" and trade_date in changepoint_dates:
                row["regime_name"] = "crisis"
                row["regime_score"] = 1.0
                row["source"] = "causal_regime_engine_v2_hmm"
            elif hmm["hmm_label"] != "neutral":
                row["source"] = "causal_regime_engine_v2_hmm"

        row["changepoint_flag"] = trade_date in changepoint_dates

        rows.append(row)
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

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ai_analyst.causal.interpretation import FEATURE_FAMILY_MAP
from ai_analyst.config import Settings
from ai_analyst.modeling.walkforward import WalkForwardSpec, generate_walk_forward_splits
from ai_analyst.reporting.io import persist_ranked_report
from ai_analyst.reporting.nightly import build_ranked_report
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path

logger = logging.getLogger(__name__)


try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError:  # pragma: no cover
    LGBMRegressor = None
    early_stopping = None
    log_evaluation = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover
    HistGradientBoostingRegressor = None

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


@dataclass(slots=True)
class TrainingArtifacts:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    report_path: Path | None
    feature_importance: pd.DataFrame | None = None
    ablation_paths: list[Path] | None = None


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "overnight_gap",
    "range_norm",
    "realized_vol_20d",
    "volume_surprise_20d",
    "share_turnover",
    "ret_1d_sector_pct",
    "ret_5d_sector_pct",
    "ret_20d_sector_pct",
    "ret_60d_sector_pct",
    "overnight_gap_sector_pct",
    "range_norm_sector_pct",
    "realized_vol_20d_sector_pct",
    "volume_surprise_20d_sector_pct",
    "share_turnover_sector_pct",
    "oil_supply_risk_1d",
    "gas_supply_risk_1d",
    "shipping_stress_1d",
    "sanctions_pressure_1d",
    "defense_escalation_1d",
    "grid_stress_1d",
    "cyber_infra_risk_1d",
    "policy_relief_prob_1d",
    "commodity_shock_score",
    "geo_novelty_score",
    "regional_concentration_risk",
    "sector_context_shock",
    "stock_context_shock",
    "analog_match_score",
    "analog_failure_risk",
    "regime_score",
    "regime_is_escalating",
    "theme_convergence_score",
    "pricing_geo_signal_strength",
    "pricing_mediator_confirmation",
    "pricing_market_response_strength",
    "pricing_divergence_score",
    "pricing_crowdedness_proxy",
    "pricing_follow_through_status",
    "causal_chain_count",
    "causal_chain_avg_confidence",
    "causal_net_sign",
    "source_reliability_score",
    "source_claim_verifiability",
    "source_freshness_score",
    "narrative_deception_risk",
    "narrative_actionability_score",
    "narrative_novelty_score",
    "cross_asset_energy_confirmation",
    "cross_asset_vol_confirmation",
    "cross_asset_credit_confirmation",
    "cross_asset_rates_confirmation",
    "cross_asset_fx_confirmation",
    "cross_asset_sector_flow_confirmation",
    "cross_asset_peer_confirmation",
    "cross_asset_aggregate_confirmation",
    "pricing_discipline_market_confirmation",
    "pricing_discipline_move_lateness",
    "pricing_discipline_overextension",
    "pricing_discipline_reaction_follow_through",
    "trade_readiness_thesis_validity",
    "trade_readiness_pricing_alignment",
    "trade_readiness_timing_quality",
    "trade_readiness_liquidity_quality",
    "trade_readiness_risk_reward_quality",
    "event_dispersion_score",
]


def load_training_frame(settings: Settings) -> pd.DataFrame:
    conn = connect(settings)
    try:
        df = conn.execute(
            """
            SELECT
                f.*,
                l.return_5d,
                l.benchmark_return_5d,
                l.excess_alpha_5d,
                l.excess_alpha_rank
            FROM feature_matrix f
            INNER JOIN label_matrix l USING (date, ticker)
            ORDER BY date, ticker
            """
        ).df()
    finally:
        conn.close()

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["excess_alpha_rank"])
    return df


def _make_model():
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
        )
    if HistGradientBoostingRegressor is not None:
        logger.warning("LightGBM is unavailable; using HistGradientBoostingRegressor fallback.")
        return HistGradientBoostingRegressor(random_state=42, max_depth=6)
    raise RuntimeError("No compatible regressor is installed. Install the [v1] extras.")


def _family_for_feature(name: str) -> str:
    for prefix, family in FEATURE_FAMILY_MAP.items():
        if str(name).startswith(prefix):
            return family
    return "technical"


def _persist_feature_family_ablation(
    settings: Settings,
    *,
    as_of: pd.Timestamp,
    feature_importance: pd.DataFrame | None,
) -> list[Path]:
    if feature_importance is None or feature_importance.empty:
        return []
    stacked = feature_importance.copy()
    if "split_no" not in stacked.columns:
        stacked = stacked.reset_index()
    value_columns = [column for column in stacked.columns if column != "split_no"]
    if not value_columns:
        return []
    long = stacked.melt(
        id_vars=["split_no"],
        value_vars=value_columns,
        var_name="feature",
        value_name="importance",
    )
    long["feature_family"] = long["feature"].map(_family_for_feature)
    summary = (
        long.groupby("feature_family", as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "metric_value"})
    )
    summary["as_of_date"] = pd.to_datetime(as_of).date()
    summary["metric_name"] = "mean_feature_importance"
    summary["source"] = "train_baseline"
    summary["transform_loaded_at"] = datetime.now(tz=UTC)
    out_path = warehouse_partition_path(
        settings,
        domain="forecast/feature_family_ablation",
        partition_date=pd.to_datetime(as_of).date(),
        stem=f"feature_family_ablation_{pd.to_datetime(as_of).date().isoformat()}",
    )
    write_parquet(summary, out_path)
    return [out_path]


def _rank_ic(frame: pd.DataFrame) -> float:
    if frame["prediction"].nunique() <= 1 or frame["excess_alpha_rank"].nunique() <= 1:
        return 0.0
    corr = frame["prediction"].corr(frame["excess_alpha_rank"], method="spearman")
    return float(corr) if corr == corr else 0.0


def _fit_model(
    model: object,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["excess_alpha_rank"]
    if LGBMRegressor is not None and isinstance(model, LGBMRegressor):
        fit_kwargs: dict[str, object] = {}
        if not val_df.empty and early_stopping is not None and log_evaluation is not None:
            fit_kwargs["eval_set"] = [(val_df[FEATURE_COLUMNS], val_df["excess_alpha_rank"])]
            fit_kwargs["eval_metric"] = "l2"
            fit_kwargs["callbacks"] = [early_stopping(50, verbose=False), log_evaluation(0)]
        model.fit(x_train, y_train, **fit_kwargs)
        return
    model.fit(x_train, y_train)


def train_baseline(settings: Settings, *, spec: WalkForwardSpec | None = None) -> TrainingArtifacts:
    spec = spec or WalkForwardSpec(label_horizon_days=settings.label_horizon_days)
    df = load_training_frame(settings)
    if df.empty:
        raise ValueError("No feature/label data available. Run feature materialization first.")

    splits = generate_walk_forward_splits(df, spec)
    if not splits:
        raise ValueError("Not enough history for the requested walk-forward specification.")

    if mlflow is not None:
        mlflow.set_tracking_uri(settings.mlruns_path.as_uri())
        mlflow.set_experiment("local-ai-analyst-v1")

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[dict[str, object]] = []

    for split_no, split in enumerate(splits, start=1):
        train_df = df.loc[split.train_mask].dropna(subset=FEATURE_COLUMNS)
        val_df = df.loc[split.validation_mask].dropna(subset=FEATURE_COLUMNS)
        test_df = df.loc[split.test_mask].dropna(subset=FEATURE_COLUMNS)
        if train_df.empty or test_df.empty:
            continue

        model = _make_model()

        run_ctx = (
            mlflow.start_run(run_name=f"walkforward_split_{split_no}")
            if mlflow is not None
            else None
        )
        if mlflow is not None:
            mlflow.log_params(
                {
                    "split_no": split_no,
                    "train_start": str(split.train_start.date()),
                    "train_end": str(split.train_end.date()),
                    "validation_end": str(split.validation_end.date()),
                    "test_end": str(split.test_end.date()),
                    "model_type": model.__class__.__name__,
                }
            )

        _fit_model(model, train_df=train_df, val_df=val_df)
        test_predictions = test_df[
            ["date", "ticker", "sector", "excess_alpha_rank", "excess_alpha_5d"]
        ].copy()
        test_predictions["prediction"] = model.predict(test_df[FEATURE_COLUMNS])
        test_predictions["split_no"] = split_no
        test_predictions["prob_outperform"] = test_predictions["prediction"].clip(0.0, 1.0)
        test_predictions["confidence_score"] = (
            (test_predictions["prediction"] - 0.5).abs() * 2.0
        ).clip(0.0, 1.0)

        # Compute SHAP feature importance
        if shap is not None and LGBMRegressor is not None and isinstance(model, LGBMRegressor):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(test_df[FEATURE_COLUMNS])
                mean_abs_shap = pd.Series(
                    abs(shap_values).mean(axis=0),
                    index=FEATURE_COLUMNS,
                    name="mean_abs_shap",
                )
                importance_row = mean_abs_shap.to_dict()
                importance_row["split_no"] = split_no
                importance_frames.append(importance_row)
                logger.info(
                    "Split %d: SHAP top-3 features: %s",
                    split_no,
                    mean_abs_shap.nlargest(3).index.tolist(),
                )
            except Exception as exc:
                logger.warning("SHAP computation failed for split %d: %s", split_no, exc)

        prediction_frames.append(test_predictions)

        daily_rank_ics = [_rank_ic(group) for _, group in test_predictions.groupby("date")]
        rank_ic = float(sum(daily_rank_ics) / len(daily_rank_ics)) if daily_rank_ics else 0.0
        hit_rate = float((test_predictions["excess_alpha_5d"] > 0).mean())

        metrics = {
            "split_no": split_no,
            "rank_ic": float(rank_ic),
            "hit_rate": hit_rate,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        }
        if LGBMRegressor is not None and isinstance(model, LGBMRegressor):
            best_iteration = model.best_iteration_
            if best_iteration:
                metrics["best_iteration"] = int(best_iteration)
        metrics_rows.append(metrics)
        if mlflow is not None:
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.end_run()
        elif run_ctx is not None:  # pragma: no cover
            run_ctx.end()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    report_path = None
    ablation_paths: list[Path] = []
    if not predictions.empty:
        latest_date = pd.to_datetime(predictions["date"]).max()
        nightly_report = build_ranked_report(predictions, as_of=latest_date, settings=settings)
        report_path = (
            settings.reports_path / f"nightly_ranked_report_{latest_date.date().isoformat()}.json"
        )
        persist_ranked_report(settings, report=nightly_report, dated_path=report_path)

    feature_importance = None
    if importance_frames:
        feature_importance = pd.DataFrame(importance_frames)
        feature_importance = feature_importance.set_index("split_no")
        if not predictions.empty:
            latest_date = pd.to_datetime(predictions["date"]).max()
            ablation_paths = _persist_feature_family_ablation(
                settings,
                as_of=latest_date,
                feature_importance=feature_importance.reset_index(),
            )

    return TrainingArtifacts(
        metrics=metrics_df,
        predictions=predictions,
        report_path=report_path,
        feature_importance=feature_importance,
        ablation_paths=ablation_paths,
    )

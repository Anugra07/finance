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
    prediction_paths: list[Path] | None = None
    benchmark_paths: list[Path] | None = None


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


def _resolve_market_scope(settings: Settings, market_scope: str | None = None) -> str:
    return str(market_scope or settings.primary_market_scope or "US").upper()


def _label_columns_for_horizon(horizon_days: int) -> dict[str, str]:
    if horizon_days == 21:
        return {
            "return": "return_21d",
            "benchmark_return": "benchmark_return_21d",
            "excess_alpha": "excess_alpha_21d",
            "rank": "excess_alpha_rank_21d",
        }
    return {
        "return": "return_5d",
        "benchmark_return": "benchmark_return_5d",
        "excess_alpha": "excess_alpha_5d",
        "rank": "excess_alpha_rank",
    }


def load_training_frame(
    settings: Settings,
    *,
    horizon_days: int = 5,
    market_scope: str | None = None,
) -> pd.DataFrame:
    label_columns = _label_columns_for_horizon(horizon_days)
    resolved_scope = _resolve_market_scope(settings, market_scope)
    conn = connect(settings)
    try:
        df = conn.execute(
            f"""
            SELECT
                f.*,
                l.{label_columns["return"]} AS target_return,
                l.{label_columns["benchmark_return"]} AS target_benchmark_return,
                l.{label_columns["excess_alpha"]} AS target_excess_alpha,
                l.{label_columns["rank"]} AS target_rank
            FROM feature_matrix f
            INNER JOIN label_matrix l
                ON f.date = l.date
               AND f.ticker = l.ticker
               AND COALESCE(f.market_code, 'US') = COALESCE(l.market_code, 'US')
            WHERE COALESCE(f.market_code, 'US') = ?
            ORDER BY date, ticker
            """,
            [resolved_scope],
        ).df()
    finally:
        conn.close()

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["target_rank"])
    return df


def load_feature_frame(
    settings: Settings,
    *,
    market_scope: str | None = None,
) -> pd.DataFrame:
    resolved_scope = _resolve_market_scope(settings, market_scope)
    conn = connect(settings)
    try:
        df = conn.execute(
            """
            SELECT *
            FROM feature_matrix
            WHERE COALESCE(market_code, 'US') = ?
            ORDER BY date, ticker
            """,
            [resolved_scope],
        ).df()
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
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
    if frame["prediction"].nunique() <= 1 or frame["target_rank"].nunique() <= 1:
        return 0.0
    corr = frame["prediction"].corr(frame["target_rank"], method="spearman")
    return float(corr) if corr == corr else 0.0


def _fit_model(
    model: object,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_rank"]
    if LGBMRegressor is not None and isinstance(model, LGBMRegressor):
        fit_kwargs: dict[str, object] = {}
        if not val_df.empty and early_stopping is not None and log_evaluation is not None:
            fit_kwargs["eval_set"] = [(val_df[FEATURE_COLUMNS], val_df["target_rank"])]
            fit_kwargs["eval_metric"] = "l2"
            fit_kwargs["callbacks"] = [early_stopping(50, verbose=False), log_evaluation(0)]
        model.fit(x_train, y_train, **fit_kwargs)
        return
    model.fit(x_train, y_train)


def _persist_model_predictions(
    settings: Settings,
    *,
    predictions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> list[Path]:
    if predictions.empty:
        return []
    frame = predictions.copy()
    frame["transform_loaded_at"] = datetime.now(tz=UTC)
    frame["known_at"] = pd.to_datetime(frame["known_at"], utc=True, errors="coerce")
    out_path = warehouse_partition_path(
        settings,
        domain="forecast/model_predictions",
        partition_date=pd.to_datetime(as_of).date(),
        stem=(
            f"model_predictions_{pd.to_datetime(as_of).date().isoformat()}_"
            f"{str(frame['market_code'].iloc[0]).lower()}_{int(frame['horizon_days'].iloc[0])}d_"
            f"{str(frame['prediction_context'].iloc[0]).lower()}"
        ),
    )
    write_parquet(
        frame[
            [
                "date",
                "ticker",
                "market_code",
                "sector",
                "horizon_days",
                "prediction_context",
                "prediction",
                "prob_outperform",
                "confidence_score",
                "observed_excess_alpha",
                "observed_excess_alpha_rank",
                "split_no",
                "known_at",
                "transform_loaded_at",
            ]
        ],
        out_path,
    )
    return [out_path]


def _benchmark_style_metrics(
    test_df: pd.DataFrame,
    *,
    horizon_days: int,
    market_code: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    vol = (
        pd.to_numeric(test_df.get("realized_vol_20d"), errors="coerce")
        .replace(0.0, pd.NA)
        .fillna(1.0)
    )
    strategies = {
        "nifty200_momentum30_style": (
            pd.to_numeric(test_df.get("ret_126d"), errors="coerce").fillna(0.0)
            + pd.to_numeric(test_df.get("ret_252d"), errors="coerce").fillna(0.0)
        )
        / vol,
        "nifty_alpha50_style": pd.to_numeric(test_df.get("ret_252d"), errors="coerce").fillna(0.0),
    }
    for strategy_name, scores in strategies.items():
        frame = test_df.copy()
        frame["prediction"] = pd.to_numeric(scores, errors="coerce").fillna(0.0)
        rows.append(
            {
                "market_code": market_code,
                "horizon_days": horizon_days,
                "strategy_name": strategy_name,
                "metric_name": "rank_ic",
                "metric_value": _rank_ic(frame),
                "source": "walkforward_style_baseline",
            }
        )
        rows.append(
            {
                "market_code": market_code,
                "horizon_days": horizon_days,
                "strategy_name": strategy_name,
                "metric_name": "hit_rate",
                "metric_value": float((frame["target_excess_alpha"].fillna(0.0) > 0).mean()),
                "source": "walkforward_style_baseline",
            }
        )
    return pd.DataFrame(rows)


def _persist_benchmark_metrics(
    settings: Settings,
    *,
    as_of: pd.Timestamp,
    benchmark_metrics: pd.DataFrame,
) -> list[Path]:
    if benchmark_metrics.empty:
        return []
    frame = benchmark_metrics.copy()
    frame["as_of_date"] = pd.to_datetime(as_of).date()
    frame["transform_loaded_at"] = datetime.now(tz=UTC)
    out_path = warehouse_partition_path(
        settings,
        domain="forecast/benchmark_strategy_metrics",
        partition_date=pd.to_datetime(as_of).date(),
        stem=f"benchmark_strategy_metrics_{pd.to_datetime(as_of).date().isoformat()}",
    )
    write_parquet(
        frame[
            [
                "as_of_date",
                "market_code",
                "horizon_days",
                "strategy_name",
                "metric_name",
                "metric_value",
                "source",
                "transform_loaded_at",
            ]
        ],
        out_path,
    )
    return [out_path]


def train_baseline(
    settings: Settings,
    *,
    spec: WalkForwardSpec | None = None,
    horizon_days: int | None = None,
    market_scope: str | None = None,
) -> TrainingArtifacts:
    resolved_horizon = int(horizon_days or settings.label_horizon_days)
    resolved_scope = _resolve_market_scope(settings, market_scope)
    spec = spec or WalkForwardSpec(label_horizon_days=resolved_horizon)
    df = load_training_frame(settings, horizon_days=resolved_horizon, market_scope=resolved_scope)
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
    benchmark_frames: list[pd.DataFrame] = []

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
                    "market_scope": resolved_scope,
                    "horizon_days": resolved_horizon,
                }
            )

        _fit_model(model, train_df=train_df, val_df=val_df)
        test_predictions = test_df[
            [
                "date",
                "ticker",
                "market_code",
                "sector",
                "target_rank",
                "target_excess_alpha",
                "known_at",
            ]
        ].copy()
        test_predictions["prediction"] = model.predict(test_df[FEATURE_COLUMNS])
        test_predictions["split_no"] = split_no
        test_predictions["horizon_days"] = resolved_horizon
        test_predictions["prediction_context"] = "walkforward_test"
        test_predictions["prob_outperform"] = test_predictions["prediction"].clip(0.0, 1.0)
        test_predictions["confidence_score"] = (
            (test_predictions["prediction"] - 0.5).abs() * 2.0
        ).clip(0.0, 1.0)
        test_predictions["observed_excess_alpha"] = test_predictions["target_excess_alpha"]
        test_predictions["observed_excess_alpha_rank"] = test_predictions["target_rank"]

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
        benchmark_frames.append(
            _benchmark_style_metrics(
                test_df,
                horizon_days=resolved_horizon,
                market_code=resolved_scope,
            )
        )

        daily_rank_ics = [_rank_ic(group) for _, group in test_predictions.groupby("date")]
        metrics = {
            "split_no": split_no,
            "rank_ic": float(sum(daily_rank_ics) / len(daily_rank_ics)) if daily_rank_ics else 0.0,
            "hit_rate": float((test_predictions["target_excess_alpha"] > 0).mean()),
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

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metrics_rows)
    report_path = None
    ablation_paths: list[Path] = []
    prediction_paths: list[Path] = []
    benchmark_paths: list[Path] = []

    if not predictions.empty:
        latest_date = pd.to_datetime(predictions["date"]).max()
        nightly_report = build_ranked_report(predictions, as_of=latest_date, settings=settings)
        report_name = (
            f"nightly_ranked_report_{resolved_scope.lower()}_"
            f"{resolved_horizon}d_{latest_date.date().isoformat()}.json"
        )
        report_path = settings.reports_path / report_name
        persist_ranked_report(settings, report=nightly_report, dated_path=report_path)
        prediction_paths.extend(
            _persist_model_predictions(
                settings,
                predictions=predictions,
                as_of=latest_date,
            )
        )
        benchmark_metrics = (
            pd.concat(benchmark_frames, ignore_index=True) if benchmark_frames else pd.DataFrame()
        )
        benchmark_paths.extend(
            _persist_benchmark_metrics(
                settings,
                as_of=latest_date,
                benchmark_metrics=benchmark_metrics,
            )
        )

    feature_importance = None
    if importance_frames:
        feature_importance = pd.DataFrame(importance_frames).set_index("split_no")
        if not predictions.empty:
            latest_date = pd.to_datetime(predictions["date"]).max()
            ablation_paths = _persist_feature_family_ablation(
                settings,
                as_of=latest_date,
                feature_importance=feature_importance.reset_index(),
            )

    live_features = load_feature_frame(settings, market_scope=resolved_scope)
    if not live_features.empty:
        latest_live_date = pd.to_datetime(live_features["date"]).max()
        latest_mask = pd.to_datetime(live_features["date"]) == latest_live_date
        live_frame = live_features.loc[latest_mask].copy()
        live_frame = live_frame.dropna(subset=FEATURE_COLUMNS)
        trainable = df.dropna(subset=FEATURE_COLUMNS)
        if not live_frame.empty and not trainable.empty:
            final_model = _make_model()
            empty_val = pd.DataFrame(columns=trainable.columns)
            _fit_model(final_model, train_df=trainable, val_df=empty_val)
            live_predictions = live_frame[
                ["date", "ticker", "market_code", "sector", "known_at"]
            ].copy()
            live_predictions["prediction"] = final_model.predict(live_frame[FEATURE_COLUMNS])
            live_predictions["split_no"] = 0
            live_predictions["horizon_days"] = resolved_horizon
            live_predictions["prediction_context"] = "live_latest"
            live_predictions["prob_outperform"] = live_predictions["prediction"].clip(0.0, 1.0)
            live_predictions["confidence_score"] = (
                (live_predictions["prediction"] - 0.5).abs() * 2.0
            ).clip(0.0, 1.0)
            live_predictions["observed_excess_alpha"] = pd.NA
            live_predictions["observed_excess_alpha_rank"] = pd.NA
            prediction_paths.extend(
                _persist_model_predictions(
                    settings,
                    predictions=live_predictions,
                    as_of=latest_live_date,
                )
            )

    return TrainingArtifacts(
        metrics=metrics_df,
        predictions=predictions,
        report_path=report_path,
        feature_importance=feature_importance,
        ablation_paths=ablation_paths,
        prediction_paths=prediction_paths,
        benchmark_paths=benchmark_paths,
    )


def _load_spy_returns(settings: Settings) -> pd.DataFrame:
    """Load SPY price data and compute returns + realized vol for HMM."""
    conn = connect(settings)
    try:
        prices = conn.execute(
            """
            SELECT date, adj_close
            FROM prices
            WHERE ticker = 'SPY'
            ORDER BY date
            """
        ).df()
    finally:
        conn.close()

    if prices.empty:
        return pd.DataFrame()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index("date").sort_index()
    prices["ret_1d"] = prices["adj_close"].pct_change()
    prices["realized_vol_20d"] = prices["ret_1d"].rolling(20).std()
    return prices.dropna()


def train_regime_specific(
    settings: Settings, *, spec: WalkForwardSpec | None = None
) -> TrainingArtifacts:
    """Train separate LightGBM models per HMM regime state."""
    try:
        from ai_analyst.regime.hmm import HMMRegimeDetector
    except ImportError as exc:
        raise RuntimeError(
            "Install the [v2] extras for HMM regime detection (hmmlearn)."
        ) from exc

    spec = spec or WalkForwardSpec(label_horizon_days=settings.label_horizon_days)
    df = load_training_frame(settings, horizon_days=settings.label_horizon_days)
    if df.empty:
        raise ValueError("No feature/label data available. Run feature materialization first.")

    spy = _load_spy_returns(settings)
    if spy.empty:
        raise ValueError("No SPY price data available. Collect SPY prices first.")

    detector = HMMRegimeDetector(n_states=settings.hmm_n_states)
    detector.fit(spy["ret_1d"], spy["realized_vol_20d"])
    result = detector.predict(spy["ret_1d"], spy["realized_vol_20d"])

    regime_labels = pd.Series(
        [result.state_labels.get(int(s), "unknown") for s in result.states],
        index=result.states.index,
        name="hmm_regime_label",
    )

    df["date_idx"] = pd.to_datetime(df["date"])
    regime_map = regime_labels.to_dict()
    df["hmm_regime_label"] = df["date_idx"].map(regime_map).fillna("unknown")

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    for regime in sorted(df["hmm_regime_label"].unique()):
        if regime == "unknown":
            continue
        regime_df = df[df["hmm_regime_label"] == regime].copy()
        if len(regime_df) < 500:
            logger.info("Skipping regime '%s': only %d rows (need 500).", regime, len(regime_df))
            continue

        splits = generate_walk_forward_splits(regime_df, spec)
        if not splits:
            logger.info("Skipping regime '%s': insufficient history for walk-forward.", regime)
            continue

        for split_no, split in enumerate(splits, start=1):
            train_df = regime_df.loc[split.train_mask].dropna(subset=FEATURE_COLUMNS)
            val_df = regime_df.loc[split.validation_mask].dropna(subset=FEATURE_COLUMNS)
            test_df = regime_df.loc[split.test_mask].dropna(subset=FEATURE_COLUMNS)
            if train_df.empty or test_df.empty:
                continue

            model = _make_model()
            _fit_model(model, train_df=train_df, val_df=val_df)

            preds = test_df[
                ["date", "ticker", "sector", "target_rank", "target_excess_alpha"]
            ].copy()
            preds["prediction"] = model.predict(test_df[FEATURE_COLUMNS])
            preds["split_no"] = split_no
            preds["regime"] = regime
            preds["prob_outperform"] = preds["prediction"].clip(0.0, 1.0)
            preds["confidence_score"] = ((preds["prediction"] - 0.5).abs() * 2.0).clip(0.0, 1.0)
            all_predictions.append(preds)

            daily_rank_ics = [_rank_ic(group) for _, group in preds.groupby("date")]
            all_metrics.append(
                {
                    "regime": regime,
                    "split_no": split_no,
                    "rank_ic": float(sum(daily_rank_ics) / len(daily_rank_ics))
                    if daily_rank_ics
                    else 0.0,
                    "hit_rate": float((preds["target_excess_alpha"] > 0).mean()),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )

    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(all_metrics)
    report_path = None
    if not predictions.empty:
        latest_date = pd.to_datetime(predictions["date"]).max()
        nightly_report = build_ranked_report(predictions, as_of=latest_date, settings=settings)
        report_path = (
            settings.reports_path
            / f"nightly_ranked_report_regime_{latest_date.date().isoformat()}.json"
        )
        persist_ranked_report(settings, report=nightly_report, dated_path=report_path)

    return TrainingArtifacts(
        metrics=metrics_df,
        predictions=predictions,
        report_path=report_path,
    )

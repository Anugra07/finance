from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.events.exposures import compute_sector_context_shocks
from ai_analyst.events.ontology import THEME_TO_FEATURE_COLUMN
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.layout import warehouse_partition_path

SHARES_METRICS = {
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "CommonStocksIncludingAdditionalPaidInCapitalSharesOutstanding",
}

GEO_CONTEXT_COLUMNS = [
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


def select_v1_universe(settings: Settings) -> pd.DataFrame:
    conn = connect(settings)
    try:
        prices = conn.execute(
            """
            SELECT ticker, date, adj_close, volume
            FROM prices
            WHERE adj_close IS NOT NULL
              AND volume IS NOT NULL
            ORDER BY ticker, date
            """
        ).df()
        members = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_at DESC) AS row_num
                FROM universe_membership
            )
            SELECT *
            FROM ranked
            WHERE row_num = 1
            """
        ).df()
    finally:
        conn.close()

    if prices.empty or members.empty:
        return pd.DataFrame()

    prices["dollar_volume"] = prices["adj_close"] * prices["volume"]
    adv = (
        prices.sort_values(["ticker", "date"])
        .groupby("ticker", group_keys=False)
        .tail(60)
        .groupby("ticker", as_index=False)["dollar_volume"]
        .mean()
        .rename(columns={"dollar_volume": "avg_dollar_volume_60d"})
    )
    selected = (
        members.merge(adv, on="ticker", how="inner")
        .sort_values("avg_dollar_volume_60d", ascending=False)
        .head(settings.v1_universe_size)
        .copy()
    )
    selected["universe_rank"] = np.arange(1, len(selected) + 1)
    selected["transform_loaded_at"] = datetime.now(tz=UTC)
    return selected


def materialize_v1_universe(settings: Settings) -> list[str]:
    selected = select_v1_universe(settings)
    if selected.empty:
        return []
    snapshot_at = pd.to_datetime(selected["snapshot_at"].max(), utc=True).to_pydatetime()
    partition_date = snapshot_at.date()
    out_path = warehouse_partition_path(
        settings,
        domain="universe/v1_top150",
        partition_date=partition_date,
        stem=f"v1_top150_{partition_date.isoformat()}",
    )
    write_parquet(selected, out_path)
    return [str(out_path)]


def _attach_shares_outstanding(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    if shares.empty:
        prices["shares_outstanding"] = np.nan
        prices["share_turnover"] = np.nan
        return prices

    enriched: list[pd.DataFrame] = []
    for cik, price_group in prices.groupby("cik", dropna=False):
        if pd.isna(cik):
            price_group = price_group.copy()
            price_group["shares_outstanding"] = np.nan
            enriched.append(price_group)
            continue

        share_group = shares.loc[shares["cik"] == cik].copy()
        if share_group.empty:
            price_group = price_group.copy()
            price_group["shares_outstanding"] = np.nan
            enriched.append(price_group)
            continue

        share_group = share_group.sort_values("filing_date")
        price_group = price_group.sort_values("date")
        merged = pd.merge_asof(
            price_group,
            share_group[["filing_date", "value"]].rename(columns={"value": "shares_outstanding"}),
            left_on="date",
            right_on="filing_date",
            direction="backward",
        )
        enriched.append(merged.drop(columns=["filing_date"]))

    out = pd.concat(enriched, ignore_index=True)
    out["share_turnover"] = out["volume"] / out["shares_outstanding"]
    return out


def _attach_geo_context_features(
    features: pd.DataFrame,
    *,
    theme_daily: pd.DataFrame,
    sector_exposures: pd.DataFrame,
    stock_exposures: pd.DataFrame,
    historical_analogs: pd.DataFrame | None = None,
    causal_state_daily: pd.DataFrame | None = None,
    causal_chain_activations: pd.DataFrame | None = None,
    theme_regimes: pd.DataFrame | None = None,
    cross_asset_confirmation_daily: pd.DataFrame | None = None,
    pricing_discipline_daily: pd.DataFrame | None = None,
    trade_readiness_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = features.copy()
    frame["date_key"] = pd.to_datetime(frame["date"]).dt.date
    for column in ["analog_match_score", "sector_context_shock", "stock_context_shock"]:
        frame[column] = 0.0

    if theme_daily.empty:
        for column in GEO_CONTEXT_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        return frame.drop(columns=["date_key"])

    theme_daily = theme_daily.copy()
    theme_daily["date"] = pd.to_datetime(theme_daily["date"]).dt.date
    intensity_wide = (
        theme_daily.pivot_table(index="date", columns="theme", values="intensity", aggfunc="sum")
        .fillna(0.0)
        .reset_index()
    )
    intensity_wide.columns.name = None
    for theme_name, feature_name in THEME_TO_FEATURE_COLUMN.items():
        if theme_name not in intensity_wide.columns:
            intensity_wide[theme_name] = 0.0
        intensity_wide = intensity_wide.rename(columns={theme_name: feature_name})

    novelty_dispersion = theme_daily.groupby("date", as_index=False).agg(
        geo_novelty_score=("avg_novelty", "mean"),
        event_dispersion_score=("event_dispersion_score", "mean"),
        theme_abs_total=("intensity", lambda values: float(np.abs(values).sum())),
        theme_abs_max=("intensity", lambda values: float(np.abs(values).max())),
    )
    commodity_scores = (
        theme_daily.loc[
            theme_daily["theme"].isin(
                [
                    "oil_supply_risk",
                    "gas_supply_risk",
                    "industrial_metal_tightness",
                ]
            )
        ]
        .groupby("date", as_index=False)["intensity"]
        .sum()
        .rename(columns={"intensity": "commodity_shock_score"})
    )
    novelty_dispersion = novelty_dispersion.merge(commodity_scores, on="date", how="left")
    novelty_dispersion["commodity_shock_score"] = novelty_dispersion[
        "commodity_shock_score"
    ].fillna(0.0)
    novelty_dispersion["regional_concentration_risk"] = np.where(
        novelty_dispersion["theme_abs_total"] > 0,
        novelty_dispersion["theme_abs_max"] / novelty_dispersion["theme_abs_total"],
        0.0,
    )
    novelty_dispersion = novelty_dispersion.drop(columns=["theme_abs_total", "theme_abs_max"])
    intensity_wide = intensity_wide.rename(columns={"date": "date_key"})
    novelty_dispersion = novelty_dispersion.rename(columns={"date": "date_key"})

    theme_feature_columns = [
        column for column in THEME_TO_FEATURE_COLUMN.values() if column in intensity_wide.columns
    ]
    frame = frame.merge(
        intensity_wide[["date_key", *theme_feature_columns]],
        on="date_key",
        how="left",
    )
    frame = frame.merge(
        novelty_dispersion,
        on="date_key",
        how="left",
    )

    if not sector_exposures.empty:
        sector_shocks = compute_sector_context_shocks(theme_daily, sector_exposures)
        if not sector_shocks.empty:
            frame = frame.merge(
                sector_shocks.rename(columns={"as_of_date": "date_key"}),
                on=["date_key", "sector"],
                how="left",
            )
            frame["sector_context_shock"] = frame["context_shock"].fillna(0.0)
            frame = frame.drop(
                columns=["context_shock", "top_theme", "supporting_themes"],
                errors="ignore",
            )

    if not stock_exposures.empty:
        stock_exposures = stock_exposures.copy()
        stock_exposures["theme"] = stock_exposures["theme"].astype(str)
        stock_theme_values = theme_daily[["date", "theme", "intensity"]].copy()
        stock_theme_values["date"] = pd.to_datetime(stock_theme_values["date"]).dt.date
        stock_adjustments = (
            stock_exposures.merge(stock_theme_values, on="theme", how="left")
            .assign(
                contribution=lambda df: df["exposure"].fillna(0.0) * df["intensity"].fillna(0.0)
            )
            .groupby(["ticker", "date"], as_index=False)["contribution"]
            .sum()
            .rename(columns={"date": "date_key", "contribution": "stock_exposure_adjustment"})
        )
        frame = frame.merge(stock_adjustments, on=["ticker", "date_key"], how="left")
        frame["stock_context_shock"] = frame["sector_context_shock"].fillna(0.0) + frame[
            "stock_exposure_adjustment"
        ].fillna(0.0)
        frame = frame.drop(columns=["stock_exposure_adjustment"], errors="ignore")
    else:
        frame["stock_context_shock"] = frame["sector_context_shock"].fillna(0.0)

    if historical_analogs is not None and not historical_analogs.empty:
        analog_scores = historical_analogs.copy()
        analog_scores["as_of_date"] = pd.to_datetime(analog_scores["as_of_date"]).dt.date
        analog_scores = analog_scores.assign(
            analogy_failure_risk=pd.to_numeric(
                analog_scores["analogy_failure_risk"],
                errors="coerce",
            ).fillna(0.0)
        )
        analog_scores = analog_scores.groupby("as_of_date", as_index=False).agg(
            analog_match_score=("similarity_score", "max"),
            analog_failure_risk=("analogy_failure_risk", "mean"),
        )
        analog_scores = analog_scores.rename(columns={"as_of_date": "date_key"})
        frame = frame.drop(columns=["analog_match_score"], errors="ignore").merge(
            analog_scores,
            on="date_key",
            how="left",
        )

    if causal_state_daily is not None and not causal_state_daily.empty:
        state_frame = causal_state_daily.copy()
        state_frame["as_of_date"] = pd.to_datetime(state_frame["as_of_date"]).dt.date
        state_frame["state_numeric"] = pd.to_numeric(
            state_frame["state_value"],
            errors="coerce",
        )
        pivot = (
            state_frame.pivot_table(
                index="as_of_date",
                columns="state_key",
                values="state_numeric",
                aggfunc="last",
            )
            .reset_index()
            .rename(columns={"as_of_date": "date_key"})
        )
        rename_map = {
            "regime_score": "regime_score",
            "regime_is_escalating": "regime_is_escalating",
            "theme_convergence_score": "theme_convergence_score",
            "pricing_geo_signal_strength": "pricing_geo_signal_strength",
            "pricing_mediator_confirmation": "pricing_mediator_confirmation",
            "pricing_market_response_strength": "pricing_market_response_strength",
            "pricing_divergence": "pricing_divergence_score",
            "pricing_crowdedness_proxy": "pricing_crowdedness_proxy",
            "pricing_follow_through_status": "pricing_follow_through_status",
        }
        pivot = pivot.rename(columns=rename_map)
        frame = frame.merge(pivot, on="date_key", how="left")

    if causal_chain_activations is not None and not causal_chain_activations.empty:
        chain_frame = causal_chain_activations.copy()
        chain_frame["as_of_date"] = pd.to_datetime(chain_frame["as_of_date"]).dt.date
        chain_frame["signed_weight"] = pd.to_numeric(chain_frame["sign"], errors="coerce").fillna(
            0.0
        ) * pd.to_numeric(chain_frame["weight"], errors="coerce").fillna(0.0)
        chain_summary = chain_frame.groupby("as_of_date", as_index=False).agg(
            causal_chain_count=("theme", "count"),
            causal_chain_avg_confidence=("activation_confidence", "mean"),
            causal_net_sign=("signed_weight", "sum"),
        )
        chain_summary = chain_summary.rename(columns={"as_of_date": "date_key"})
        frame = frame.merge(chain_summary, on="date_key", how="left")

    if theme_regimes is not None and not theme_regimes.empty:
        regime_frame = theme_regimes.copy()
        regime_frame["as_of_date"] = pd.to_datetime(regime_frame["as_of_date"]).dt.date
        regime_frame["regime_is_escalating"] = np.where(
            regime_frame["regime_name"].astype(str) == "escalating",
            1.0,
            0.0,
        )
        regime_frame = (
            regime_frame.groupby("as_of_date", as_index=False)
            .agg(
                regime_score=("regime_score", "max"),
                regime_is_escalating=("regime_is_escalating", "max"),
            )
            .rename(columns={"as_of_date": "date_key"})
        )
        for column in ["regime_score", "regime_is_escalating"]:
            if column in frame.columns:
                frame = frame.drop(columns=[column])
        frame = frame.merge(regime_frame, on="date_key", how="left")

    def _merge_key_value_state(
        input_frame: pd.DataFrame | None,
        *,
        date_column: str,
        key_column: str,
        value_column: str,
        rename_map: dict[str, str],
    ) -> None:
        nonlocal frame
        if input_frame is None or input_frame.empty:
            return
        state = input_frame.copy()
        state[date_column] = pd.to_datetime(state[date_column]).dt.date
        pivot = (
            state.pivot_table(
                index=date_column,
                columns=key_column,
                values=value_column,
                aggfunc="last",
            )
            .reset_index()
            .rename(columns={date_column: "date_key"})
            .rename(columns=rename_map)
        )
        frame = frame.merge(pivot, on="date_key", how="left")

    _merge_key_value_state(
        cross_asset_confirmation_daily,
        date_column="as_of_date",
        key_column="confirmation_key",
        value_column="value",
        rename_map={
            "energy_confirmation": "cross_asset_energy_confirmation",
            "vol_confirmation": "cross_asset_vol_confirmation",
            "credit_confirmation": "cross_asset_credit_confirmation",
            "rates_confirmation": "cross_asset_rates_confirmation",
            "fx_confirmation": "cross_asset_fx_confirmation",
            "sector_flow_confirmation": "cross_asset_sector_flow_confirmation",
            "peer_confirmation": "cross_asset_peer_confirmation",
            "aggregate_confirmation": "cross_asset_aggregate_confirmation",
        },
    )
    _merge_key_value_state(
        pricing_discipline_daily,
        date_column="as_of_date",
        key_column="discipline_key",
        value_column="value",
        rename_map={
            "market_confirmation": "pricing_discipline_market_confirmation",
            "move_lateness": "pricing_discipline_move_lateness",
            "overextension": "pricing_discipline_overextension",
            "reaction_vs_follow_through": "pricing_discipline_reaction_follow_through",
        },
    )
    _merge_key_value_state(
        trade_readiness_daily,
        date_column="as_of_date",
        key_column="readiness_key",
        value_column="value",
        rename_map={
            "thesis_validity": "trade_readiness_thesis_validity",
            "pricing_alignment": "trade_readiness_pricing_alignment",
            "timing_quality": "trade_readiness_timing_quality",
            "liquidity_quality": "trade_readiness_liquidity_quality",
            "risk_reward_quality": "trade_readiness_risk_reward_quality",
        },
    )

    for column in GEO_CONTEXT_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.drop(columns=["date_key"])


def build_feature_and_label_frames(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = connect(settings)
    try:
        prices = conn.execute(
            """
            SELECT
                p.*,
                u.sector,
                u.cik
            FROM prices p
            LEFT JOIN (
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_at DESC) AS row_num
                    FROM v1_universe
                )
                SELECT ticker, sector, cik
                FROM ranked
                WHERE row_num = 1
            ) u
            ON p.ticker = u.ticker
            WHERE p.ticker = 'SPY'
               OR p.ticker IN (SELECT ticker FROM v1_universe)
            ORDER BY p.ticker, p.date
            """
        ).df()
        shares = conn.execute(
            """
            SELECT cik, filing_date, value
            FROM edgar_companyfacts
            WHERE metric_name IN (
                'CommonStockSharesOutstanding',
                'EntityCommonStockSharesOutstanding',
                'CommonStocksIncludingAdditionalPaidInCapitalSharesOutstanding'
            )
              AND value IS NOT NULL
            ORDER BY cik, filing_date
            """
        ).df()
        theme_daily = conn.execute("SELECT * FROM theme_intensity_daily").df()
        sector_exposures = conn.execute("SELECT * FROM sector_theme_exposure").df()
        stock_exposures = conn.execute("SELECT * FROM stock_theme_exposure").df()
        historical_analogs = conn.execute("SELECT * FROM historical_analogs").df()
        causal_state_daily = conn.execute("SELECT * FROM causal_state_daily").df()
        causal_chain_activations = conn.execute("SELECT * FROM causal_chain_activations").df()
        theme_regimes = conn.execute("SELECT * FROM theme_regimes").df()
        cross_asset_confirmation_daily = conn.execute(
            "SELECT * FROM cross_asset_confirmation_daily"
        ).df()
        pricing_discipline_daily = conn.execute("SELECT * FROM pricing_discipline_daily").df()
        trade_readiness_daily = conn.execute("SELECT * FROM trade_readiness_daily").df()
    finally:
        conn.close()

    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    shares["filing_date"] = pd.to_datetime(shares["filing_date"]).astype("datetime64[ns]")
    prices = prices.sort_values(["ticker", "date"])
    prices = _attach_shares_outstanding(prices, shares)

    benchmark = prices.loc[prices["ticker"] == "SPY", ["date", "adj_close"]].drop_duplicates()
    benchmark = benchmark.rename(columns={"adj_close": "spy_adj_close"}).sort_values("date")
    benchmark["benchmark_return_5d"] = np.log(
        benchmark["spy_adj_close"] / benchmark["spy_adj_close"].shift(5)
    )
    benchmark = benchmark[["date", "benchmark_return_5d"]]

    feature_frames: list[pd.DataFrame] = []

    for _ticker, group in prices.groupby("ticker", group_keys=False):
        group = group.sort_values("date").copy()
        group["log_close"] = np.log(group["adj_close"])
        group["ret_1d"] = group["log_close"].diff(1)
        group["ret_5d"] = group["log_close"].diff(5)
        group["ret_20d"] = group["log_close"].diff(20)
        group["ret_60d"] = group["log_close"].diff(60)
        group["overnight_gap"] = np.log(group["adj_open"] / group["adj_close"].shift(1))
        group["range_norm"] = (group["adj_high"] - group["adj_low"]) / group["adj_close"]
        group["realized_vol_20d"] = group["ret_1d"].rolling(20).std() * np.sqrt(20)
        group["volume_surprise_20d"] = group["volume"] / group["volume"].rolling(20).mean()
        group["return_5d"] = np.log(group["adj_close"].shift(-5) / group["adj_close"])
        feature_frames.append(group)

    features = pd.concat(feature_frames, ignore_index=True)
    features = features.merge(benchmark, on="date", how="left")
    features["excess_alpha_5d"] = features["return_5d"] - features["benchmark_return_5d"]
    features = features.loc[features["ticker"] != "SPY"].copy()
    features = _attach_geo_context_features(
        features,
        theme_daily=theme_daily,
        sector_exposures=sector_exposures,
        stock_exposures=stock_exposures,
        historical_analogs=historical_analogs,
        causal_state_daily=causal_state_daily,
        causal_chain_activations=causal_chain_activations,
        theme_regimes=theme_regimes,
        cross_asset_confirmation_daily=cross_asset_confirmation_daily,
        pricing_discipline_daily=pricing_discipline_daily,
        trade_readiness_daily=trade_readiness_daily,
    )

    base_features = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "overnight_gap",
        "range_norm",
        "realized_vol_20d",
        "volume_surprise_20d",
        "share_turnover",
    ]
    for feature_name in base_features:
        features[f"{feature_name}_sector_pct"] = features.groupby(["date", "sector"])[
            feature_name
        ].rank(pct=True)

    features["known_at"] = pd.to_datetime(features["known_at"], utc=True)
    features["transform_loaded_at"] = datetime.now(tz=UTC)
    feature_rank_cols = [f"{name}_sector_pct" for name in base_features]
    required_feature_cols = base_features + feature_rank_cols + ["benchmark_return_5d", "return_5d"]

    labels = features[
        ["date", "ticker", "return_5d", "benchmark_return_5d", "excess_alpha_5d", "known_at"]
    ].copy()
    labels["excess_alpha_rank"] = labels.groupby("date")["excess_alpha_5d"].rank(pct=True)
    labels["transform_loaded_at"] = datetime.now(tz=UTC)
    labels = labels.dropna(
        subset=["return_5d", "benchmark_return_5d", "excess_alpha_5d", "excess_alpha_rank"]
    )
    features = features.merge(labels[["date", "ticker"]], on=["date", "ticker"], how="inner")

    feature_cols = [
        "date",
        "ticker",
        "sector",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "overnight_gap",
        "range_norm",
        "realized_vol_20d",
        "volume_surprise_20d",
        "share_turnover",
        *GEO_CONTEXT_COLUMNS,
        "known_at",
        "transform_loaded_at",
    ]
    features = features.dropna(subset=required_feature_cols)
    features = features[feature_cols + feature_rank_cols]
    labels["date"] = pd.to_datetime(labels["date"])
    return features, labels


def materialize_features_and_labels(settings: Settings) -> tuple[list[str], list[str]]:
    features, labels = build_feature_and_label_frames(settings)
    feature_paths: list[str] = []
    label_paths: list[str] = []
    if features.empty:
        return feature_paths, label_paths

    for trade_date, frame in features.groupby(features["date"].dt.date):
        out_path = warehouse_partition_path(
            settings,
            domain="features/daily",
            partition_date=trade_date,
            stem=f"features_{trade_date.isoformat()}",
        )
        write_parquet(frame, out_path)
        feature_paths.append(str(out_path))

    for trade_date, frame in labels.groupby(labels["date"].dt.date):
        out_path = warehouse_partition_path(
            settings,
            domain="labels/daily",
            partition_date=trade_date,
            stem=f"labels_{trade_date.isoformat()}",
        )
        write_parquet(frame, out_path)
        label_paths.append(str(out_path))

    return feature_paths, label_paths

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from math import floor
from typing import Any

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.core.models import (
    AccountProfile,
    OrderabilityState,
    ShortlistCandidate,
    ShortlistDecision,
)
from ai_analyst.events.context import LocalMacroContextSource
from ai_analyst.warehouse.database import connect


def _resolve_market_scope(settings: Settings, market_scope: str | None = None) -> str:
    return str(market_scope or settings.primary_market_scope or "US").upper()


def _zerodha_roundtrip_costs(notional: float) -> tuple[float, float]:
    if notional <= 0:
        return 0.0, 0.0
    buy_stt = notional * 0.001
    sell_stt = notional * 0.001
    transaction_buy = notional * 0.0000307
    transaction_sell = notional * 0.0000307
    sebi_buy = notional * 0.000001
    sebi_sell = notional * 0.000001
    stamp_duty = notional * 0.00015
    gst = 0.18 * (transaction_buy + transaction_sell + sebi_buy + sebi_sell)
    dp_charge = 15.34
    entry_cost = (
        buy_stt
        + transaction_buy
        + sebi_buy
        + stamp_duty
        + (0.18 * (transaction_buy + sebi_buy))
    )
    exit_cost = (
        sell_stt
        + transaction_sell
        + sebi_sell
        + (0.18 * (transaction_sell + sebi_sell))
        + dp_charge
    )
    total_cost = entry_cost + exit_cost + gst
    return float(entry_cost), float(total_cost - entry_cost)


def _previous_trading_day(target: pd.Timestamp, holidays: set[pd.Timestamp]) -> pd.Timestamp:
    probe = pd.Timestamp(target).normalize()
    while probe.weekday() >= 5 or probe in holidays:
        probe -= pd.Timedelta(days=1)
    return probe


def _prediction_bucket_edge(history: pd.DataFrame, prediction: float) -> float:
    if history.empty:
        return 0.0
    frame = history.copy()
    frame["bucket"] = pd.qcut(
        frame["prediction"].rank(method="first"),
        q=min(10, max(2, len(frame))),
        labels=False,
        duplicates="drop",
    )
    bucket_summary = frame.groupby("bucket", as_index=False).agg(
        prediction_floor=("prediction", "min"),
        prediction_ceiling=("prediction", "max"),
        mean_edge=("observed_excess_alpha", "mean"),
    )
    for _, row in bucket_summary.sort_values("prediction_floor").iterrows():
        if float(row["prediction_floor"]) <= prediction <= float(row["prediction_ceiling"]):
            return float(row["mean_edge"] or 0.0)
    return float(bucket_summary["mean_edge"].iloc[-1]) if not bucket_summary.empty else 0.0


def _rank_ic(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    if frame["prediction"].nunique() <= 1 or frame["observed_excess_alpha_rank"].nunique() <= 1:
        return 0.0
    corr = frame["prediction"].corr(frame["observed_excess_alpha_rank"], method="spearman")
    return float(corr) if corr == corr else 0.0


def _model_benchmark_gate(
    historical: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    if historical.empty or benchmark_metrics.empty:
        return False, {"status": "missing_benchmark_history"}
    eval_frame = historical.dropna(
        subset=["prediction", "observed_excess_alpha", "observed_excess_alpha_rank"]
    ).copy()
    if eval_frame.empty:
        return False, {"status": "missing_benchmark_history"}

    model_hit_rate = float((eval_frame["observed_excess_alpha"] > 0).mean())
    daily_rank_ics = [_rank_ic(group) for _, group in eval_frame.groupby("date")]
    model_rank_ic = (
        float(sum(daily_rank_ics) / len(daily_rank_ics)) if daily_rank_ics else 0.0
    )
    if model_rank_ic == 0.0 and len(eval_frame) >= 3:
        model_rank_ic = _rank_ic(eval_frame)

    bench = benchmark_metrics.copy()
    latest_bench_date = pd.to_datetime(bench["as_of_date"]).max()
    bench = bench.loc[pd.to_datetime(bench["as_of_date"]) == latest_bench_date].copy()
    strategy_summary = []
    gate_pass = True
    for strategy_name, group in bench.groupby("strategy_name"):
        metric_map = {
            str(row["metric_name"]): float(row["metric_value"] or 0.0)
            for _, row in group.iterrows()
        }
        beats_strategy = (
            model_rank_ic > float(metric_map.get("rank_ic", 0.0))
            and model_hit_rate > float(metric_map.get("hit_rate", 0.0))
        )
        gate_pass = gate_pass and beats_strategy
        strategy_summary.append(
            {
                "strategy_name": strategy_name,
                "benchmark_rank_ic": float(metric_map.get("rank_ic", 0.0)),
                "benchmark_hit_rate": float(metric_map.get("hit_rate", 0.0)),
                "model_beats_strategy": beats_strategy,
            }
        )
    return gate_pass, {
        "status": "ok",
        "as_of_date": latest_bench_date.date().isoformat(),
        "model_rank_ic": round(model_rank_ic, 6),
        "model_hit_rate": round(model_hit_rate, 6),
        "strategies": strategy_summary,
    }


def _timing_gate_status(prediction_5d: float | None) -> tuple[str, list[str]]:
    if prediction_5d is None or pd.isna(prediction_5d):
        return "missing", ["failed_timing_gate"]
    score = float(prediction_5d)
    if score >= 0.6:
        return "pass", []
    if score >= 0.5:
        return "watch", ["failed_timing_gate"]
    return "fail", ["failed_timing_gate"]


def _decision_mode(
    *,
    conviction: float,
    pricing_confidence: float,
    timing_gate_status: str,
    orderability_status: str,
    downgrade_reasons: list[str],
) -> str:
    hard_stop_reasons = {
        "stale_market_data",
        "insufficient_capital",
        "weak_pricing_confirmation",
        "weak_tradability",
    }
    if any(reason in hard_stop_reasons for reason in downgrade_reasons):
        return "research_only"
    if timing_gate_status != "pass" or "weak_monthly_model" in downgrade_reasons:
        return "watch"
    if orderability_status not in {"affordable", "supported"}:
        return "research_only"
    if pricing_confidence < 0.55:
        return "research_only"
    if conviction >= 0.7:
        return "actionable_bullish"
    return "watch"


def build_shortlist(
    settings: Settings,
    *,
    as_of: datetime,
    market_scope: str = "IN",
    capital: float | None = None,
    base_currency: str | None = None,
    target_horizon_days: int = 21,
    max_candidates: int = 3,
) -> dict[str, object]:
    resolved_scope = _resolve_market_scope(settings, market_scope)
    resolved_capital = float(capital or settings.small_account_default_capital)
    resolved_currency = str(base_currency or settings.small_account_default_currency).upper()
    account_profile = AccountProfile(
        capital=resolved_capital,
        base_currency=resolved_currency,
        market_scope=resolved_scope,  # type: ignore[arg-type]
        max_names=max_candidates,
    )

    if resolved_scope != "IN":
        decision = ShortlistDecision(
            system_mode="research_only",
            market_scope=resolved_scope,  # type: ignore[arg-type]
            capital=resolved_capital,
            base_currency=resolved_currency,
            target_horizon_days=target_horizon_days,
            market_summary={"status": "unsupported_market_scope"},
            shortlist=[],
            downgrade_reasons=["unsupported_market_scope"],
            confidence_breakdown={"decision_confidence": 0.0},
        )
        return asdict(decision)

    conn = connect(settings)
    try:
        monthly_live = conn.execute(
            """
            SELECT *
            FROM model_predictions
            WHERE market_code = ?
              AND horizon_days = ?
              AND prediction_context = 'live_latest'
              AND date <= ?
            ORDER BY date DESC, prediction DESC, ticker
            """,
            [resolved_scope, target_horizon_days, as_of.date()],
        ).df()
        timing_live = conn.execute(
            """
            SELECT ticker, prediction
            FROM model_predictions
            WHERE market_code = ?
              AND horizon_days = 5
              AND prediction_context = 'live_latest'
              AND date <= ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
            """,
            [resolved_scope, as_of.date()],
        ).df()
        historical = conn.execute(
            """
            SELECT date, ticker, prediction, observed_excess_alpha, observed_excess_alpha_rank
            FROM model_predictions
            WHERE market_code = ?
              AND horizon_days = ?
              AND prediction_context = 'walkforward_test'
              AND observed_excess_alpha IS NOT NULL
            ORDER BY date, ticker
            """,
            [resolved_scope, target_horizon_days],
        ).df()
        prices = conn.execute(
            """
            SELECT ticker, date, close, adj_close, known_at, exchange_code, currency, volume
            FROM prices
            WHERE COALESCE(market_code, 'US') = ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
            """,
            [resolved_scope],
        ).df()
        security_master = conn.execute(
            """
            SELECT *
            FROM security_master
            WHERE market_code = ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_at DESC) = 1
            """,
            [resolved_scope],
        ).df()
        latest_features = conn.execute(
            """
            SELECT *
            FROM feature_matrix
            WHERE COALESCE(market_code, 'US') = ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
            """,
            [resolved_scope],
        ).df()
        holidays = conn.execute(
            """
            SELECT holiday_date
            FROM market_holidays
            WHERE market_code = ?
            """,
            [resolved_scope],
        ).df()
        benchmark_metrics = conn.execute(
            """
            SELECT *
            FROM benchmark_strategy_metrics
            WHERE market_code = ?
              AND horizon_days = ?
            ORDER BY as_of_date DESC, strategy_name, metric_name
            """,
            [resolved_scope, target_horizon_days],
        ).df()
    finally:
        conn.close()

    context = LocalMacroContextSource(settings)
    theme_intensities = context.get_theme_intensities(as_of=as_of)
    sector_rankings = context.get_sector_rankings(as_of=as_of, limit=10)

    if monthly_live.empty:
        decision = ShortlistDecision(
            system_mode="research_only",
            market_scope=resolved_scope,  # type: ignore[arg-type]
            capital=resolved_capital,
            base_currency=resolved_currency,
            target_horizon_days=target_horizon_days,
            market_summary={
                "status": "weak_monthly_model",
                "top_themes": theme_intensities.head(5).to_dict(orient="records"),
            },
            shortlist=[],
            downgrade_reasons=["weak_monthly_model"],
            confidence_breakdown={"decision_confidence": 0.0},
        )
        return asdict(decision)

    latest_prediction_date = pd.to_datetime(monthly_live["date"]).max()
    latest_prediction_mask = pd.to_datetime(monthly_live["date"]) == latest_prediction_date
    monthly_live = monthly_live.loc[latest_prediction_mask].copy()
    monthly_live = monthly_live.sort_values(["prediction", "ticker"], ascending=[False, True])

    price_map = prices.set_index("ticker").to_dict(orient="index") if not prices.empty else {}
    timing_map = (
        timing_live.set_index("ticker")["prediction"].to_dict() if not timing_live.empty else {}
    )
    security_map = (
        security_master.set_index("ticker").to_dict(orient="index")
        if not security_master.empty
        else {}
    )
    feature_map = (
        latest_features.set_index("ticker").to_dict(orient="index")
        if not latest_features.empty
        else {}
    )
    holiday_set = (
        set(pd.to_datetime(holidays["holiday_date"]).dt.normalize())
        if not holidays.empty
        else set()
    )
    expected_trade_date = _previous_trading_day(pd.Timestamp(as_of.date()), holiday_set)
    stale_market_data = latest_prediction_date.normalize() < expected_trade_date
    benchmark_gate_pass, benchmark_gate_summary = _model_benchmark_gate(
        historical,
        benchmark_metrics,
    )

    candidates: list[ShortlistCandidate] = []
    rejected: list[dict[str, object]] = []

    for _, row in monthly_live.iterrows():
        ticker = str(row["ticker"])
        price_row = price_map.get(ticker, {})
        feature_row = feature_map.get(ticker, {})
        security_row = security_map.get(ticker, {})
        latest_price_date = pd.to_datetime(price_row.get("date"), errors="coerce")
        current_price = float(price_row.get("close") or price_row.get("adj_close") or 0.0)
        if current_price <= 0:
            rejected.append({"ticker": ticker, "reasons": ["stale_market_data"]})
            continue
        price_is_stale = bool(pd.isna(latest_price_date)) or (
            latest_price_date.normalize() < expected_trade_date
        )
        max_affordable_shares = floor(resolved_capital / current_price)
        suggested_shares = floor(resolved_capital / current_price)
        if suggested_shares < 1:
            suggested_shares = min(max_affordable_shares, 1)
        notional = current_price * suggested_shares
        entry_cost, exit_cost = _zerodha_roundtrip_costs(notional)
        expected_gross_edge = _prediction_bucket_edge(historical, float(row["prediction"]))
        total_cost = entry_cost + exit_cost
        expected_net_edge = expected_gross_edge - (total_cost / max(notional, 1.0))

        orderability_reasons: list[str] = []
        if max_affordable_shares < 1:
            orderability_status = "insufficient_capital"
            orderability_reasons.append("insufficient_capital")
        elif suggested_shares < 2:
            orderability_status = "constrained"
            orderability_reasons.append("weak_tradability")
        else:
            orderability_status = "affordable"
        if expected_net_edge <= 0:
            orderability_reasons.append("weak_tradability")

        timing_gate_status, timing_reasons = _timing_gate_status(timing_map.get(ticker))
        pricing_confidence = float(
            feature_row.get("cross_asset_aggregate_confirmation")
            or feature_row.get("pricing_discipline_market_confirmation")
            or 0.0
        )
        trade_readiness = float(feature_row.get("trade_readiness_timing_quality") or 0.0)
        downgrade_reasons = list(
            dict.fromkeys(
                (["stale_market_data"] if stale_market_data or price_is_stale else [])
                + timing_reasons
                + orderability_reasons
                + (["weak_pricing_confirmation"] if pricing_confidence <= 0 else [])
                + (
                    ["weak_monthly_model"]
                    if float(row["prediction"]) < 0.55 or not benchmark_gate_pass
                    else []
                )
            )
        )
        conviction = max(
            0.0,
            min(
                1.0,
                (0.55 * float(row["prediction"]))
                + (0.15 * float(timing_map.get(ticker, 0.0) or 0.0))
                + (0.15 * pricing_confidence)
                + (0.15 * trade_readiness),
            ),
        )
        orderability = OrderabilityState(
            max_affordable_shares=max_affordable_shares,
            estimated_cash_use=round(notional + entry_cost, 2),
            estimated_total_entry_cost=round(entry_cost, 2),
            estimated_total_exit_cost=round(exit_cost, 2),
            expected_net_edge_after_costs=round(expected_net_edge, 6),
            orderability_status=orderability_status,
            residual_cash=round(max(0.0, resolved_capital - notional - entry_cost), 2),
        )
        decision_mode = _decision_mode(
            conviction=conviction,
            pricing_confidence=pricing_confidence,
            timing_gate_status=timing_gate_status,
            orderability_status=orderability_status,
            downgrade_reasons=downgrade_reasons,
        )
        price_evidence_date = (
            pd.to_datetime(price_row.get("date")).date()
            if price_row.get("date") is not None
            else latest_prediction_date.date()
        )
        evidence_ids = [
            f"price::{ticker}::{price_evidence_date}",
            f"prediction::{ticker}::{target_horizon_days}d::{latest_prediction_date.date()}",
        ]
        top_themes = [
            str(item.get("theme"))
            for item in theme_intensities.head(2).to_dict(orient="records")
            if item.get("theme")
        ]
        candidate = ShortlistCandidate(
            ticker=ticker,
            exchange_code=str(
                price_row.get("exchange_code") or security_row.get("exchange_code") or "NSE"
            ),
            currency=str(price_row.get("currency") or security_row.get("currency") or "INR"),
            current_price=current_price,
            max_affordable_shares=max_affordable_shares,
            estimated_cash_use=round(notional + entry_cost, 2),
            monthly_rank_score=round(float(row["prediction"]), 6),
            timing_gate_status=timing_gate_status,
            decision_mode=decision_mode,
            conviction=round(conviction, 4),
            thesis_summary=(
                f"{ticker} ranks well on the {target_horizon_days}-day India model with "
                f"{security_row.get('sector', feature_row.get('sector', 'unknown'))} exposure."
            ),
            pricing_status=(
                "confirmed"
                if pricing_confidence >= 0.6
                else "weakly_confirmed"
                if pricing_confidence >= 0.4
                else "unconfirmed"
            ),
            falsification_triggers=[
                "monthly_rank_slips_below_threshold",
                "5d_timing_gate_deteriorates",
                "pricing_confirmation_remains_missing",
            ],
            evidence_ids=evidence_ids,
            fact_layer={
                "summary": (
                    f"{ticker} last closed at {round(current_price, 2)} {resolved_currency} on "
                    f"{price_evidence_date.isoformat()}."
                ),
                "sector": str(security_row.get("sector", feature_row.get("sector", "Unknown"))),
                "latest_price_date": price_evidence_date.isoformat(),
                "monthly_rank_score": round(float(row["prediction"]), 6),
                "evidence_ids": evidence_ids,
            },
            interpretation_layer={
                "summary": (
                    f"{ticker} is aligned with the active India monthly model and the current "
                    f"top themes {top_themes or ['unclassified']}."
                ),
                "top_themes": top_themes,
                "sector": str(security_row.get("sector", feature_row.get("sector", "Unknown"))),
                "benchmark_gate_passed": benchmark_gate_pass,
            },
            pricing_layer={
                "summary": (
                    "Pricing is confirmed enough for action."
                    if pricing_confidence >= 0.6
                    else "Pricing support is partial."
                    if pricing_confidence >= 0.4
                    else "Pricing support is weak or missing."
                ),
                "pricing_status": (
                    "confirmed"
                    if pricing_confidence >= 0.6
                    else "weakly_confirmed"
                    if pricing_confidence >= 0.4
                    else "unconfirmed"
                ),
                "pricing_confidence": round(pricing_confidence, 4),
                "trade_readiness": round(trade_readiness, 4),
                "timing_gate_status": timing_gate_status,
                "price_is_stale": price_is_stale,
            },
            decision_layer={
                "mode": decision_mode,
                "target_horizon_days": target_horizon_days,
                "conviction": round(conviction, 4),
                "downgrade_reasons": downgrade_reasons,
            },
            falsification_layer={
                "triggers": [
                    "monthly_rank_slips_below_threshold",
                    "5d_timing_gate_deteriorates",
                    "pricing_confirmation_remains_missing",
                ],
                "what_to_monitor_next": [
                    "fresh_nse_price_bar",
                    "pricing_confirmation",
                    "5d_timing_gate",
                ],
            },
            cost_tradability={
                "suggested_share_count": suggested_shares,
                **asdict(orderability),
                "expected_gross_edge": round(expected_gross_edge, 6),
            },
            downgrade_reasons=downgrade_reasons,
        )
        if decision_mode != "actionable_bullish":
            rejected.append(
                {
                    "ticker": ticker,
                    "decision_mode": decision_mode,
                    "reasons": downgrade_reasons or ["research_only"],
                }
            )
        candidates.append(candidate)

    candidates = sorted(
        candidates,
        key=lambda item: (
            item.decision_mode != "actionable_bullish",
            -item.monthly_rank_score,
            item.ticker,
        ),
    )[:max_candidates]
    primary_candidate = None
    if candidates:
        top = candidates[0]
        next_score = candidates[1].monthly_rank_score if len(candidates) > 1 else 0.0
        if (
            top.decision_mode == "actionable_bullish"
            and top.conviction >= 0.7
            and (top.monthly_rank_score - next_score) >= 0.10
            and not top.downgrade_reasons
            and benchmark_gate_pass
        ):
            primary_candidate = top

    benchmark_summary = {}
    if not benchmark_metrics.empty:
        latest_bench_date = pd.to_datetime(benchmark_metrics["as_of_date"]).max()
        latest_bench_mask = pd.to_datetime(benchmark_metrics["as_of_date"]) == latest_bench_date
        benchmark_summary = (
            benchmark_metrics.loc[latest_bench_mask]
            .sort_values(["strategy_name", "metric_name"])
            .to_dict(orient="records")
        )

    overall_confidence = candidates[0].conviction if candidates else 0.0
    pricing_confidence_value = 0.0
    if (
        not latest_features.empty
        and "cross_asset_aggregate_confirmation" in latest_features.columns
    ):
        pricing_confidence_value = float(
            pd.to_numeric(
                latest_features.get("cross_asset_aggregate_confirmation"),
                errors="coerce",
            )
            .fillna(0.0)
            .mean()
        )
    decision = ShortlistDecision(
        system_mode="shortlist",
        market_scope=resolved_scope,  # type: ignore[arg-type]
        capital=resolved_capital,
        base_currency=resolved_currency,
        target_horizon_days=target_horizon_days,
        market_summary={
            "latest_prediction_date": latest_prediction_date.date().isoformat(),
            "top_themes": theme_intensities.head(5).to_dict(orient="records"),
            "sector_rankings": sector_rankings.head(5).to_dict(orient="records"),
            "stale_market_data": stale_market_data,
            "benchmark_gate": benchmark_gate_summary,
            "benchmark_gate_passed": benchmark_gate_pass,
            "benchmark_context": benchmark_summary,
            "account_profile": asdict(account_profile),
        },
        shortlist=candidates,
        primary_candidate=primary_candidate,
        rejected_candidates=rejected[:10],
        downgrade_reasons=list(
            dict.fromkeys(
                (["stale_market_data"] if stale_market_data else [])
                + ([] if benchmark_gate_pass else ["weak_monthly_model"])
            )
        ),
        confidence_breakdown={
            "decision_confidence": round(overall_confidence, 4),
            "model_confidence": round(float(monthly_live["prediction"].max()), 4),
            "pricing_confidence": round(pricing_confidence_value, 4),
            "benchmark_gate_passed": benchmark_gate_pass,
        },
    )
    return asdict(decision)

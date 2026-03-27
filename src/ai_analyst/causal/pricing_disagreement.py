from __future__ import annotations

import pandas as pd

from ai_analyst.causal.types import CausalValue, PricingDisagreementState


def _status(confidence: float) -> str:
    if confidence >= 0.65:
        return "supported"
    if confidence >= 0.4:
        return "weakly_supported"
    return "conflicted"


def build_pricing_disagreement_state(
    *,
    theme_intensities: pd.DataFrame,
    sector_rankings: pd.DataFrame,
    prices: pd.DataFrame,
) -> PricingDisagreementState:
    geo_signal = (
        float(theme_intensities["intensity"].head(3).sum()) if not theme_intensities.empty else 0.0
    )
    sector_response = (
        float(sector_rankings["sector_score"].abs().head(5).mean())
        if not sector_rankings.empty
        else 0.0
    )
    market_response = 0.0
    crowdedness = 0.0
    follow_through = 0.0
    if not prices.empty and {"adj_close", "ticker"}.issubset(prices.columns):
        price_frame = prices.sort_values(["ticker", "date"]).copy()
        price_frame["ret_1d"] = price_frame.groupby("ticker")["adj_close"].pct_change()
        market_response = float(price_frame["ret_1d"].abs().tail(20).mean() or 0.0)
        crowdedness = float(price_frame["ret_1d"].tail(20).std() or 0.0)
        follow_through = float(price_frame["ret_1d"].tail(5).mean() or 0.0)

    divergence = max(0.0, geo_signal - sector_response - market_response)
    geo_conf = min(1.0, geo_signal / 5.0) if geo_signal else 0.0
    mediator_conf = min(1.0, max(0.0, (market_response + sector_response) / 2.0))
    market_conf = min(1.0, market_response * 5.0)
    divergence_conf = min(1.0, divergence / 3.0) if divergence else 0.2
    crowded_conf = min(1.0, crowdedness * 10.0)
    follow_conf = min(1.0, abs(follow_through) * 20.0)

    return PricingDisagreementState(
        geo_signal_strength=CausalValue(
            value=round(geo_signal, 4),
            confidence=round(geo_conf, 4),
            status=_status(geo_conf) if geo_signal else "missing",
        ),
        mediator_confirmation=CausalValue(
            value=round(sector_response + market_response, 4),
            confidence=round(mediator_conf, 4),
            status=_status(mediator_conf),
        ),
        market_response_strength=CausalValue(
            value=round(market_response, 4),
            confidence=round(market_conf, 4),
            status=_status(market_conf) if market_response else "missing",
        ),
        divergence_score=CausalValue(
            value=round(divergence, 4),
            confidence=round(divergence_conf, 4),
            status=_status(divergence_conf),
        ),
        crowdedness_proxy=CausalValue(
            value=round(crowdedness, 4),
            confidence=round(crowded_conf, 4),
            status=_status(crowded_conf) if crowdedness else "missing",
        ),
        follow_through_status=CausalValue(
            value=round(follow_through, 4),
            confidence=round(follow_conf, 4),
            status=_status(follow_conf) if follow_through else "missing",
        ),
    )

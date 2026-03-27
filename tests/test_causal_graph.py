from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.causal.causal_graph import CausalGraphEngine
from ai_analyst.causal.pricing_disagreement import build_pricing_disagreement_state


def test_causal_graph_builds_chains_and_state() -> None:
    as_of = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    events = pd.DataFrame(
        [
            {
                "event_id": "evt_red_sea",
                "event_time": as_of,
                "topic": "Chokepoint disruption in the Red Sea",
                "theme": "shipping_stress",
                "market_relevance": 0.9,
            }
        ]
    )
    theme_intensities = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-03"),
                "theme": "shipping_stress",
                "intensity": 2.4,
                "event_count": 2,
                "avg_severity": 0.8,
                "avg_novelty": 0.6,
            },
            {
                "date": pd.Timestamp("2024-06-03"),
                "theme": "oil_supply_risk",
                "intensity": 1.2,
                "event_count": 1,
                "avg_severity": 0.7,
                "avg_novelty": 0.5,
            },
        ]
    )
    sector_rankings = pd.DataFrame(
        [
            {"sector": "Energy", "sector_score": 1.2},
            {"sector": "Consumer Discretionary", "sector_score": -0.8},
            {"sector": "Industrials", "sector_score": -0.3},
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": "XOM",
                "date": pd.Timestamp("2024-06-01"),
                "adj_close": 100.0,
            },
            {
                "ticker": "XOM",
                "date": pd.Timestamp("2024-06-02"),
                "adj_close": 102.0,
            },
            {
                "ticker": "DAL",
                "date": pd.Timestamp("2024-06-01"),
                "adj_close": 45.0,
            },
            {
                "ticker": "DAL",
                "date": pd.Timestamp("2024-06-02"),
                "adj_close": 44.0,
            },
        ]
    )
    theme_regimes = pd.DataFrame(
        [{"theme": "shipping_stress", "regime_name": "escalating", "regime_score": 0.72}]
    )
    analogs = [
        {"horizon": "1-3 days", "analog_key": "theme_state:2024-01-15"},
        {"horizon": "1-3 weeks", "analog_key": "theme_state:2023-10-20"},
    ]

    pricing_state = build_pricing_disagreement_state(
        theme_intensities=theme_intensities,
        sector_rankings=sector_rankings,
        prices=prices,
    )
    engine = CausalGraphEngine()
    chains = engine.build_chains(
        events=events,
        theme_intensities=theme_intensities,
        sector_rankings=sector_rankings,
    )
    state = engine.build_state(
        as_of=as_of,
        theme_intensities=theme_intensities,
        sector_rankings=sector_rankings,
        macro=pd.DataFrame(),
        prices=prices,
        source_assessment_frame=pd.DataFrame(),
        narrative_risk_frame=pd.DataFrame(),
        pricing_disagreement=pricing_state,
        analog_matches=analogs,
        chains=chains,
        theme_regimes=theme_regimes,
    )

    assert chains
    assert any(chain.theme == "shipping_stress" for chain in chains)
    assert state.regime.label.value == "escalating"
    assert state.themes.active_themes
    assert state.pricing_disagreement.divergence_score.value >= 0.0
    assert len(state.horizon_views) == 3

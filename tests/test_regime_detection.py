from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ai_analyst.causal.regime_engine import build_theme_regimes


def test_hmm_regime_detector_fits_and_predicts_3_states():
    """HMM fits on synthetic 3-regime data and recovers states."""
    try:
        from ai_analyst.regime.hmm import HMMRegimeDetector
    except ImportError:
        pytest.skip("hmmlearn not installed")

    np.random.seed(42)
    n = 300
    # Bull: high return, low vol | Bear: low return, high vol | Neutral: mid
    returns = np.concatenate([
        np.random.normal(0.002, 0.005, n),   # bull
        np.random.normal(-0.003, 0.02, n),   # bear
        np.random.normal(0.0, 0.01, n),      # neutral
    ])
    vol = np.concatenate([
        np.full(n, 0.005),
        np.full(n, 0.02),
        np.full(n, 0.01),
    ])
    dates = pd.date_range("2020-01-01", periods=3 * n, freq="B")
    ret_series = pd.Series(returns, index=dates[:len(returns)], name="ret_1d")
    vol_series = pd.Series(vol, index=dates[:len(vol)], name="realized_vol_20d")

    detector = HMMRegimeDetector(n_states=3)
    detector.fit(ret_series, vol_series)
    result = detector.predict(ret_series, vol_series)

    # Should have 3 states labeled bull/bear/neutral
    assert set(result.state_labels.values()) == {"bull", "bear", "neutral"}
    assert len(result.states) == len(ret_series)
    assert result.state_probabilities.shape == (len(ret_series), 3)
    assert result.transition_matrix.shape == (3, 3)


def test_hmm_predict_latest():
    """predict_latest returns a label and probability dict."""
    try:
        from ai_analyst.regime.hmm import HMMRegimeDetector
    except ImportError:
        pytest.skip("hmmlearn not installed")

    np.random.seed(42)
    n = 200
    # Use varying volatility to avoid singular covariance
    returns = np.random.normal(0.001, 0.01, n)
    vol = np.abs(np.random.normal(0.01, 0.003, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    ret_series = pd.Series(returns, index=dates, name="ret_1d")
    vol_series = pd.Series(vol, index=dates, name="realized_vol_20d")

    detector = HMMRegimeDetector(n_states=3)
    detector.fit(ret_series, vol_series)
    label, probs = detector.predict_latest(ret_series, vol_series)

    assert label in {"bull", "bear", "neutral"}
    assert set(probs.keys()) == {"bull", "bear", "neutral"}
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_changepoint_detector_finds_breaks():
    """ChangepointDetector detects a step-function break."""
    try:
        from ai_analyst.regime.changepoint import ChangepointDetector
    except ImportError:
        pytest.skip("ruptures not installed")

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    # Step function: low regime then high regime
    values = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    series = pd.Series(values, index=dates)

    detector = ChangepointDetector(penalty=3.0)
    breaks = detector.detect(series)

    # Should detect at least one break near the 100-point boundary
    assert len(breaks) >= 1


def test_build_theme_regimes_backward_compatible():
    """build_theme_regimes still works with no HMM or changepoint args."""
    theme_daily = pd.DataFrame(
        [
            {"date": "2024-06-01", "theme": "oil_supply_risk", "intensity": 3.0},
            {"date": "2024-06-01", "theme": "defense_escalation", "intensity": 1.5},
            {"date": "2024-06-02", "theme": "oil_supply_risk", "intensity": 0.5},
        ]
    )
    result = build_theme_regimes(theme_daily)
    assert not result.empty
    assert "regime_name" in result.columns
    assert "regime_score" in result.columns
    # First day: total intensity = 4.5 >= 2.5 → escalating
    row1 = result[result["as_of_date"] == date(2024, 6, 1)]
    assert row1.iloc[0]["regime_name"] == "escalating"
    # Second day: total < 2.5, energy theme → energy_supply_stress
    row2 = result[result["as_of_date"] == date(2024, 6, 2)]
    assert row2.iloc[0]["regime_name"] == "energy_supply_stress"


def test_build_theme_regimes_with_hmm_augmentation():
    """When HMM detector is passed, HMM columns are added."""
    try:
        from ai_analyst.regime.hmm import HMMRegimeDetector
    except ImportError:
        pytest.skip("hmmlearn not installed")

    np.random.seed(42)
    n = 300
    # Create clearly separable 3-regime synthetic data
    dates = pd.bdate_range("2020-01-01", periods=3 * n)
    returns = np.concatenate([
        np.random.normal(0.002, 0.005, n),   # bull
        np.random.normal(-0.003, 0.02, n),   # bear
        np.random.normal(0.0, 0.01, n),      # neutral
    ])
    vol = np.concatenate([
        np.abs(np.random.normal(0.005, 0.001, n)),
        np.abs(np.random.normal(0.02, 0.003, n)),
        np.abs(np.random.normal(0.01, 0.002, n)),
    ])

    # Theme data must match dates
    theme_daily = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "theme": ["oil_supply_risk"] * len(dates),
            "intensity": np.random.uniform(0.5, 2.0, len(dates)),
        }
    )

    prices = pd.DataFrame(
        {"ret_1d": returns, "realized_vol_20d": vol},
        index=dates,
    )

    detector = HMMRegimeDetector(n_states=3)
    detector.fit(prices["ret_1d"], prices["realized_vol_20d"])

    result = build_theme_regimes(
        theme_daily,
        hmm_detector=detector,
        prices=prices,
    )
    assert not result.empty
    assert "hmm_label" in result.columns
    assert "changepoint_flag" in result.columns

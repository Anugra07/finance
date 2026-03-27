from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RegimeDetectionResult:
    """Result of HMM regime detection."""

    states: pd.Series  # integer state per date
    state_probabilities: pd.DataFrame  # date x n_states probability matrix
    transition_matrix: np.ndarray
    state_labels: dict[int, str]  # mapped to "bull"/"bear"/"neutral"


class HMMRegimeDetector:
    """Hidden Markov Model regime detector using returns and volatility observations.

    Fits a GaussianHMM with n_states (default 3: bull/bear/neutral) on a 2D
    observation matrix of [log_returns, realized_volatility].

    After fitting, states are labelled by their mean return:
    highest mean = "bull", lowest = "bear", middle = "neutral".
    """

    def __init__(self, n_states: int = 3, covariance_type: str = "full") -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self._model = None
        self._state_labels: dict[int, str] = {}

    def _build_observations(
        self, returns: pd.Series, volatility: pd.Series
    ) -> np.ndarray:
        df = pd.DataFrame({"ret": returns, "vol": volatility}).dropna()
        return df.values

    def _label_states(self) -> dict[int, str]:
        """Assign bull/bear/neutral labels based on state mean returns."""
        means = self._model.means_[:, 0]  # first column = returns
        sorted_indices = np.argsort(means)
        labels = {}
        label_names = ["bear", "neutral", "bull"]
        if self.n_states == 2:
            label_names = ["bear", "bull"]
        elif self.n_states > 3:
            label_names = [f"state_{i}" for i in range(self.n_states)]
            label_names[sorted_indices[0]] = "bear"
            label_names[sorted_indices[-1]] = "bull"
            return {int(i): label_names[i] for i in range(self.n_states)}

        for rank, state_idx in enumerate(sorted_indices):
            labels[int(state_idx)] = label_names[rank]
        return labels

    def fit(self, returns: pd.Series, volatility: pd.Series) -> HMMRegimeDetector:
        """Fit GaussianHMM on [returns, volatility] observation matrix."""
        from hmmlearn.hmm import GaussianHMM

        obs = self._build_observations(returns, volatility)
        if len(obs) < 50:
            raise ValueError(f"Need at least 50 observations, got {len(obs)}")

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self._model.fit(obs)
        self._state_labels = self._label_states()
        return self

    def predict(
        self, returns: pd.Series, volatility: pd.Series
    ) -> RegimeDetectionResult:
        """Decode most likely state sequence and return probabilities."""
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        df = pd.DataFrame({"ret": returns, "vol": volatility}).dropna()
        obs = df.values
        states = self._model.predict(obs)
        probs = self._model.predict_proba(obs)

        state_series = pd.Series(states, index=df.index, name="hmm_state")
        prob_df = pd.DataFrame(
            probs,
            index=df.index,
            columns=[self._state_labels.get(i, f"state_{i}") for i in range(self.n_states)],
        )

        return RegimeDetectionResult(
            states=state_series,
            state_probabilities=prob_df,
            transition_matrix=self._model.transmat_,
            state_labels=self._state_labels,
        )

    def predict_latest(
        self, returns: pd.Series, volatility: pd.Series
    ) -> tuple[str, dict[str, float]]:
        """Return current regime label and probability distribution."""
        result = self.predict(returns, volatility)
        latest_state = int(result.states.iloc[-1])
        label = self._state_labels.get(latest_state, "unknown")
        probs = {col: float(result.state_probabilities[col].iloc[-1]) for col in result.state_probabilities.columns}
        return label, probs

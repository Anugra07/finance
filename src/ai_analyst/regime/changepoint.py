from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


class ChangepointDetector:
    """Structural break detection using the ruptures PELT algorithm.

    Uses the RBF kernel model with a configurable penalty to detect
    points where the statistical properties of a time series change.
    """

    def __init__(
        self,
        model: str = "rbf",
        min_size: int = 20,
        penalty: float = 3.0,
    ) -> None:
        self.model = model
        self.min_size = min_size
        self.penalty = penalty

    def detect(self, series: pd.Series) -> list[date]:
        """Return dates where structural breaks are detected."""
        import ruptures

        clean = series.dropna()
        if len(clean) < self.min_size * 2:
            return []

        signal = clean.values.astype(float)
        algo = ruptures.Pelt(model=self.model, min_size=self.min_size).fit(signal)
        breakpoints = algo.predict(pen=self.penalty)

        # ruptures returns indices (1-based, with the last being len(signal))
        # Convert to dates, excluding the terminal point
        dates: list[date] = []
        for bp in breakpoints:
            if bp >= len(clean):
                continue
            idx = clean.index[bp]
            if isinstance(idx, pd.Timestamp):
                dates.append(idx.date())
            elif isinstance(idx, date):
                dates.append(idx)
        return dates

    def detect_multi(
        self, df: pd.DataFrame, columns: list[str]
    ) -> dict[str, list[date]]:
        """Detect changepoints across multiple series."""
        results: dict[str, list[date]] = {}
        for col in columns:
            if col in df.columns:
                results[col] = self.detect(df[col])
        return results

    def recent_changepoint_flag(
        self, series: pd.Series, lookback_days: int = 20
    ) -> bool:
        """Return True if a changepoint was detected in the last N observations."""
        breakpoints = self.detect(series)
        if not breakpoints:
            return False
        clean = series.dropna()
        if clean.empty:
            return False
        latest_date = clean.index[-1]
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.date()
        for bp_date in breakpoints:
            if (latest_date - bp_date).days <= lookback_days:
                return True
        return False

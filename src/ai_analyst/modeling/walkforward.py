from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class WalkForwardSpec:
    train_years: int = 3
    validation_years: int = 1
    test_months: int = 3
    step_months: int = 3
    embargo_days: int = 5
    label_horizon_days: int = 5


@dataclass(slots=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: pd.Series
    validation_mask: pd.Series
    test_mask: pd.Series


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    if "date" not in frame.columns:
        raise ValueError("Walk-forward splits require a 'date' column.")
    dates = pd.to_datetime(frame["date"], utc=False).dt.normalize()
    if dates.isna().all():
        raise ValueError("Walk-forward splits require at least one valid date.")
    return dates


def generate_walk_forward_splits(
    frame: pd.DataFrame,
    spec: WalkForwardSpec,
) -> list[WalkForwardSplit]:
    if frame.empty:
        return []

    dates = _normalize_dates(frame)
    min_date = dates.min()
    max_date = dates.max()
    if pd.isna(min_date) or pd.isna(max_date):
        return []

    train_window = pd.DateOffset(years=spec.train_years)
    validation_window = pd.DateOffset(years=spec.validation_years)
    test_window = pd.DateOffset(months=spec.test_months)
    step_window = pd.DateOffset(months=spec.step_months)
    embargo = pd.Timedelta(days=spec.embargo_days)

    first_test_start = min_date + train_window + validation_window + (embargo * 2)
    if first_test_start > max_date:
        return []

    splits: list[WalkForwardSplit] = []
    test_start = first_test_start

    while test_start <= max_date:
        validation_end = test_start - embargo - pd.Timedelta(days=1)
        validation_start = (validation_end - validation_window) + pd.Timedelta(days=1)
        train_end = validation_start - embargo - pd.Timedelta(days=1)
        train_start = min_date
        test_end = min(test_start + test_window - pd.Timedelta(days=1), max_date)

        if train_end <= train_start or validation_end < validation_start or test_end < test_start:
            break

        train_mask = (dates >= train_start) & (dates <= train_end)
        validation_mask = (dates >= validation_start) & (dates <= validation_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        if train_mask.any() and validation_mask.any() and test_mask.any():
            splits.append(
                WalkForwardSplit(
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_mask=train_mask,
                    validation_mask=validation_mask,
                    test_mask=test_mask,
                )
            )

        test_start = test_start + step_window

    return splits

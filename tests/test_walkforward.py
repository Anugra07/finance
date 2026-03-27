from __future__ import annotations

import pandas as pd

from ai_analyst.modeling.walkforward import WalkForwardSpec, generate_walk_forward_splits


def _make_df(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({"date": dates, "ticker": "AAPL"})


class TestWalkForwardSpec:
    def test_defaults(self) -> None:
        spec = WalkForwardSpec()
        assert spec.train_years == 3
        assert spec.validation_years == 1
        assert spec.test_months == 3
        assert spec.step_months == 3
        assert spec.embargo_days == 5
        assert spec.label_horizon_days == 5

    def test_custom_values(self) -> None:
        spec = WalkForwardSpec(train_years=2, test_months=6, step_months=6)
        assert spec.train_years == 2
        assert spec.test_months == 6
        assert spec.step_months == 6


class TestGenerateWalkForwardSplits:
    def test_empty_dataframe_returns_no_splits(self) -> None:
        df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        splits = generate_walk_forward_splits(df, WalkForwardSpec())
        assert splits == []

    def test_insufficient_history_returns_no_splits(self) -> None:
        # 2 years of data, but spec requires 3 train + 1 val + embargo + test
        df = _make_df("2022-01-01", "2023-12-31")
        splits = generate_walk_forward_splits(df, WalkForwardSpec())
        assert splits == []

    def test_sufficient_history_produces_splits(self) -> None:
        # 6 years of data — enough for 3 train + 1 val + embargo + 3 month test
        df = _make_df("2018-01-01", "2024-06-30")
        spec = WalkForwardSpec(train_years=3, validation_years=1, test_months=3)
        splits = generate_walk_forward_splits(df, spec)
        assert len(splits) >= 1

    def test_masks_do_not_overlap(self) -> None:
        df = _make_df("2015-01-01", "2024-12-31")
        spec = WalkForwardSpec(train_years=3, validation_years=1, test_months=3, step_months=6)
        splits = generate_walk_forward_splits(df, spec)
        assert len(splits) >= 1

        for split in splits:
            train_idx = set(df.index[split.train_mask])
            val_idx = set(df.index[split.validation_mask])
            test_idx = set(df.index[split.test_mask])
            assert train_idx & val_idx == set(), "Train and validation overlap"
            assert train_idx & test_idx == set(), "Train and test overlap"
            assert val_idx & test_idx == set(), "Validation and test overlap"

    def test_splits_are_chronologically_ordered(self) -> None:
        df = _make_df("2015-01-01", "2024-12-31")
        spec = WalkForwardSpec(train_years=3, validation_years=1, test_months=3)
        splits = generate_walk_forward_splits(df, spec)
        assert len(splits) >= 1

        for split in splits:
            assert split.train_start <= split.train_end
            assert split.train_end < split.validation_start
            assert split.validation_start <= split.validation_end
            assert split.validation_end < split.test_start
            assert split.test_start <= split.test_end

    def test_embargo_creates_gap_between_windows(self) -> None:
        df = _make_df("2015-01-01", "2024-12-31")
        spec = WalkForwardSpec(embargo_days=10)
        splits = generate_walk_forward_splits(df, spec)
        assert len(splits) >= 1

        for split in splits:
            # Train end + embargo < validation start
            gap_train_val = (split.validation_start - split.train_end).days
            assert gap_train_val >= spec.embargo_days
            # Validation end + embargo < test start
            gap_val_test = (split.test_start - split.validation_end).days
            assert gap_val_test >= spec.embargo_days

    def test_step_advances_cursor(self) -> None:
        df = _make_df("2015-01-01", "2024-12-31")
        spec = WalkForwardSpec(train_years=3, validation_years=1, test_months=3, step_months=3)
        splits = generate_walk_forward_splits(df, spec)
        assert len(splits) >= 2

        # Each subsequent split should have a later test_start
        for i in range(1, len(splits)):
            assert splits[i].test_start > splits[i - 1].test_start

    def test_all_masks_cover_some_rows(self) -> None:
        df = _make_df("2015-01-01", "2024-12-31")
        spec = WalkForwardSpec(train_years=3, validation_years=1, test_months=3)
        splits = generate_walk_forward_splits(df, spec)

        for split in splits:
            assert split.train_mask.sum() > 0
            assert split.validation_mask.sum() > 0
            assert split.test_mask.sum() > 0

    def test_expanding_train_window(self) -> None:
        df = _make_df("2015-01-01", "2024-12-31")
        spec = WalkForwardSpec(train_years=3, validation_years=1, test_months=3, step_months=3)
        splits = generate_walk_forward_splits(df, spec)
        if len(splits) >= 2:
            # Train window should expand (or stay same) since train_start is always min_date
            assert splits[0].train_start == splits[1].train_start
            assert splits[1].train_mask.sum() >= splits[0].train_mask.sum()

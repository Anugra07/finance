"""Operational monitoring utilities for the warehouse and model pipeline.

Provides two key capabilities:

1. **Freshness monitor** — checks how stale each data domain is against
   configurable SLA budgets (e.g. prices must be < 2 days old).

2. **Feature drift detection** — uses the Kolmogorov–Smirnov test to flag
   features whose distribution has shifted between train and test windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.warehouse.database import connect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Freshness Monitor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FreshnessBudget:
    """Maximum acceptable age for a data domain."""

    domain: str
    query: str
    max_staleness: timedelta


DEFAULT_BUDGETS: list[FreshnessBudget] = [
    FreshnessBudget(
        domain="prices",
        query="SELECT MAX(known_at) FROM prices",
        max_staleness=timedelta(days=2),
    ),
    FreshnessBudget(
        domain="macro_vintages",
        query="SELECT MAX(known_at) FROM macro_vintages",
        max_staleness=timedelta(days=7),
    ),
    FreshnessBudget(
        domain="normalized_events",
        query="SELECT MAX(event_time) FROM normalized_events",
        max_staleness=timedelta(hours=24),
    ),
    FreshnessBudget(
        domain="universe_membership",
        query="SELECT MAX(snapshot_at) FROM universe_membership",
        max_staleness=timedelta(days=30),
    ),
    FreshnessBudget(
        domain="theme_intensity_daily",
        query="SELECT MAX(date) FROM theme_intensity_daily",
        max_staleness=timedelta(days=3),
    ),
    FreshnessBudget(
        domain="sector_rankings",
        query="SELECT MAX(as_of_date) FROM sector_rankings",
        max_staleness=timedelta(days=3),
    ),
]


@dataclass(slots=True)
class FreshnessResult:
    domain: str
    latest_value: datetime | None
    age: timedelta | None
    max_staleness: timedelta
    is_stale: bool
    exists: bool


def check_freshness(
    settings: Settings,
    *,
    as_of: datetime | None = None,
    budgets: list[FreshnessBudget] | None = None,
) -> list[FreshnessResult]:
    """Check data freshness against configured SLA budgets.

    Returns one ``FreshnessResult`` per domain indicating whether data is
    within the acceptable staleness window.
    """
    as_of = as_of or datetime.now(tz=UTC)
    budgets = budgets or DEFAULT_BUDGETS
    results: list[FreshnessResult] = []
    conn = connect(settings)
    try:
        for budget in budgets:
            try:
                row = conn.execute(budget.query).fetchone()
                if row is None or row[0] is None:
                    results.append(
                        FreshnessResult(
                            domain=budget.domain,
                            latest_value=None,
                            age=None,
                            max_staleness=budget.max_staleness,
                            is_stale=True,
                            exists=False,
                        )
                    )
                    continue
                latest = pd.to_datetime(row[0], utc=True)
                latest_dt = latest.to_pydatetime()
                age = as_of - latest_dt
                results.append(
                    FreshnessResult(
                        domain=budget.domain,
                        latest_value=latest_dt,
                        age=age,
                        max_staleness=budget.max_staleness,
                        is_stale=age > budget.max_staleness,
                        exists=True,
                    )
                )
            except Exception as exc:
                logger.warning("Freshness check failed for %s: %s", budget.domain, exc)
                results.append(
                    FreshnessResult(
                        domain=budget.domain,
                        latest_value=None,
                        age=None,
                        max_staleness=budget.max_staleness,
                        is_stale=True,
                        exists=False,
                    )
                )
    finally:
        conn.close()
    return results


def freshness_summary(results: list[FreshnessResult]) -> dict[str, object]:
    """Return a JSON-serializable freshness report."""
    domains: list[dict[str, object]] = []
    for r in results:
        domains.append(
            {
                "domain": r.domain,
                "latest": r.latest_value.isoformat() if r.latest_value else None,
                "age_hours": round(r.age.total_seconds() / 3600, 1) if r.age else None,
                "max_staleness_hours": round(r.max_staleness.total_seconds() / 3600, 1),
                "status": "ok" if not r.is_stale else ("missing" if not r.exists else "stale"),
            }
        )
    stale_count = sum(1 for r in results if r.is_stale)
    return {
        "overall_status": "healthy" if stale_count == 0 else "degraded",
        "stale_domains": stale_count,
        "total_domains": len(results),
        "domains": domains,
    }


# ---------------------------------------------------------------------------
# 2. Feature Drift Detection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DriftResult:
    feature: str
    ks_statistic: float
    p_value: float
    is_drifted: bool


def detect_feature_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    *,
    threshold: float = 0.05,
) -> list[DriftResult]:
    """Run KS test between train and test distributions for each feature.

    Features where the p-value falls below ``threshold`` are flagged as drifted.
    Requires ``scipy`` — returns an empty list if not installed.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        logger.warning("scipy not installed — skipping drift detection.")
        return []

    results: list[DriftResult] = []
    for feat in features:
        if feat not in train_df.columns or feat not in test_df.columns:
            continue
        train_values = train_df[feat].dropna()
        test_values = test_df[feat].dropna()
        if train_values.empty or test_values.empty:
            continue
        stat, pval = ks_2samp(train_values, test_values)
        results.append(
            DriftResult(
                feature=feat,
                ks_statistic=round(float(stat), 6),
                p_value=round(float(pval), 6),
                is_drifted=pval < threshold,
            )
        )
    return results


def drift_summary(results: list[DriftResult]) -> dict[str, object]:
    """Return a JSON-serializable drift report."""
    drifted = [r for r in results if r.is_drifted]
    return {
        "total_features_checked": len(results),
        "drifted_features": len(drifted),
        "drifted": [
            {
                "feature": r.feature,
                "ks_statistic": r.ks_statistic,
                "p_value": r.p_value,
            }
            for r in drifted
        ],
    }

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.paper_trading.ledger import TradeLedger
from ai_analyst.portfolio.allocator import build_rebalance_plan
from ai_analyst.warehouse.database import connect

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PaperTradeSignal:
    signal_id: str
    timestamp: datetime
    ticker: str
    direction: str  # "long" | "short" | "flat"
    confidence: float
    target_weight: float
    regime: str
    source: str  # "model" | "llm" | "combined"


@dataclass(slots=True)
class PaperTradeExecution:
    execution_id: str
    signal_id: str
    ticker: str
    direction: str
    entry_price: float
    entry_date: date
    exit_price: float | None = None
    exit_date: date | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    holding_days: int | None = None
    status: str = "open"
    regime: str = "unknown"


def _fetch_close_price(settings: Settings, ticker: str, as_of_date: date) -> float | None:
    """Fetch the most recent close price on or before as_of_date."""
    conn = connect(settings)
    try:
        row = conn.execute(
            """
            SELECT adj_close
            FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
            """,
            [ticker, as_of_date],
        ).fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()


def _get_current_regime(settings: Settings, as_of_date: date) -> str:
    """Get the latest theme regime label."""
    conn = connect(settings)
    try:
        row = conn.execute(
            """
            SELECT regime_name
            FROM theme_regimes
            WHERE as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            [as_of_date],
        ).fetchone()
        return str(row[0]) if row else "unknown"
    except Exception:
        return "unknown"
    finally:
        conn.close()


class PaperTradingEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ledger = TradeLedger(settings)

    def generate_signals(self, as_of: datetime) -> list[PaperTradeSignal]:
        """Generate trade signals from the latest rebalance plan."""
        plan = build_rebalance_plan(self.settings)
        if plan.get("status") != "ok":
            logger.info("No actionable rebalance plan: %s", plan.get("status"))
            return []

        regime = _get_current_regime(self.settings, as_of.date())
        allocations = plan.get("allocations", [])
        max_positions = self.settings.paper_trade_max_positions

        signals: list[PaperTradeSignal] = []
        for alloc in allocations[:max_positions]:
            weight = float(alloc.get("target_weight", 0))
            if weight <= 0:
                continue
            signals.append(
                PaperTradeSignal(
                    signal_id=uuid.uuid4().hex[:12],
                    timestamp=as_of,
                    ticker=str(alloc["ticker"]),
                    direction="long",
                    confidence=float(alloc.get("score", 0)),
                    target_weight=weight,
                    regime=regime,
                    source="model",
                )
            )
        return signals

    def execute_signals(
        self, signals: list[PaperTradeSignal], as_of: datetime
    ) -> list[PaperTradeExecution]:
        """Record entries at close price for each signal."""
        executions: list[PaperTradeExecution] = []
        for signal in signals:
            price = _fetch_close_price(self.settings, signal.ticker, as_of.date())
            if price is None:
                logger.warning("No price for %s on %s, skipping.", signal.ticker, as_of.date())
                continue
            exec_ = PaperTradeExecution(
                execution_id=uuid.uuid4().hex[:12],
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                direction=signal.direction,
                entry_price=price,
                entry_date=as_of.date(),
                regime=signal.regime,
            )
            executions.append(exec_)
        return executions

    def close_expired_positions(
        self, as_of: datetime, horizon_days: int | None = None
    ) -> list[PaperTradeExecution]:
        """Close positions that have exceeded the holding period."""
        horizon = horizon_days or self.settings.paper_trade_horizon_days
        open_positions = self.ledger.load_open_positions()
        if open_positions.empty:
            return []

        closed: list[PaperTradeExecution] = []
        for _, row in open_positions.iterrows():
            entry = pd.Timestamp(row["entry_date"]).date()
            if (as_of.date() - entry).days < horizon:
                continue
            exit_price = _fetch_close_price(self.settings, str(row["ticker"]), as_of.date())
            if exit_price is None:
                continue
            entry_price = float(row["entry_price"])
            direction = str(row["direction"])
            pnl = exit_price - entry_price if direction == "long" else entry_price - exit_price
            pnl_pct = pnl / entry_price if entry_price != 0 else 0.0
            closed.append(
                PaperTradeExecution(
                    execution_id=str(row["execution_id"]),
                    signal_id=str(row["signal_id"]),
                    ticker=str(row["ticker"]),
                    direction=direction,
                    entry_price=entry_price,
                    entry_date=entry,
                    exit_price=exit_price,
                    exit_date=as_of.date(),
                    pnl=round(pnl, 4),
                    pnl_pct=round(pnl_pct, 6),
                    holding_days=(as_of.date() - entry).days,
                    status="closed",
                    regime=str(row.get("regime", "unknown")),
                )
            )
        return closed

    def run_day(self, as_of: datetime) -> dict[str, Any]:
        """Orchestrate a full daily paper-trading cycle."""
        # 1. Close expired positions
        closed = self.close_expired_positions(as_of)
        for exec_ in closed:
            self.ledger.record_execution(exec_)

        # 2. Generate signals
        signals = self.generate_signals(as_of)
        for signal in signals:
            self.ledger.record_signal(signal)

        # 3. Execute signals
        executions = self.execute_signals(signals, as_of)
        for exec_ in executions:
            self.ledger.record_execution(exec_)

        return {
            "as_of": as_of.isoformat(),
            "signals_generated": len(signals),
            "positions_opened": len(executions),
            "positions_closed": len(closed),
            "closed_pnl": sum(e.pnl or 0 for e in closed),
        }

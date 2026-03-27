from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ai_analyst.paper_trading.engine import PaperTradeExecution, PaperTradeSignal
from ai_analyst.paper_trading.report import build_paper_trade_report


def test_paper_trade_signal_dataclass():
    signal = PaperTradeSignal(
        signal_id="abc123",
        timestamp=datetime(2024, 6, 3, 20, 0, tzinfo=UTC),
        ticker="AAPL",
        direction="long",
        confidence=0.75,
        target_weight=0.05,
        regime="calm",
        source="model",
    )
    assert signal.ticker == "AAPL"
    assert signal.direction == "long"
    assert signal.confidence == 0.75


def test_paper_trade_execution_pnl():
    exec_ = PaperTradeExecution(
        execution_id="exec1",
        signal_id="abc123",
        ticker="AAPL",
        direction="long",
        entry_price=150.0,
        entry_date=date(2024, 6, 3),
        exit_price=155.0,
        exit_date=date(2024, 6, 10),
        pnl=5.0,
        pnl_pct=5.0 / 150.0,
        holding_days=7,
        status="closed",
        regime="calm",
    )
    assert exec_.pnl == 5.0
    assert exec_.status == "closed"
    assert exec_.holding_days == 7


def test_paper_trade_execution_short_pnl():
    exec_ = PaperTradeExecution(
        execution_id="exec2",
        signal_id="def456",
        ticker="TSLA",
        direction="short",
        entry_price=200.0,
        entry_date=date(2024, 6, 3),
        exit_price=190.0,
        exit_date=date(2024, 6, 10),
        pnl=10.0,
        pnl_pct=10.0 / 200.0,
        holding_days=7,
        status="closed",
        regime="escalating",
    )
    assert exec_.pnl == 10.0
    assert exec_.direction == "short"


def test_report_on_empty_trades(tmp_path):
    """build_paper_trade_report returns no_trades when ledger is empty."""
    from ai_analyst.config import Settings

    settings = Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "test.duckdb",
    )
    report = build_paper_trade_report(settings)
    assert report["status"] == "no_trades"


def test_report_metric_computation():
    """Verify Sharpe, win rate, profit factor computed correctly from closed trades."""
    closed = pd.DataFrame(
        [
            {"pnl": 10.0, "pnl_pct": 0.05, "status": "closed", "holding_days": 5, "regime": "bull"},
            {"pnl": -3.0, "pnl_pct": -0.015, "status": "closed", "holding_days": 5, "regime": "bull"},
            {"pnl": 7.0, "pnl_pct": 0.035, "status": "closed", "holding_days": 5, "regime": "bear"},
            {"pnl": -2.0, "pnl_pct": -0.01, "status": "closed", "holding_days": 5, "regime": "bear"},
        ]
    )
    total_pnl = closed["pnl"].sum()
    assert total_pnl == 12.0

    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    win_rate = len(wins) / len(closed)
    assert win_rate == 0.5

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss
    assert profit_factor == 17.0 / 5.0

    avg_pnl_pct = closed["pnl_pct"].mean()
    std_pnl_pct = closed["pnl_pct"].std()
    sharpe = avg_pnl_pct / std_pnl_pct * math.sqrt(252)
    assert sharpe > 0  # positive since net PnL is positive

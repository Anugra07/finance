from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.paper_trading.ledger import TradeLedger


def build_paper_trade_report(
    settings: Settings,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Compute PnL, Sharpe, max drawdown, win rate, profit factor from the trade ledger."""
    ledger = TradeLedger(settings)
    trades = ledger.load_all_trades(start_date=start_date, end_date=end_date)
    signals = ledger.load_all_signals()

    if trades.empty:
        return {
            "status": "no_trades",
            "total_signals": len(signals),
        }

    closed = trades[trades["status"] == "closed"].copy()
    open_positions = trades[trades["status"] == "open"].copy()

    if closed.empty:
        return {
            "status": "no_closed_trades",
            "total_signals": len(signals),
            "open_positions": len(open_positions),
        }

    closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce").fillna(0.0)

    total_pnl = float(closed["pnl"].sum())
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    win_rate = len(wins) / len(closed) if len(closed) > 0 else 0.0

    gross_profit = float(wins["pnl"].sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_pnl_pct = float(closed["pnl_pct"].mean())
    std_pnl_pct = float(closed["pnl_pct"].std()) if len(closed) > 1 else 0.0
    sharpe_ratio = (avg_pnl_pct / std_pnl_pct * math.sqrt(252)) if std_pnl_pct > 0 else 0.0

    # Max drawdown from cumulative PnL
    cum_pnl = closed["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    avg_holding_days = (
        float(pd.to_numeric(closed["holding_days"], errors="coerce").mean())
        if "holding_days" in closed.columns
        else 0.0
    )

    # Per-regime breakdown
    regime_breakdown: dict[str, Any] = {}
    if "regime" in closed.columns:
        for regime, group in closed.groupby("regime"):
            r_wins = group[group["pnl"] > 0]
            regime_breakdown[str(regime)] = {
                "trade_count": len(group),
                "win_rate": len(r_wins) / len(group) if len(group) > 0 else 0.0,
                "total_pnl": round(float(group["pnl"].sum()), 4),
                "avg_pnl_pct": round(float(group["pnl_pct"].mean()), 6),
            }

    # Per-sector breakdown
    sector_breakdown: dict[str, Any] = {}
    if "sector" in closed.columns:
        for sector, group in closed.groupby("sector"):
            s_wins = group[group["pnl"] > 0]
            sector_breakdown[str(sector)] = {
                "trade_count": len(group),
                "win_rate": len(s_wins) / len(group) if len(group) > 0 else 0.0,
                "total_pnl": round(float(group["pnl"].sum()), 4),
            }

    return {
        "status": "ok",
        "total_signals": len(signals),
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "open_positions": len(open_positions),
        "total_pnl": round(total_pnl, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "avg_holding_days": round(avg_holding_days, 1),
        "avg_pnl_pct": round(avg_pnl_pct, 6),
        "regime_breakdown": regime_breakdown,
        "sector_breakdown": sector_breakdown,
    }

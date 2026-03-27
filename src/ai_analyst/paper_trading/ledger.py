from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.utils.io import write_parquet
from ai_analyst.warehouse.layout import warehouse_partition_path

if TYPE_CHECKING:
    from ai_analyst.paper_trading.engine import PaperTradeExecution, PaperTradeSignal


class TradeLedger:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def record_signal(self, signal: PaperTradeSignal) -> Path:
        row = asdict(signal)
        row["transform_loaded_at"] = datetime.now(tz=UTC)
        df = pd.DataFrame([row])
        partition_date = signal.timestamp.date()
        return write_parquet(
            df,
            warehouse_partition_path(
                self.settings,
                domain="paper_trading/signals",
                partition_date=partition_date,
                stem=f"signal_{signal.signal_id}",
            ),
        )

    def record_execution(self, execution: PaperTradeExecution) -> Path:
        row = asdict(execution)
        row["signal_timestamp"] = row.pop("signal_id")  # keep reference
        row["signal_id"] = execution.signal_id
        row["transform_loaded_at"] = datetime.now(tz=UTC)
        df = pd.DataFrame([row])
        partition_date = execution.entry_date
        return write_parquet(
            df,
            warehouse_partition_path(
                self.settings,
                domain="paper_trading/executions",
                partition_date=partition_date,
                stem=f"exec_{execution.execution_id}",
            ),
        )

    def load_open_positions(self) -> pd.DataFrame:
        """Load executions that are still open (status='open')."""
        root = self.settings.warehouse_root / "paper_trading" / "executions"
        if not root.exists():
            return pd.DataFrame()
        files = sorted(root.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "status" not in df.columns:
            return pd.DataFrame()
        return df[df["status"] == "open"].copy()

    def load_all_trades(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load all executions, optionally filtered by date range."""
        root = self.settings.warehouse_root / "paper_trading" / "executions"
        if not root.exists():
            return pd.DataFrame()
        files = sorted(root.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if start_date and "entry_date" in df.columns:
            df = df[df["entry_date"] >= start_date]
        if end_date and "entry_date" in df.columns:
            df = df[df["entry_date"] <= end_date]
        return df

    def load_all_signals(self) -> pd.DataFrame:
        """Load all signals."""
        root = self.settings.warehouse_root / "paper_trading" / "signals"
        if not root.exists():
            return pd.DataFrame()
        files = sorted(root.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

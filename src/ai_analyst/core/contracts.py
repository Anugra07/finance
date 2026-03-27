from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Protocol

import pandas as pd


class PriceSource(Protocol):
    def fetch_history(
        self,
        ticker: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return daily bars and corporate actions for a ticker."""


class MacroContextSource(Protocol):
    def get_theme_intensities(self, *, as_of: datetime) -> pd.DataFrame:
        """Return theme intensities for a point in time."""

    def get_event_feed(
        self,
        *,
        ticker: str,
        as_of: datetime,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """Return normalized event context for a ticker and time."""


class AnalogStore(Protocol):
    def upsert_documents(self, rows: Iterable[dict[str, object]]) -> None:
        """Persist embeddings or analog documents."""

    def search(self, query_embedding: list[float], *, limit: int = 5) -> list[dict[str, object]]:
        """Return nearest analog matches."""

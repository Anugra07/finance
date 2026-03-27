from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.core.contracts import MacroContextSource
from ai_analyst.events.sector_opportunity import load_sector_rankings
from ai_analyst.warehouse.database import connect


class LocalMacroContextSource(MacroContextSource):
    """Local warehouse-backed macro/event context adapter.

    This is the default V1.5 implementation behind the `MacroContextSource`
    abstraction. It reads warehouse-backed theme intensities and normalized
    events, and can be swapped later with a World Monitor-backed adapter.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_theme_intensities(self, *, as_of: datetime) -> pd.DataFrame:
        conn = connect(self.settings)
        try:
            daily = conn.execute(
                """
                SELECT *
                FROM theme_intensity_daily
                WHERE date <= ?
                  AND date >= ?
                ORDER BY date DESC, intensity DESC, theme
                """,
                [as_of.date(), (as_of - timedelta(days=1)).date()],
            ).df()
            if not daily.empty:
                latest_date = pd.to_datetime(daily["date"]).max()
                return daily.loc[pd.to_datetime(daily["date"]) == latest_date].sort_values(
                    ["intensity", "theme"], ascending=[False, True]
                )

            return conn.execute(
                """
                SELECT
                    theme,
                    SUM(COALESCE(market_relevance, 0.0)) AS intensity,
                    MAX(event_time) AS latest_event_time
                FROM normalized_events
                WHERE event_time <= ?
                  AND event_time >= ?
                  AND theme IS NOT NULL
                GROUP BY theme
                ORDER BY intensity DESC, theme
                """,
                [as_of, as_of - timedelta(days=1)],
            ).df()
        finally:
            conn.close()

    def get_event_feed(
        self,
        *,
        ticker: str,
        as_of: datetime,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        conn = connect(self.settings)
        try:
            return conn.execute(
                """
                SELECT *
                FROM normalized_events
                WHERE event_time <= ?
                  AND event_time >= ?
                  AND (
                    list_contains(affected_entities, ?)
                    OR list_contains(affected_sectors, (
                        SELECT sector
                        FROM universe_membership
                        WHERE ticker = ?
                        ORDER BY snapshot_at DESC
                        LIMIT 1
                    ))
                    OR topic ILIKE '%' || ? || '%'
                  )
                ORDER BY event_time DESC, source
                """,
                [
                    as_of,
                    as_of - timedelta(days=lookback_days),
                    ticker.upper(),
                    ticker.upper(),
                    ticker.upper(),
                ],
            ).df()
        finally:
            conn.close()

    def get_recent_events(
        self,
        *,
        as_of: datetime,
        lookback_days: int = 7,
        limit: int = 15,
    ) -> pd.DataFrame:
        conn = connect(self.settings)
        try:
            return conn.execute(
                """
                SELECT *
                FROM normalized_events
                WHERE event_time <= ?
                  AND event_time >= ?
                ORDER BY event_time DESC, market_relevance DESC, source
                LIMIT ?
                """,
                [as_of, as_of - timedelta(days=lookback_days), limit],
            ).df()
        finally:
            conn.close()

    def get_sector_rankings(
        self,
        *,
        as_of: datetime,
        limit: int = 10,
    ) -> pd.DataFrame:
        return load_sector_rankings(self.settings, as_of=as_of.date(), limit=limit)

    def get_solution_ideas(
        self,
        *,
        as_of: datetime,
        limit: int = 5,
    ) -> pd.DataFrame:
        theme_frame = self.get_theme_intensities(as_of=as_of)
        if theme_frame.empty:
            return pd.DataFrame()

        conn = connect(self.settings)
        try:
            solutions = conn.execute("SELECT * FROM solution_mappings").df()
        finally:
            conn.close()

        if solutions.empty:
            return solutions

        ranked_themes = (
            theme_frame.head(limit)[["theme", "intensity"]]
            .dropna(subset=["theme"])
            .drop_duplicates(subset=["theme"])
        )
        if ranked_themes.empty:
            return pd.DataFrame()
        merged = solutions.merge(ranked_themes, on="theme", how="inner")
        if merged.empty:
            return merged
        return merged.sort_values(["intensity", "label"], ascending=[False, True]).head(limit)

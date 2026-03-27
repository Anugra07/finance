from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.warehouse.database import connect
from ai_analyst.warehouse.schema import CANONICAL_SCHEMAS


@dataclass(slots=True)
class SnapshotBundle:
    as_of: datetime
    macro: pd.DataFrame
    prices: pd.DataFrame
    companyfacts: pd.DataFrame
    submissions: pd.DataFrame
    universe: pd.DataFrame
    events: pd.DataFrame
    event_relations: pd.DataFrame
    event_source_assessment: pd.DataFrame
    event_narrative_risk: pd.DataFrame
    evidence_catalog: pd.DataFrame
    theme_intensities: pd.DataFrame
    theme_regimes: pd.DataFrame
    sector_rankings: pd.DataFrame
    historical_analogs: pd.DataFrame
    cross_asset_confirmation: pd.DataFrame
    pricing_discipline: pd.DataFrame
    trade_readiness: pd.DataFrame
    causal_state: pd.DataFrame
    causal_chains: pd.DataFrame


class SnapshotBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return connect(self.settings)

    def _safe_query(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        params: dict[str, object],
        *,
        fallback_relation: str,
    ) -> pd.DataFrame:
        try:
            return conn.execute(query, params).df()
        except duckdb.CatalogException:
            schema = CANONICAL_SCHEMAS[fallback_relation]
            return pd.DataFrame(columns=list(schema.keys()))

    def build(self, *, as_of: datetime, tickers: list[str] | None = None) -> SnapshotBundle:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        else:
            as_of = as_of.astimezone(UTC)

        conn = self._connect()
        try:
            base_params = {"as_of": as_of}
            price_params = {"as_of": as_of}
            ticker_filter = ""
            if tickers:
                ticker_filter = "AND ticker IN (SELECT unnest($tickers))"
                price_params["tickers"] = tickers

            macro = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY series_id, observation_date
                            ORDER BY realtime_start DESC, known_at DESC
                        ) AS row_num
                    FROM macro_vintages
                    WHERE known_at <= $as_of
                      AND realtime_start <= CAST($as_of AS DATE)
                      AND COALESCE(realtime_end, DATE '9999-12-31') >= CAST($as_of AS DATE)
                )
                SELECT * EXCLUDE (row_num)
                FROM ranked
                WHERE row_num = 1
                ORDER BY series_id, observation_date
                """,
                base_params,
            ).df()

            prices = conn.execute(
                f"""
                SELECT *
                FROM prices
                WHERE known_at <= $as_of
                {ticker_filter}
                ORDER BY ticker, date
                """,
                price_params,
            ).df()

            companyfacts = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY cik, taxonomy, metric_name, unit, period_end
                            ORDER BY snapshot_at DESC, filing_date DESC
                        ) AS row_num
                    FROM edgar_companyfacts
                    WHERE snapshot_at <= $as_of
                )
                SELECT * EXCLUDE (row_num)
                FROM ranked
                WHERE row_num = 1
                ORDER BY cik, taxonomy, metric_name, period_end
                """,
                base_params,
            ).df()

            submissions = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY cik, accession_number
                            ORDER BY snapshot_at DESC
                        ) AS row_num
                    FROM edgar_submissions
                    WHERE snapshot_at <= $as_of
                      AND COALESCE(acceptance_datetime, snapshot_at) <= $as_of
                )
                SELECT * EXCLUDE (row_num)
                FROM ranked
                WHERE row_num = 1
                ORDER BY acceptance_datetime, accession_number
                """,
                base_params,
            ).df()

            universe = conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY ticker
                            ORDER BY snapshot_at DESC
                        ) AS row_num
                    FROM universe_membership
                    WHERE snapshot_at <= $as_of
                )
                SELECT * EXCLUDE (row_num)
                FROM ranked
                WHERE row_num = 1
                ORDER BY ticker
                """,
                base_params,
            ).df()

            events = conn.execute(
                """
                SELECT *
                FROM normalized_events
                WHERE event_time <= $as_of
                  AND event_time >= $as_of - INTERVAL 30 DAY
                ORDER BY event_time DESC, source, topic
                """,
                base_params,
            ).df()

            event_relations = self._safe_query(
                conn,
                """
                SELECT *
                FROM event_relations
                WHERE event_id IN (
                    SELECT event_id
                    FROM normalized_events
                    WHERE event_time <= $as_of
                      AND event_time >= $as_of - INTERVAL 30 DAY
                )
                ORDER BY event_id, relation_type, target_type, target_id
                """,
                base_params,
                fallback_relation="event_relations",
            )
            event_source_assessment = self._safe_query(
                conn,
                """
                SELECT *
                FROM event_source_assessment
                WHERE event_id IN (
                    SELECT event_id
                    FROM normalized_events
                    WHERE event_time <= $as_of
                      AND event_time >= $as_of - INTERVAL 30 DAY
                )
                ORDER BY reliability_score DESC, event_id
                """,
                base_params,
                fallback_relation="event_source_assessment",
            )
            event_narrative_risk = self._safe_query(
                conn,
                """
                SELECT *
                FROM event_narrative_risk
                WHERE event_id IN (
                    SELECT event_id
                    FROM normalized_events
                    WHERE event_time <= $as_of
                      AND event_time >= $as_of - INTERVAL 30 DAY
                )
                ORDER BY actionability_score DESC, event_id
                """,
                base_params,
                fallback_relation="event_narrative_risk",
            )
            evidence_catalog = self._safe_query(
                conn,
                """
                SELECT *
                FROM evidence_catalog
                WHERE event_id IN (
                    SELECT event_id
                    FROM normalized_events
                    WHERE event_time <= $as_of
                      AND event_time >= $as_of - INTERVAL 30 DAY
                )
                ORDER BY timestamp DESC, evidence_id
                """,
                base_params,
                fallback_relation="evidence_catalog",
            )

            theme_intensities = conn.execute(
                """
                WITH latest_date AS (
                    SELECT MAX(date) AS date
                    FROM theme_intensity_daily
                    WHERE date <= CAST($as_of AS DATE)
                )
                SELECT t.*
                FROM theme_intensity_daily t
                INNER JOIN latest_date d ON t.date = d.date
                ORDER BY intensity DESC, theme
                """,
                base_params,
            ).df()

            theme_regimes = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM theme_regimes
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT r.*
                FROM theme_regimes r
                INNER JOIN latest_date d ON r.as_of_date = d.as_of_date
                ORDER BY regime_score DESC, theme
                """,
                base_params,
                fallback_relation="theme_regimes",
            )

            sector_rankings = conn.execute(
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM sector_rankings
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT s.*
                FROM sector_rankings s
                INNER JOIN latest_date d ON s.as_of_date = d.as_of_date
                ORDER BY rank_desc
                """,
                base_params,
            ).df()

            historical_analogs = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM historical_analogs
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT a.*
                FROM historical_analogs a
                INNER JOIN latest_date d ON a.as_of_date = d.as_of_date
                ORDER BY analog_type, similarity_score DESC, analog_key
                """,
                base_params,
                fallback_relation="historical_analogs",
            )
            cross_asset_confirmation = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM cross_asset_confirmation_daily
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT *
                FROM cross_asset_confirmation_daily
                WHERE as_of_date = (SELECT as_of_date FROM latest_date)
                ORDER BY confirmation_key
                """,
                base_params,
                fallback_relation="cross_asset_confirmation_daily",
            )
            pricing_discipline = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM pricing_discipline_daily
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT *
                FROM pricing_discipline_daily
                WHERE as_of_date = (SELECT as_of_date FROM latest_date)
                ORDER BY discipline_key
                """,
                base_params,
                fallback_relation="pricing_discipline_daily",
            )
            trade_readiness = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM trade_readiness_daily
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT *
                FROM trade_readiness_daily
                WHERE as_of_date = (SELECT as_of_date FROM latest_date)
                ORDER BY readiness_key
                """,
                base_params,
                fallback_relation="trade_readiness_daily",
            )

            causal_state = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM causal_state_daily
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT c.*
                FROM causal_state_daily c
                INNER JOIN latest_date d ON c.as_of_date = d.as_of_date
                ORDER BY state_key
                """,
                base_params,
                fallback_relation="causal_state_daily",
            )

            causal_chains = self._safe_query(
                conn,
                """
                WITH latest_date AS (
                    SELECT MAX(as_of_date) AS as_of_date
                    FROM causal_chain_activations
                    WHERE as_of_date <= CAST($as_of AS DATE)
                )
                SELECT c.*
                FROM causal_chain_activations c
                INNER JOIN latest_date d ON c.as_of_date = d.as_of_date
                ORDER BY ABS(weight) DESC, sector, ticker
                """,
                base_params,
                fallback_relation="causal_chain_activations",
            )
        finally:
            conn.close()

        return SnapshotBundle(
            as_of=as_of,
            macro=macro,
            prices=prices,
            companyfacts=companyfacts,
            submissions=submissions,
            universe=universe,
            events=events,
            event_relations=event_relations,
            event_source_assessment=event_source_assessment,
            event_narrative_risk=event_narrative_risk,
            evidence_catalog=evidence_catalog,
            theme_intensities=theme_intensities,
            theme_regimes=theme_regimes,
            sector_rankings=sector_rankings,
            historical_analogs=historical_analogs,
            cross_asset_confirmation=cross_asset_confirmation,
            pricing_discipline=pricing_discipline,
            trade_readiness=trade_readiness,
            causal_state=causal_state,
            causal_chains=causal_chains,
        )

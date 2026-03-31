from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.core.contracts import PriceSource
from ai_analyst.utils.dates import market_close_known_at, utc_now
from ai_analyst.utils.http import get_json
from ai_analyst.utils.io import read_json, write_json, write_parquet
from ai_analyst.warehouse.layout import (
    raw_snapshot_path,
    snapshot_time_from_path,
    warehouse_partition_path,
)

logger = logging.getLogger(__name__)


def _iter_raw_json(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _to_trade_date(value: object) -> date | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _to_float(value: object) -> float | None:
    try:
        if value in (None, "", "."):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prices_frame(
    *,
    ticker: str,
    metadata: dict[str, Any],
    prices_payload: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for price in prices_payload:
        trade_date = _to_trade_date(price.get("date"))
        if trade_date is None:
            continue
        rows.append(
            {
                "ticker": ticker,
                "market_code": "US",
                "country_code": "US",
                "exchange_code": metadata.get("exchangeCode"),
                "currency": metadata.get("currency") or "USD",
                "instrument_type": "equity",
                "tradeable": True,
                "symbol_native": ticker,
                "symbol_vendor": ticker,
                "date": trade_date,
                "open": _to_float(price.get("open")),
                "high": _to_float(price.get("high")),
                "low": _to_float(price.get("low")),
                "close": _to_float(price.get("close")),
                "adj_close": _to_float(price.get("adjClose")) or _to_float(price.get("close")),
                "adj_open": _to_float(price.get("adjOpen")) or _to_float(price.get("open")),
                "adj_high": _to_float(price.get("adjHigh")) or _to_float(price.get("high")),
                "adj_low": _to_float(price.get("adjLow")) or _to_float(price.get("low")),
                "adj_volume": _to_float(price.get("adjVolume")) or _to_float(price.get("volume")),
                "volume": _to_float(price.get("volume")),
                "div_cash": _to_float(price.get("divCash")) or 0.0,
                "split_factor": _to_float(price.get("splitFactor")) or 1.0,
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "start_date": metadata.get("startDate"),
                "end_date": metadata.get("endDate"),
            }
        )
    return pd.DataFrame(rows)


def _actions_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return prices[["ticker", "date", "div_cash", "split_factor"]].copy()
    


class TiingoPriceSource(PriceSource):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _url(self, path: str) -> str:
        return f"{self.settings.tiingo_base_url.rstrip('/')}/{path.lstrip('/')}"

    def _base_params(self) -> dict[str, object]:
        return {"token": self.settings.require_tiingo()}

    def _metadata(self, ticker: str) -> dict[str, Any]:
        payload = get_json(self.session, self._url(f"/daily/{ticker}"), params=self._base_params())
        return payload if isinstance(payload, dict) else {}

    def _prices(
        self,
        ticker: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            **self._base_params(),
            "resampleFreq": "daily",
        }
        if start_date:
            params["startDate"] = start_date.isoformat()
        if end_date:
            params["endDate"] = end_date.isoformat()
        payload = get_json(self.session, self._url(f"/daily/{ticker}/prices"), params=params)
        return payload if isinstance(payload, list) else []

    def fetch_history(
        self,
        ticker: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ticker = ticker.upper()
        metadata = self._metadata(ticker)
        prices_payload = self._prices(ticker, start_date=start_date, end_date=end_date)
        prices = _prices_frame(ticker=ticker, metadata=metadata, prices_payload=prices_payload)
        actions = _actions_frame(prices)
        return prices, actions

    def collect_raw(
        self,
        tickers: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        snapshot_at: datetime | None = None,
    ) -> list[Path]:
        snapshot_at = snapshot_at or utc_now()
        outputs: list[Path] = []
        for raw_ticker in tickers:
            ticker = raw_ticker.upper()
            try:
                metadata = self._metadata(ticker)
                prices = self._prices(ticker, start_date=start_date, end_date=end_date)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                logger.warning("Tiingo request failed for %s (%s): %s", ticker, status, exc)
                continue
            payload = {
                "ticker": ticker,
                "metadata": metadata,
                "prices": prices,
                "collected_at": snapshot_at.isoformat(),
            }
            out_path = raw_snapshot_path(
                self.settings,
                source="tiingo_prices",
                stem=ticker.lower(),
                snapshot_at=snapshot_at,
            )
            write_json(out_path, payload)
            outputs.append(out_path)
        return outputs


def transform_prices(settings: Settings) -> tuple[list[Path], list[Path]]:
    price_outputs: list[Path] = []
    action_outputs: list[Path] = []

    for path in _iter_raw_json(settings.raw_root / "tiingo_prices"):
        payload = read_json(path)
        snapshot_at = snapshot_time_from_path(path)
        transform_loaded_at = utc_now()
        ticker = str(payload.get("ticker") or path.stem.split("_")[0]).upper()
        prices = _prices_frame(
            ticker=ticker,
            metadata=payload.get("metadata") or {},
            prices_payload=payload.get("prices") or [],
        )
        if prices.empty:
            continue

        prices["known_at"] = prices["date"].map(
            lambda trade_date: market_close_known_at(trade_date, settings, market_scope="US")
        )
        prices["source_snapshot"] = str(path)
        prices["transform_loaded_at"] = transform_loaded_at
        prices_out = prices[
            [
                "ticker",
                "market_code",
                "country_code",
                "exchange_code",
                "currency",
                "instrument_type",
                "tradeable",
                "symbol_native",
                "symbol_vendor",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_volume",
                "volume",
                "div_cash",
                "split_factor",
                "known_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ].copy()

        actions = _actions_frame(prices)
        actions["market_code"] = "US"
        actions["exchange_code"] = prices.get("exchange_code", "US")
        actions["known_at"] = actions["date"].map(
            lambda trade_date: market_close_known_at(trade_date, settings, market_scope="US")
        )
        actions["source_snapshot"] = str(path)
        actions["transform_loaded_at"] = transform_loaded_at

        prices_path = warehouse_partition_path(
            settings,
            domain="prices/daily",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(prices_out, prices_path)
        price_outputs.append(prices_path)

        actions_path = warehouse_partition_path(
            settings,
            domain="prices/actions",
            partition_date=snapshot_at.date(),
            stem=path.stem,
        )
        write_parquet(actions, actions_path)
        action_outputs.append(actions_path)

    return price_outputs, action_outputs

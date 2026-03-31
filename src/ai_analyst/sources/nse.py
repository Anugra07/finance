from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from ai_analyst.config import Settings
from ai_analyst.utils.dates import market_close_known_at, utc_now
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


def _normalize_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    return symbol.replace("&", "AND").replace(".", "-")


def _normalize_tradeable(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    return normalized not in {"", "0", "false", "n", "no", "suspended"}


def _to_float(value: object) -> float | None:
    try:
        if value in (None, "", "."):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_date(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "local-ai-analyst nse collector",
            "Accept": "text/csv,application/json,text/html",
            "Referer": "https://www.nseindia.com/",
        }
    )
    return session


def _nse_bhavcopy_target(settings: Settings, trade_date: date) -> tuple[str, str]:
    transition_date = date(2024, 7, 8)
    if trade_date >= transition_date:
        return (
            settings.nse_udiff_bhavcopy_url_template.format(
                date_yyyymmdd=trade_date.strftime("%Y%m%d")
            ),
            "udiff_zip",
        )
    return (
        settings.nse_legacy_bhavcopy_url_template.format(
            date_ddmmyyyy=trade_date.strftime("%d%m%Y")
        ),
        "legacy_csv",
    )


def _extract_bhavcopy_rows(
    content: bytes,
    *,
    report_format: str,
    trade_date: date,
) -> list[dict[str, object]]:
    if report_format == "udiff_zip":
        with ZipFile(BytesIO(content)) as archive:
            csv_members = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if not csv_members:
                raise ValueError("Official NSE bhavcopy zip did not contain a CSV member.")
            with archive.open(csv_members[0]) as handle:
                frame = pd.read_csv(handle)
    else:
        frame = pd.read_csv(StringIO(content.decode("utf-8")))

    if frame.empty:
        return []

    symbol_column = next(
        (
            column
            for column in ("TckrSymb", "SYMBOL", "symbol", "ticker")
            if column in frame.columns
        ),
        None,
    )
    series_column = next(
        (column for column in ("SctySrs", "SERIES", "series") if column in frame.columns),
        None,
    )
    open_column = next(
        (column for column in ("OpnPric", "OPEN", "open") if column in frame.columns),
        None,
    )
    high_column = next(
        (column for column in ("HghPric", "HIGH", "high") if column in frame.columns),
        None,
    )
    low_column = next(
        (column for column in ("LwPric", "LOW", "low") if column in frame.columns),
        None,
    )
    close_column = next(
        (column for column in ("ClsPric", "CLOSE", "close") if column in frame.columns),
        None,
    )
    volume_column = next(
        (column for column in ("TtlTradgVol", "TOTTRDQTY", "volume") if column in frame.columns),
        None,
    )
    date_column = next(
        (column for column in ("TradDt", "TIMESTAMP", "date") if column in frame.columns),
        None,
    )
    isin_column = next((column for column in ("ISIN", "isin") if column in frame.columns), None)
    if not all([symbol_column, open_column, high_column, low_column, close_column]):
        raise ValueError("Official NSE bhavcopy schema is missing required OHLC columns.")

    normalized = pd.DataFrame(
        {
            "ticker": frame[symbol_column].map(_normalize_symbol),
            "series": frame[series_column] if series_column else "EQ",
            "date": pd.to_datetime(
                frame[date_column] if date_column else trade_date,
                errors="coerce",
            ).dt.date,
            "open": pd.to_numeric(frame[open_column], errors="coerce"),
            "high": pd.to_numeric(frame[high_column], errors="coerce"),
            "low": pd.to_numeric(frame[low_column], errors="coerce"),
            "close": pd.to_numeric(frame[close_column], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_column], errors="coerce")
            if volume_column
            else 0.0,
            "isin": frame[isin_column].astype(str) if isin_column else "",
        }
    )
    normalized["series"] = normalized["series"].astype(str).str.upper().str.strip()
    normalized = normalized.loc[normalized["series"] == "EQ"].copy()
    normalized = normalized.dropna(subset=["ticker", "date", "close"])
    if normalized.empty:
        return []
    normalized["adj_close"] = normalized["close"]
    normalized["adj_open"] = normalized["open"]
    normalized["adj_high"] = normalized["high"]
    normalized["adj_low"] = normalized["low"]
    normalized["adj_volume"] = normalized["volume"].fillna(0.0)
    normalized["tradeable"] = True
    normalized["div_cash"] = 0.0
    normalized["split_factor"] = 1.0
    normalized["symbol"] = normalized["ticker"]
    return normalized[
        [
            "ticker",
            "symbol",
            "series",
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
            "tradeable",
            "div_cash",
            "split_factor",
            "isin",
        ]
    ].to_dict(orient="records")


def collect_nifty200_constituents(
    settings: Settings,
    *,
    snapshot_at: datetime | None = None,
) -> Path:
    snapshot_at = snapshot_at or utc_now()
    session = _session()
    response = session.get(settings.nse_nifty200_url, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    payload = {
        "rows": frame.to_dict(orient="records"),
        "source_url": settings.nse_nifty200_url,
        "collected_at": snapshot_at.isoformat(),
    }
    out_path = raw_snapshot_path(
        settings,
        source="nse_nifty200",
        stem="nifty200_constituents",
        snapshot_at=snapshot_at,
    )
    write_json(out_path, payload)
    return out_path


def collect_nse_securities_master(
    settings: Settings,
    *,
    snapshot_at: datetime | None = None,
) -> Path:
    snapshot_at = snapshot_at or utc_now()
    session = _session()
    response = session.get(settings.nse_securities_master_url, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    payload = {
        "rows": frame.to_dict(orient="records"),
        "source_url": settings.nse_securities_master_url,
        "collected_at": snapshot_at.isoformat(),
    }
    out_path = raw_snapshot_path(
        settings,
        source="nse_securities_master",
        stem="securities_master",
        snapshot_at=snapshot_at,
    )
    write_json(out_path, payload)
    return out_path


def collect_nse_holidays(
    settings: Settings,
    *,
    snapshot_at: datetime | None = None,
) -> Path:
    snapshot_at = snapshot_at or utc_now()
    session = _session()
    response = session.get(settings.nse_holidays_url, timeout=60)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    rows: list[dict[str, object]] = []
    for table in tables:
        lowered = {str(column).lower(): column for column in table.columns}
        date_col = next(
            (
                actual
                for lower, actual in lowered.items()
                if "date" in lower and "trading" not in lower
            ),
            None,
        )
        if date_col is None:
            continue
        desc_col = next(
            (actual for lower, actual in lowered.items() if "description" in lower),
            None,
        )
        for _, row in table.iterrows():
            holiday_date = _to_date(row.get(date_col))
            if holiday_date is None:
                continue
            rows.append(
                {
                    "holiday_date": holiday_date.isoformat(),
                    "description": str(row.get(desc_col) or "NSE holiday"),
                }
            )
    payload = {
        "rows": rows,
        "source_url": settings.nse_holidays_url,
        "collected_at": snapshot_at.isoformat(),
    }
    out_path = raw_snapshot_path(
        settings,
        source="nse_holidays",
        stem="holidays",
        snapshot_at=snapshot_at,
    )
    write_json(out_path, payload)
    return out_path


def collect_nse_prices(
    settings: Settings,
    *,
    trade_date: date | None = None,
    lookback_days: int = 0,
    snapshot_at: datetime | None = None,
) -> list[Path]:
    snapshot_at = snapshot_at or utc_now()
    session = _session()
    anchor_date = trade_date or pd.Timestamp.now(tz=settings.india_market_timezone).date()
    outputs: list[Path] = []
    for offset in range(int(lookback_days) + 1):
        current_date = anchor_date - timedelta(days=offset)
        url, report_format = _nse_bhavcopy_target(settings, current_date)
        response = session.get(url, timeout=60)
        if response.status_code == 404:
            logger.info("NSE bhavcopy not available for %s at %s", current_date, url)
            continue
        response.raise_for_status()
        rows = _extract_bhavcopy_rows(
            response.content,
            report_format=report_format,
            trade_date=current_date,
        )
        if not rows:
            logger.info("NSE bhavcopy for %s contained no supported EQ rows.", current_date)
            continue
        payload = {
            "rows": rows,
            "source_url": url,
            "official_source": True,
            "report_format": report_format,
            "trade_date": current_date.isoformat(),
            "collected_at": snapshot_at.isoformat(),
        }
        out_path = raw_snapshot_path(
            settings,
            source="nse_prices",
            stem=f"prices_{current_date.isoformat()}",
            snapshot_at=snapshot_at,
        )
        write_json(out_path, payload)
        outputs.append(out_path)
    if not outputs:
        raise ValueError(
            "No official NSE bhavcopy files were collected. "
            "Check the trade date or increase lookback_days."
        )
    return outputs


def transform_nifty200_constituents(settings: Settings) -> list[Path]:
    outputs: list[Path] = []
    for path in _iter_raw_json(settings.raw_root / "nse_nifty200"):
        payload = read_json(path)
        rows = payload.get("rows") or []
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        snapshot_at = snapshot_time_from_path(path)
        transform_loaded_at = utc_now()
        rename_map = {
            "Company Name": "security",
            "Industry": "sub_industry",
            "Series": "instrument_type",
            "Symbol": "ticker",
        }
        frame = frame.rename(columns=rename_map)
        frame["ticker"] = frame["ticker"].map(_normalize_symbol)
        frame["security"] = frame.get("security", frame["ticker"])
        frame["sector"] = frame.get("Industry", frame.get("sub_industry", "Unknown"))
        frame["sub_industry"] = frame.get("sub_industry", frame["sector"])
        frame["market_code"] = "IN"
        frame["country_code"] = "IN"
        frame["exchange_code"] = "NSE"
        frame["currency"] = "INR"
        frame["instrument_type"] = frame.get("instrument_type", "equity").fillna("equity")
        frame["tradeable"] = True
        frame["symbol_native"] = frame["ticker"]
        frame["symbol_vendor"] = frame["ticker"].map(lambda value: f"{value}.NS")
        frame["cik"] = ""
        frame["as_of_date"] = snapshot_at.date()
        frame["snapshot_at"] = snapshot_at
        frame["source_snapshot"] = str(path)
        frame["transform_loaded_at"] = transform_loaded_at

        universe = frame[
            [
                "as_of_date",
                "ticker",
                "market_code",
                "country_code",
                "exchange_code",
                "currency",
                "instrument_type",
                "tradeable",
                "symbol_native",
                "symbol_vendor",
                "security",
                "sector",
                "sub_industry",
                "cik",
                "snapshot_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ].drop_duplicates(subset=["ticker", "market_code"])
        out_path = warehouse_partition_path(
            settings,
            domain="universe/sp500_current",
            partition_date=snapshot_at.date(),
            stem=f"nifty200_{path.stem}",
        )
        write_parquet(universe, out_path)
        outputs.append(out_path)

        security_master = universe.assign(
            security_group="nifty200",
            listing_date=pd.NaT,
        )[
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
                "security",
                "sector",
                "sub_industry",
                "cik",
                "security_group",
                "listing_date",
                "snapshot_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ]
        security_master_path = warehouse_partition_path(
            settings,
            domain="universe/security_master",
            partition_date=snapshot_at.date(),
            stem=f"nifty200_security_master_{path.stem}",
        )
        write_parquet(security_master, security_master_path)
    return outputs


def transform_nse_securities_master(settings: Settings) -> list[Path]:
    outputs: list[Path] = []
    for path in _iter_raw_json(settings.raw_root / "nse_securities_master"):
        payload = read_json(path)
        rows = payload.get("rows") or []
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        snapshot_at = snapshot_time_from_path(path)
        transform_loaded_at = utc_now()
        frame["ticker"] = frame.get("SYMBOL", "").map(_normalize_symbol)
        frame["security"] = frame.get("NAME OF COMPANY", frame["ticker"])
        frame["market_code"] = "IN"
        frame["country_code"] = "IN"
        frame["exchange_code"] = "NSE"
        frame["currency"] = "INR"
        frame["instrument_type"] = "equity"
        frame["tradeable"] = frame.get("SERIES", "EQ").map(
            lambda value: str(value).strip().upper() == "EQ"
        )
        frame["symbol_native"] = frame["ticker"]
        frame["symbol_vendor"] = frame["ticker"].map(lambda value: f"{value}.NS")
        frame["sector"] = frame.get("INDUSTRY", "Unknown").fillna("Unknown")
        frame["sub_industry"] = frame["sector"]
        frame["cik"] = ""
        frame["security_group"] = frame.get("SERIES", "EQ").fillna("EQ")
        frame["listing_date"] = pd.to_datetime(
            frame.get(" DATE OF LISTING"),
            errors="coerce",
            dayfirst=True,
        ).dt.date
        frame["snapshot_at"] = snapshot_at
        frame["source_snapshot"] = str(path)
        frame["transform_loaded_at"] = transform_loaded_at
        security_master = frame[
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
                "security",
                "sector",
                "sub_industry",
                "cik",
                "security_group",
                "listing_date",
                "snapshot_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ].drop_duplicates(subset=["ticker", "market_code"])
        out_path = warehouse_partition_path(
            settings,
            domain="universe/security_master",
            partition_date=snapshot_at.date(),
            stem=f"nse_security_master_{path.stem}",
        )
        write_parquet(security_master, out_path)
        outputs.append(out_path)
    return outputs


def transform_nse_holidays(settings: Settings) -> list[Path]:
    outputs: list[Path] = []
    for path in _iter_raw_json(settings.raw_root / "nse_holidays"):
        payload = read_json(path)
        rows = payload.get("rows") or []
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        snapshot_at = snapshot_time_from_path(path)
        transform_loaded_at = utc_now()
        frame["holiday_date"] = pd.to_datetime(frame["holiday_date"], errors="coerce").dt.date
        frame["market_code"] = "IN"
        frame["exchange_code"] = "NSE"
        frame["source_snapshot"] = str(path)
        frame["transform_loaded_at"] = transform_loaded_at
        holidays = frame[
            [
                "market_code",
                "exchange_code",
                "holiday_date",
                "description",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ].dropna(subset=["holiday_date"])
        out_path = warehouse_partition_path(
            settings,
            domain="market/holidays",
            partition_date=snapshot_at.date(),
            stem=f"nse_holidays_{path.stem}",
        )
        write_parquet(holidays, out_path)
        outputs.append(out_path)
    return outputs


def transform_nse_prices(settings: Settings) -> tuple[list[Path], list[Path]]:
    price_outputs: list[Path] = []
    action_outputs: list[Path] = []

    for path in _iter_raw_json(settings.raw_root / "nse_prices"):
        payload = read_json(path)
        rows = payload.get("rows") or []
        if not rows:
            continue
        snapshot_at = snapshot_time_from_path(path)
        transform_loaded_at = utc_now()
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame["ticker"] = frame.get("ticker", frame.get("symbol", "")).map(_normalize_symbol)
        frame["market_code"] = "IN"
        frame["country_code"] = "IN"
        frame["exchange_code"] = "NSE"
        frame["currency"] = "INR"
        frame["instrument_type"] = "equity"
        frame["tradeable"] = frame.get("tradeable", True).map(_normalize_tradeable)
        frame["symbol_native"] = frame["ticker"]
        frame["symbol_vendor"] = frame["ticker"].map(lambda value: f"{value}.NS")
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.date
        for column in [
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
        ]:
            if column not in frame.columns:
                default = 1.0 if column == "split_factor" else 0.0
                frame[column] = default
            frame[column] = frame[column].map(_to_float)
        frame["adj_close"] = frame["adj_close"].fillna(frame["close"])
        frame["adj_open"] = frame["adj_open"].fillna(frame["open"])
        frame["adj_high"] = frame["adj_high"].fillna(frame["high"])
        frame["adj_low"] = frame["adj_low"].fillna(frame["low"])
        frame["adj_volume"] = frame["adj_volume"].fillna(frame["volume"])
        frame["div_cash"] = frame["div_cash"].fillna(0.0)
        frame["split_factor"] = frame["split_factor"].fillna(1.0)
        frame["known_at"] = frame["date"].map(
            lambda trade_date: market_close_known_at(trade_date, settings, market_scope="IN")
        )
        frame["source_snapshot"] = str(path)
        frame["transform_loaded_at"] = transform_loaded_at
        prices = frame[
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
        ].dropna(subset=["ticker", "date", "close"])
        price_path = warehouse_partition_path(
            settings,
            domain="prices/daily",
            partition_date=snapshot_at.date(),
            stem=f"nse_prices_{path.stem}",
        )
        write_parquet(prices, price_path)
        price_outputs.append(price_path)

        actions = prices[
            [
                "ticker",
                "market_code",
                "exchange_code",
                "date",
                "div_cash",
                "split_factor",
                "known_at",
                "source_snapshot",
                "transform_loaded_at",
            ]
        ].copy()
        actions_path = warehouse_partition_path(
            settings,
            domain="prices/actions",
            partition_date=snapshot_at.date(),
            stem=f"nse_actions_{path.stem}",
        )
        write_parquet(actions, actions_path)
        action_outputs.append(actions_path)

    return price_outputs, action_outputs

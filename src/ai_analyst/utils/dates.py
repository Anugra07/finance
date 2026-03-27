from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from ai_analyst.config import Settings


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_iso_datetime(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def market_close_known_at(observation_date: date, settings: Settings) -> datetime:
    market_tz = ZoneInfo(settings.market_timezone)
    dt = datetime.combine(
        observation_date,
        time(settings.market_close_hour, settings.market_close_minute),
        tzinfo=market_tz,
    )
    return dt.astimezone(UTC)

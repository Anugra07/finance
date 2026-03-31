from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MACRO_SERIES = [
    "DGS10",
    "T10Y2Y",
    "CPIAUCSL",
    "DTWEXBGS",
    "BAMLH0A0HYM2",
    "VIXCLS",
    "SP500",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    workspace_root: Path = Field(
        default=Path("."),
        validation_alias=AliasChoices("AI_ANALYST_WORKSPACE_ROOT"),
    )
    duckdb_path: Path = Field(
        default=Path("data/warehouse/analyst.duckdb"),
        validation_alias=AliasChoices("AI_ANALYST_DUCKDB_PATH"),
    )
    data_root: Path = Field(default=Path("data"))
    reports_root: Path = Field(default=Path("reports"))
    mlruns_root: Path = Field(default=Path("mlruns"))

    fred_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FRED_API_KEY", "AI_ANALYST_FRED_API_KEY"),
    )
    tiingo_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TIINGO_API_KEY", "AI_ANALYST_TIINGO_API_KEY"),
    )
    sec_user_agent_name: str = Field(
        default="local-ai-analyst",
        validation_alias=AliasChoices("SEC_USER_AGENT_NAME", "AI_ANALYST_SEC_USER_AGENT_NAME"),
    )
    sec_user_agent_email: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SEC_USER_AGENT_EMAIL", "AI_ANALYST_SEC_USER_AGENT_EMAIL"),
    )
    sec_rate_limit_per_second: float = Field(
        default=10.0,
        validation_alias=AliasChoices(
            "SEC_RATE_LIMIT_PER_SECOND",
            "AI_ANALYST_SEC_RATE_LIMIT_PER_SECOND",
        ),
    )

    fred_api_base_url: str = "https://api.stlouisfed.org/fred"
    sec_data_base_url: str = "https://data.sec.gov"
    tiingo_base_url: str = "https://api.tiingo.com/tiingo"
    sp500_constituents_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    nse_base_url: str = "https://www.nseindia.com"
    nse_nifty200_url: str = (
        "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
    )
    nse_securities_master_url: str = (
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    )
    nse_holidays_url: str = "https://www.nseindia.com/resources/exchange-communication-holidays"
    nse_udiff_bhavcopy_url_template: str = (
        "https://nsearchives.nseindia.com/content/cm/"
        "BhavCopy_NSE_CM_0_0_0_{date_yyyymmdd}_F_0000.csv.zip"
    )
    nse_legacy_bhavcopy_url_template: str = (
        "https://nsearchives.nseindia.com/content/cm/NSE_CM_bhavcopy_{date_ddmmyyyy}.csv"
    )
    worldmonitor_base_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("WORLDMONITOR_BASE_URL", "AI_ANALYST_WORLDMONITOR_BASE_URL"),
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_HOST", "AI_ANALYST_OLLAMA_HOST"),
    )
    ollama_forecast_model: str = Field(
        default="deepseek-r1:14b",
        validation_alias=AliasChoices(
            "OLLAMA_FORECAST_MODEL",
            "AI_ANALYST_OLLAMA_FORECAST_MODEL",
        ),
    )
    ollama_critic_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OLLAMA_CRITIC_MODEL",
            "AI_ANALYST_OLLAMA_CRITIC_MODEL",
        ),
    )
    ollama_timeout_seconds: float = Field(
        default=180.0,
        validation_alias=AliasChoices(
            "OLLAMA_TIMEOUT_SECONDS",
            "AI_ANALYST_OLLAMA_TIMEOUT_SECONDS",
        ),
    )

    market_timezone: str = Field(
        default="America/New_York",
        validation_alias=AliasChoices("AI_ANALYST_MARKET_TIMEZONE"),
    )
    market_close_hour: int = Field(
        default=19,
        validation_alias=AliasChoices("AI_ANALYST_MARKET_CLOSE_HOUR"),
    )
    market_close_minute: int = Field(
        default=0,
        validation_alias=AliasChoices("AI_ANALYST_MARKET_CLOSE_MINUTE"),
    )
    india_market_timezone: str = Field(
        default="Asia/Kolkata",
        validation_alias=AliasChoices("AI_ANALYST_INDIA_MARKET_TIMEZONE"),
    )
    india_market_close_hour: int = Field(
        default=15,
        validation_alias=AliasChoices("AI_ANALYST_INDIA_MARKET_CLOSE_HOUR"),
    )
    india_market_close_minute: int = Field(
        default=30,
        validation_alias=AliasChoices("AI_ANALYST_INDIA_MARKET_CLOSE_MINUTE"),
    )

    v1_universe_size: int = Field(
        default=150,
        validation_alias=AliasChoices("AI_ANALYST_V1_UNIVERSE_SIZE"),
    )
    india_universe_size: int = Field(
        default=200,
        validation_alias=AliasChoices("AI_ANALYST_INDIA_UNIVERSE_SIZE"),
    )
    label_horizon_days: int = Field(default=5)
    primary_market_scope: str = Field(
        default="US",
        validation_alias=AliasChoices("AI_ANALYST_PRIMARY_MARKET_SCOPE"),
    )
    primary_horizon_days: int = Field(
        default=5,
        validation_alias=AliasChoices("AI_ANALYST_PRIMARY_HORIZON_DAYS"),
    )
    small_account_default_capital: float = Field(
        default=5000.0,
        validation_alias=AliasChoices("AI_ANALYST_SMALL_ACCOUNT_DEFAULT_CAPITAL"),
    )
    small_account_default_currency: str = Field(
        default="INR",
        validation_alias=AliasChoices("AI_ANALYST_SMALL_ACCOUNT_DEFAULT_CURRENCY"),
    )
    india_price_source_mode: str = Field(
        default="official_nse",
        validation_alias=AliasChoices("AI_ANALYST_INDIA_PRICE_SOURCE_MODE"),
    )
    us_benchmark_ticker: str = Field(
        default="SPY",
        validation_alias=AliasChoices("AI_ANALYST_US_BENCHMARK_TICKER"),
    )
    india_benchmark_ticker: str = Field(
        default="NIFTY200",
        validation_alias=AliasChoices("AI_ANALYST_INDIA_BENCHMARK_TICKER"),
    )
    macro_series: list[str] = Field(default_factory=lambda: DEFAULT_MACRO_SERIES.copy())
    trust_tier: str = Field(
        default="experimental",
        validation_alias=AliasChoices("AI_ANALYST_TRUST_TIER"),
    )

    # Paper trading
    paper_trade_horizon_days: int = Field(default=5)
    paper_trade_max_positions: int = Field(default=20)
    paper_trade_model_weight: float = Field(default=0.6)
    paper_trade_llm_weight: float = Field(default=0.4)

    # Regime detection
    hmm_n_states: int = Field(default=3)
    hmm_retrain_days: int = Field(default=90)

    @field_validator(
        "workspace_root", "duckdb_path", "data_root", "reports_root", "mlruns_root", mode="before"
    )
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return value if isinstance(value, Path) else Path(value)

    @field_validator("macro_series", mode="before")
    @classmethod
    def _coerce_macro_series(cls, value: Any) -> list[str]:
        if value is None:
            return DEFAULT_MACRO_SERIES.copy()
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @property
    def workspace(self) -> Path:
        return self.workspace_root.expanduser().resolve()

    @property
    def duckdb_file(self) -> Path:
        path = self.duckdb_path
        if not path.is_absolute():
            path = self.workspace / path
        return path

    @property
    def raw_root(self) -> Path:
        return (self.workspace / self.data_root / "raw").resolve()

    @property
    def warehouse_root(self) -> Path:
        return (self.workspace / self.data_root / "warehouse").resolve()

    @property
    def reports_path(self) -> Path:
        return (self.workspace / self.reports_root).resolve()

    @property
    def mlruns_path(self) -> Path:
        return (self.workspace / self.mlruns_root).resolve()

    @property
    def sec_user_agent(self) -> str:
        email = self.sec_user_agent_email or "missing-email@example.com"
        return f"{self.sec_user_agent_name} {email}"

    def require_fred(self) -> str:
        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY is required for FRED/ALFRED collection.")
        return self.fred_api_key

    def require_tiingo(self) -> str:
        if not self.tiingo_api_key:
            raise ValueError("TIINGO_API_KEY is required for price collection.")
        return self.tiingo_api_key

    def require_sec_identity(self) -> str:
        if not self.sec_user_agent_email:
            raise ValueError("SEC_USER_AGENT_EMAIL is required for SEC collection.")
        return self.sec_user_agent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

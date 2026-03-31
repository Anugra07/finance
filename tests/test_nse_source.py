from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.sources.nse import (
    collect_nse_prices,
    transform_nifty200_constituents,
    transform_nse_holidays,
    transform_nse_securities_master,
)
from ai_analyst.utils.io import read_json, write_json
from ai_analyst.warehouse.database import connect, refresh_views


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_nse_transforms_materialize_security_master_and_holidays(tmp_path) -> None:
    settings = _settings(tmp_path)

    nifty_root = settings.raw_root / "nse_nifty200" / "2026-03-30"
    nifty_root.mkdir(parents=True, exist_ok=True)
    write_json(
        nifty_root / "nifty200_constituents_20260330T000000Z.json",
        {
            "rows": [
                {
                    "Symbol": "RELIANCE",
                    "Company Name": "Reliance Industries Ltd",
                    "Industry": "Energy",
                    "Series": "EQ",
                }
            ]
        },
    )

    master_root = settings.raw_root / "nse_securities_master" / "2026-03-30"
    master_root.mkdir(parents=True, exist_ok=True)
    write_json(
        master_root / "securities_master_20260330T000000Z.json",
        {
            "rows": [
                {
                    "SYMBOL": "RELIANCE",
                    "NAME OF COMPANY": "Reliance Industries Ltd",
                    "SERIES": "EQ",
                    "INDUSTRY": "Energy",
                    " DATE OF LISTING": "29-Nov-1995",
                }
            ]
        },
    )

    holiday_root = settings.raw_root / "nse_holidays" / "2026-03-30"
    holiday_root.mkdir(parents=True, exist_ok=True)
    write_json(
        holiday_root / "holidays_20260330T000000Z.json",
        {"rows": [{"holiday_date": "2026-08-15", "description": "Independence Day"}]},
    )

    assert transform_nifty200_constituents(settings)
    assert transform_nse_securities_master(settings)
    assert transform_nse_holidays(settings)
    refresh_views(settings)

    conn = connect(settings)
    try:
        master = conn.execute(
            """
            SELECT ticker, market_code, exchange_code, currency, tradeable
            FROM security_master
            WHERE ticker = 'RELIANCE'
            ORDER BY snapshot_at DESC
            LIMIT 1
            """
        ).df()
        holidays = conn.execute(
            """
            SELECT holiday_date, description
            FROM market_holidays
            WHERE market_code = 'IN'
            """
        ).df()
    finally:
        conn.close()

    assert not master.empty
    assert master.iloc[0]["market_code"] == "IN"
    assert master.iloc[0]["exchange_code"] == "NSE"
    assert master.iloc[0]["currency"] == "INR"
    assert bool(master.iloc[0]["tradeable"]) is True
    assert pd.Timestamp(holidays.iloc[0]["holiday_date"]).date().isoformat() == "2026-08-15"


def test_collect_nse_prices_uses_official_udiff_bhavcopy(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)

    csv_payload = (
        b"TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TradDt,ISIN\n"
        b"RELIANCE,EQ,2400,2430,2390,2420,1000000,2026-03-30,INE002A01018\n"
    )
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, mode="w") as archive:
        archive.writestr("BhavCopy_NSE_CM_0_0_0_20260330_F_0000.csv", csv_payload)

    class FakeResponse:
        status_code = 200
        content = zip_buffer.getvalue()

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def get(self, url: str, timeout: int = 60):  # noqa: ARG002
            assert "BhavCopy_NSE_CM_0_0_0_20260330_F_0000.csv.zip" in url
            return FakeResponse()

    monkeypatch.setattr("ai_analyst.sources.nse._session", lambda: FakeSession())
    outputs = collect_nse_prices(settings, trade_date=pd.Timestamp("2026-03-30").date())

    assert len(outputs) == 1
    payload = read_json(outputs[0])
    assert payload["official_source"] is True
    assert payload["report_format"] == "udiff_zip"
    assert payload["rows"][0]["ticker"] == "RELIANCE"
    assert payload["rows"][0]["close"] == 2420

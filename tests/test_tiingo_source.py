from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ai_analyst.config import Settings
from ai_analyst.sources.tiingo import transform_prices
from ai_analyst.utils.dates import market_close_known_at
from ai_analyst.utils.io import write_json
from ai_analyst.warehouse.layout import raw_snapshot_path


def _settings(tmp_path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        duckdb_path=tmp_path / "data/warehouse/analyst.duckdb",
        sec_user_agent_email="test@example.com",
    )


def test_tiingo_transform_materializes_prices_and_actions(tmp_path) -> None:
    settings = _settings(tmp_path)
    snapshot_at = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
    payload = {
        "ticker": "AAPL",
        "metadata": {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "exchangeCode": "NASDAQ",
        },
        "prices": [
            {
                "date": "2024-01-04T00:00:00.000Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adjOpen": 100.0,
                "adjHigh": 101.0,
                "adjLow": 99.0,
                "adjClose": 100.5,
                "adjVolume": 1000.0,
                "volume": 1000.0,
                "divCash": 0.24,
                "splitFactor": 1.0,
            }
        ],
    }
    write_json(
        raw_snapshot_path(
            settings,
            source="tiingo_prices",
            stem="aapl",
            snapshot_at=snapshot_at,
        ),
        payload,
    )

    price_outputs, action_outputs = transform_prices(settings)

    assert price_outputs
    assert action_outputs

    prices_frame = pd.read_parquet(price_outputs[0])
    actions_frame = pd.read_parquet(action_outputs[0])

    assert prices_frame.iloc[0]["ticker"] == "AAPL"
    assert float(prices_frame.iloc[0]["adj_close"]) == 100.5
    assert prices_frame.iloc[0]["known_at"] == market_close_known_at(
        pd.Timestamp("2024-01-04").date(),
        settings,
    )
    assert float(actions_frame.iloc[0]["div_cash"]) == 0.24

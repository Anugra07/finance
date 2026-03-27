# Architecture Notes

## Current implementation

The repo now contains a working V1 foundation plus a first geo-energy context track:

- source adapters for `FRED`, `ALFRED`, `SEC EDGAR`, `Tiingo`, and current S&P 500 membership
- optional `World Monitor` raw collector for sanctions, maritime warnings, cyber threats, and market context
- raw JSON snapshot storage under `data/raw/...`
- Parquet transforms under `data/warehouse/...`
- DuckDB view refresh and a point-in-time `SnapshotBuilder`
- V1 universe selection, factor generation, label generation, walk-forward splits, and baseline training/report scaffolds
- normalized geo-energy events, theme intensity aggregation, hand-coded sector-theme exposures, and sector opportunity ranking

## Stable abstractions

- `PriceSource`
- `MacroContextSource`
- `SnapshotBuilder`
- `WalkForwardSpec`
- `AnalogStore`
- `ContextPack`

## Parallel context architecture

The finance core stays intact. The geo-energy layer runs beside it:

1. `finance warehouse -> finance features -> finance ranking model`
2. `raw geo/energy events -> normalized events -> theme intensities -> sector/theme transmission -> sector opportunities`
3. `sector + stock context features -> analyst context pack -> scenario engine -> decision layer`

This keeps non-financial context additive and auditable instead of rewriting the price/fundamental core.

## Current warehouse additions

- `normalized_events`
- `event_entities`
- `theme_intensity_hourly`
- `theme_intensity_daily`
- `sector_theme_exposure`
- `industry_theme_exposure`
- `stock_theme_exposure`
- `sector_rankings`
- `theme_regimes`
- `historical_analogs`
- `solution_mappings`

## Next implementation steps

1. Add more event collectors that populate the canonical `normalized_events` table beyond World Monitor.
2. Persist sector and stock exposure refinements beyond the hand-coded sector priors.
3. Add EDGAR filing text extraction for section-level chunking.
4. Add embedding generation and `Qdrant` indexing for analog retrieval.
5. Add Ollama forecast + critic passes on top of `ContextPackBuilder`.
6. Add geo-energy scenario families, MAPIE intervals, and portfolio optimization.

## Current commands

```bash
ai-analyst bootstrap
ai-analyst collect fred-current
ai-analyst collect fred-vintages
ai-analyst collect sec-submissions --cik 0000320193
ai-analyst collect sec-companyfacts --cik 0000320193
ai-analyst collect prices --tickers AAPL,MSFT,NVDA,SPY
ai-analyst collect universe
ai-analyst collect worldmonitor
ai-analyst transform all
ai-analyst transform worldmonitor
ai-analyst db refresh
ai-analyst geo seed-defaults
ai-analyst geo build-context
ai-analyst features build
ai-analyst snapshot --as-of 2024-01-31T23:59:59Z
ai-analyst train baseline
```

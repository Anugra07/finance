# Requirements Ledger

This file captures the active build contract so the scope does not drift as the project grows.

## Locked V1 finance core

- Universe: top 150 liquid current S&P 500 names
- Frequency: daily
- Horizon: 5 trading days
- Ranking target: sector-neutral excess-alpha rank
- Guardrail: abstain mode stays on
- Rule: no model work before the point-in-time warehouse is correct

## Core architecture

The system is not "finance-only" and it is not a generic event chatbot. The intended structure is:

1. finance core
2. geo-energy context layer
3. sector/theme transmission layer
4. scenario engine
5. decision and ranking layer
6. analyst explanation layer

The geo-energy layer is additive. It should never rewrite the finance warehouse or price feature logic.

## Geo-energy transmission design

Pipeline:

`raw geo/energy events -> normalized event scores -> theme intensities -> sector exposures -> stock/context features -> sector and stock ranking`

This enables questions such as:

- which sectors benefit from oil stress or LNG tightness
- which sectors are hurt by shipping disruption or sanctions expansion
- which sectors act as hedges or stabilizers
- which ideas likely benefit if conflict widens or policy relief arrives

## Canonical event schema

`event_time`
`ingest_time`
`source`
`theme`
`region`
`affected commodities`
`affected sectors`
`severity`
`confidence`
`novelty`
`duration estimate`
`raw_ref`

## Canonical themes

- `oil_supply_risk`
- `gas_supply_risk`
- `shipping_stress`
- `sanctions_pressure`
- `grid_stress`
- `defense_escalation`
- `cyber_infra_risk`
- `industrial_metal_tightness`
- `policy_relief`
- `macro_demand_softness`

## Required warehouse tables

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

## Required feature columns

- `oil_supply_risk_1d`
- `gas_supply_risk_1d`
- `shipping_stress_1d`
- `sanctions_pressure_1d`
- `defense_escalation_1d`
- `grid_stress_1d`
- `cyber_infra_risk_1d`
- `policy_relief_prob_1d`
- `commodity_shock_score`
- `geo_novelty_score`
- `regional_concentration_risk`
- `sector_context_shock`
- `stock_context_shock`
- `analog_match_score`
- `event_dispersion_score`

## Daily analyst output contract

Every daily report should eventually provide:

1. Geo-energy condition summary
2. Theme intensity dashboard
3. Sector boom / stress ranking
4. Beneficiary / hedge / solution ideas
5. Stock ranking within top sectors

## World Monitor placement

World Monitor is an optional adapter and comparison source, not the finance foundation.

- It can supply sanctions, maritime, cyber, and market context.
- It should sit behind `MacroContextSource`.
- The local finance warehouse remains the source of truth for prices, macro vintages, SEC data, features, and backtests.
- Current local integration target:
  - `/api/market/v1/get-sector-summary`
  - `/api/market/v1/list-etf-flows`
  - `/api/sanctions/v1/list-sanctions-pressure`
  - `/api/maritime/v1/list-navigational-warnings`
  - `/api/cyber/v1/list-cyber-threats`

## World Monitor local setup notes

- Local repo path: `/Users/anugragupta/Desktop/finance/worldmonitor-forked`
- Docs indicate Docker self-hosting at `http://localhost:3000`
- Machine check on this Mac:
  - Node: `v20.19.5`
  - npm: `10.8.2`
  - Docker: available
  - Docker Compose: available
- World Monitor docs prefer Node 22+ for host seeding, but Docker is the cleanest path for local operation

## Current World Monitor run status

- Local Docker stack is up:
  - `worldmonitor`
  - `worldmonitor-ais-relay`
  - `worldmonitor-redis`
  - `worldmonitor-redis-rest`
- Verified live local endpoints:
  - `/api/market/v1/get-sector-summary`
  - `/api/market/v1/list-market-quotes`
  - `/api/market/v1/list-etf-flows`
  - `/api/cyber/v1/list-cyber-threats`
- Current endpoint state:
  - sectors: live
  - market quotes: live
  - ETF flows: live
  - cyber threats: live
  - sanctions pressure: currently empty
  - navigational warnings: currently empty
- Self-hosting fixes already applied to the local World Monitor fork:
  - `Dockerfile.relay` now installs build tooling required by native Node modules
  - `docker-compose.yml` now passes Redis and market/env keys into `ais-relay`
  - `scripts/ais-relay.cjs` now accepts local `http://redis-rest:80` for self-hosted Redis REST
  - `scripts/ais-relay.cjs` now uses the correct HTTP client for local Redis REST writes
  - `scripts/run-seeders-local-docker.sh` was added to run seeders inside a clean Node 22 Docker environment

## Current analyst warehouse status

- Live FRED current snapshots have been collected and transformed
- Live ALFRED vintage snapshots have been collected in chunked form and transformed
- Live World Monitor snapshot has been collected into `data/raw/worldmonitor/`
- The snapshot has been transformed into:
  - `normalized_events`
  - `event_entities`
  - `theme_intensity_hourly`
  - `theme_intensity_daily`
- Default geo-energy reference data has been materialized into:
  - `sector_theme_exposure`
  - `solution_mappings`
- Current verified row counts after refresh:
  - `macro_observations`: 55,759
  - `macro_vintages`: 355,667
  - `normalized_events`: 50
  - `event_entities`: 250
  - `theme_intensity_hourly`: 2
  - `theme_intensity_daily`: 2
  - `sector_theme_exposure`: 25
  - `solution_mappings`: 9
- Current normalized event dates:
  - `2026-03-12` for one dated cyber event
  - `2026-03-25` for the latest local World Monitor snapshot
- SEC universe auto-collection is now wired:
  - `ai-analyst collect sec-universe --limit 1` completed successfully
  - `ai-analyst transform sec` materialized fresh submissions, filing index, and companyfacts files
- Local Ollama status:
  - `http://localhost:11434/api/tags` responds successfully on this Mac
  - the new `ai-analyst analyst forecast` command is wired to the local Ollama HTTP API
  - current configured local model: `deepseek-r1:14b`
  - current configured host: `http://localhost:11434`
  - the default-config live forecast path has been verified end to end
- Source adapter status:
  - `src/ai_analyst/sources/fred.py`: implemented and live-verified
  - `src/ai_analyst/sources/sec.py`: implemented and live-verified on a limited universe run
  - `src/ai_analyst/sources/tiingo.py`: implemented and test-verified
- Geo-energy reference seeding now writes:
  - `sector_theme_exposure`
  - `industry_theme_exposure`
  - `stock_theme_exposure`
  - `solution_mappings`
- Known generated-data cleanup already done:
  - quarantined a corrupt `v1_top150` parquet artifact
  - quarantined accidental `1970-01-01` theme partitions caused by zero timestamps

## Known blockers

- The CLI now lazy-loads command modules so partial workflows can still run cleanly while later modules are under construction
- `sector_rankings` are still empty because there is no live `feature_matrix` materialized yet
- World Monitor health is still not fully green because several optional/global feeds remain empty or rate-limited
- AviationStack requests are currently returning `403/429` in the local relay
- UCDP is disabled until a `UCDP_ACCESS_TOKEN` is configured
- `TIINGO_API_KEY` is still unset in the local analyst repo, so live price ingestion is not yet verified
- The analyst repo still needs live Tiingo price collection plus full feature materialization before V1 training can run end to end

## Freshness budget

- prices: daily after market close
- macro: on release cadence with release timestamps
- events: hourly aggregates plus daily rollups
- LLM: answer strictly from the latest context pack with freshness flags

## Current implementation status

- finance PIT warehouse is implemented
- SEC submissions, companyfacts, filing index transforms, and latest-universe SEC collection are implemented
- Tiingo price and corporate-action transforms are implemented
- V1 feature pipeline is implemented
- baseline training/report scaffolds are implemented
- geo-energy schema, theme aggregation, sector exposures, sector ranking, and World Monitor collector/transform have been added
- starter industry and stock theme exposures are now seeded alongside sector exposures
- nightly report output now includes geo-energy condition summary, theme dashboard, sector boom/stress rankings, solution ideas, and stock ranking within highlighted sectors
- ContextPack now has a working local downstream consumer:
  - two-pass analyst forecast via Ollama HTTP
  - forecast + critic merge path
  - CLI entrypoint: `ai-analyst analyst forecast`
  - local model output is normalized back into the contract when the model returns percentages or loose verdict labels
  - stale critical inputs now force `low_conviction` with explicit abstain reasons
  - citations are normalized back to approved context-pack field labels
- analog retrieval, analyst critic pass, scenarios, portfolios, and ops remain in progress

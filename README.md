# Local AI Analyst

Local-first research stack for:

- a point-in-time macro + filings + price warehouse
- a parallel geo-energy context engine
- leakage-safe factor engineering
- nightly stock ranking
- explainable analyst outputs
- later scenario, analog, and portfolio layers

## Repo layout

```text
data/
  raw/
  warehouse/
docs/
mlruns/
notebooks/
reports/
src/ai_analyst/
tests/
```

## Quick start

1. Create a virtual environment.
2. Install the package plus the V1 and dev extras.
3. Copy `.env.example` to `.env`.
4. Add `FRED_API_KEY`, `TIINGO_API_KEY`, and SEC identity headers.
5. Run the bootstrap commands below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[v1,dev]'
cp .env.example .env
ai-analyst bootstrap
ai-analyst collect fred-current
ai-analyst collect fred-vintages
ai-analyst collect sec-submissions --cik 0000320193
ai-analyst collect sec-companyfacts --cik 0000320193
ai-analyst collect prices --tickers AAPL,MSFT,NVDA,SPY
ai-analyst collect worldmonitor
ai-analyst transform all
ai-analyst db refresh
ai-analyst geo seed-defaults
ai-analyst geo build-context
ai-analyst snapshot --as-of 2024-01-31T23:59:59Z
```

## Root Start Script

You can bring the local stack up from the repo root with:

```bash
chmod +x ./start.sh
./start.sh
```

Useful variants:

```bash
./start.sh --check-only
./start.sh --skip-worldmonitor
./start.sh --skip-ollama
./start.sh --skip-bootstrap
```

The full testing flow is documented in [`docs/testing-guide.md`](docs/testing-guide.md).

## Local Ollama setup

The analyst layer can now use a local Ollama model by default.

Current local configuration:

- `OLLAMA_HOST=http://localhost:11434`
- `OLLAMA_FORECAST_MODEL=deepseek-r1:14b`
- `OLLAMA_CRITIC_MODEL=deepseek-r1:14b`

Useful commands:

```bash
ai-analyst analyst ollama-status
ai-analyst context-pack --ticker AAPL --as-of 2026-03-25T20:00:00Z
ai-analyst analyst forecast --ticker AAPL --as-of 2026-03-25T20:00:00Z
```

Notes:

- The current local model is working end to end, but it is relatively slow for interactive use.
- The analyst layer now normalizes loose model outputs back into the project contract, including:
  - confidence in `[0, 1]`
  - horizon fixed to `5 trading days`
  - verdict mapped to `outperform`, `neutral`, `underperform`, or `low_conviction`
  - citations mapped back to approved context-pack fields
- If critical inputs are stale, the analyst forecast now adds abstain reasons and forces a `low_conviction` verdict.

## Current scope in this implementation

This turn implements the V1 foundation:

- env/config management
- raw snapshot storage
- FRED + ALFRED collectors
- SEC submissions + companyfacts collectors with fair-access guardrails
- Tiingo price + corporate actions loader
- World Monitor collector/transform scaffold for sanctions, maritime, cyber, and market context
- Parquet warehouse transformers
- DuckDB refresh and point-in-time snapshot builder
- V1 universe, features, walk-forward split, baseline training/report scaffolds
- geo-energy theme intensities, sector-theme exposures, sector rankings, and context-pack extensions
- unit tests for PIT and feature logic

The V1.5 and V2 modules are also represented in the package structure so later work can plug into stable interfaces.

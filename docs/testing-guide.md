# Testing Guide

## Scope

This guide covers:

- the root startup script
- local service health checks
- repository lint and unit tests
- warehouse smoke tests
- analyst smoke tests

All commands assume you are in:

```bash
cd /Users/anugragupta/Desktop/finance
```

## 1. Cold-Start Test

Start from a stopped state:

```bash
lsof -nP -iTCP -sTCP:LISTEN | rg '(:3000|:8079|:11434|:5678)'
docker ps
```

The finance stack should be absent before this test.

Run the root startup script:

```bash
chmod +x ./start.sh
./start.sh
```

Expected result:

- DuckDB bootstrap completes
- World Monitor starts on `localhost:3000`
- Ollama starts on `localhost:11434`
- the script prints a status summary and next commands

## 2. Startup Script Health Checks

Check status without starting anything:

```bash
./start.sh --check-only
```

Start only the Python side:

```bash
./start.sh --skip-worldmonitor --skip-ollama
```

Start without re-running bootstrap:

```bash
./start.sh --skip-bootstrap
```

## 3. Service Smoke Tests

Verify World Monitor:

```bash
curl -fsS 'http://localhost:3000/api/market/v1/get-sector-summary?period=1d' | head
```

Verify Ollama:

```bash
curl -fsS 'http://localhost:11434/api/tags' | head
```

Verify the app can see Ollama:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app analyst ollama-status
```

## 4. Code Quality Tests

Run lint:

```bash
ruff check src tests
```

Run the full unit test suite:

```bash
pytest -q
```

If you want targeted smoke coverage first:

```bash
pytest -q tests/test_universe_source.py tests/test_feature_engineering.py
pytest -q tests/test_llm_forecast.py tests/test_nightly_report.py
```

## 5. Warehouse Smoke Tests

Refresh DuckDB views:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app db refresh
```

Check freshness:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app db freshness
```

Build a point-in-time snapshot:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app snapshot --as-of 2024-01-31T23:59:59Z
```

## 6. Data Pipeline Smoke Tests

Universe and prices:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app collect universe
PYTHONPATH=src python3 -m ai_analyst.cli.app transform universe
PYTHONPATH=src python3 -m ai_analyst.cli.app collect prices-universe --limit 25 --start-date 2024-01-01
PYTHONPATH=src python3 -m ai_analyst.cli.app transform prices
PYTHONPATH=src python3 -m ai_analyst.cli.app db refresh
```

Macro and geo context:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app collect fred-current
PYTHONPATH=src python3 -m ai_analyst.cli.app collect fred-vintages
PYTHONPATH=src python3 -m ai_analyst.cli.app collect worldmonitor
PYTHONPATH=src python3 -m ai_analyst.cli.app transform worldmonitor
PYTHONPATH=src python3 -m ai_analyst.cli.app geo seed-defaults
PYTHONPATH=src python3 -m ai_analyst.cli.app geo build-context
```

## 7. Analyst Smoke Tests

Build a context pack:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app context-pack --ticker AAPL --as-of 2026-03-25T20:00:00Z
```

Run a local analyst forecast:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app analyst forecast --ticker AAPL --as-of 2026-03-25T20:00:00Z
```

Expected behavior:

- the output is valid JSON
- stale inputs force `low_conviction`
- citations stay inside approved context-pack sections

## 8. Remaining V1 Validation

The next unfinished live validation steps are:

1. complete SEC companyfacts coverage for the active universe
2. rebuild `feature_matrix` and `label_matrix`
3. run `train baseline`
4. confirm a real nightly ranked report is written into `reports/`

Commands:

```bash
PYTHONPATH=src python3 -m ai_analyst.cli.app collect sec-universe
PYTHONPATH=src python3 -m ai_analyst.cli.app transform sec
PYTHONPATH=src python3 -m ai_analyst.cli.app db refresh
PYTHONPATH=src python3 -m ai_analyst.cli.app features build
PYTHONPATH=src python3 -m ai_analyst.cli.app train baseline
```

## 9. Troubleshooting

If `./start.sh` says Python cannot import `ai_analyst`:

```bash
python3 -m pip install -e '.[v1,v15,dev]'
```

If World Monitor does not come up:

```bash
cd /Users/anugragupta/Desktop/finance/worldmonitor-forked
docker compose logs --tail=100
```

If Ollama does not answer:

```bash
cat /tmp/local-ai-analyst-ollama.log
```

If DuckDB is locked:

- make sure no background `python -m ai_analyst...` process is still running
- rerun `db refresh` after the process exits

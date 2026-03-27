# World Monitor Analyst Panel Integration Plan

## Summary
Integrate the local finance analyst into World Monitor as a new `analyst` panel, enabled in all variants, while keeping the finance core separate and reusing World Monitor’s settings, panel system, and local sidecar lifecycle. The UI will be a dashboard + chat hybrid: brief on load, ask-anything input, predictions when available, and always show proof, analogs, risks, freshness, and abstain reasons.

The integration will not rewrite the Python finance engine in JS. World Monitor will call a local analyst API, and the same backend contract can later power terminal commands.

## Key Changes
### 1. Local analyst API and lifecycle
- Add a persistent local analyst HTTP service in the finance repo, exposed as a new entrypoint such as `ai-analyst serve`.
- Add World Monitor proxy routes under `/api/analyst/v1/*` in the local sidecar, with the sidecar responsible for:
  - starting the Python analyst service if it is not already running
  - health-checking it
  - passing through requests from the UI
  - injecting settings-derived runtime config from World Monitor
- Reuse World Monitor settings as the source of truth for:
  - Ollama endpoint/model
  - Groq/OpenRouter fallback settings if later enabled
  - shared data credentials already present in World Monitor settings
- Finance repo `.env` remains fallback-only for standalone CLI use, not the primary config path for the UI.

### 2. New public interfaces
- Add local analyst API endpoints:
  - `GET /api/analyst/v1/health`
  - `POST /api/analyst/v1/sync`
  - `POST /api/analyst/v1/brief`
  - `POST /api/analyst/v1/ask`
- `health` returns service readiness, model availability, and freshness summary.
- `sync` runs an incremental refresh and returns per-domain status for prices, macro, SEC, geo-events, and model artifacts.
- `brief` returns a structured dashboard payload:
  - summary
  - top themes
  - top sectors
  - top stocks
  - analog periods
  - main risks
  - citations
  - freshness
  - abstain/forecast availability flags
- `ask` returns a structured answer payload:
  - thesis
  - prediction/verdict
  - confidence
  - proof/evidence
  - similar past cases
  - risks / critic view
  - citations
  - freshness
  - abstain reasons
- Add a stable request context shape shared by UI and backend:
  - `variant`
  - `question`
  - `ticker` or `focusedEntity`
  - `selectedCountry`
  - `watchlist`
  - `visibleLayers`
  - `as_of`
- Add a backend response mode field:
  - `forecast_ready`
  - `context_only`
  - `low_conviction`
  This prevents the UI from pretending a forecast exists when the ranking engine or freshness gate is not satisfied.

### 3. World Monitor UI integration
- Add a new panel key `analyst` to the panel registry and include it in every variant’s `panelKeys`.
- Keep existing `stock-analysis`, `stock-backtest`, and `daily-market-brief` panels unchanged in v1.
- New panel behavior:
  - top bar: status, freshness, model/provider, `Sync now`
  - dashboard section: summary, theme intensity, sector ranking, top stocks, analogs, risk watch
  - chat section: free-form analyst Q&A with response cards
  - evidence section inside each answer: citations, source freshness, analog periods, SHAP/top-driver summary when present
- Variant behavior:
  - finance: sector/stocks and expected alpha are primary
  - full: geopolitical-to-market transmission and beneficiaries/losers are primary
  - tech: semis/cyber/policy transmission emphasized
  - commodity/other variants: theme-to-sector/asset implications emphasized
- Panel loads the latest brief on first open, not a full heavy sync.
- `Sync now` triggers `sync`; after success, panel refreshes `brief`.
- Use cached last-good brief/answer state for instant render if available.

### 4. Backend completion required for UI contract
- Finish the minimum backend features needed for the UI to be trustworthy:
  - analog retrieval must return actual dated similar periods, not placeholder zeros
  - `ContextPack` must be served through the new API and include geo tables, sector rankings, and freshness
  - nightly/report outputs must be consumable as brief data
  - V1 feature/ranking outputs remain optional for v1 UI, but when absent the backend must downgrade to `context_only`
- Add a lightweight shared orchestration path for UI-triggered sync:
  - refresh raw data
  - transform
  - refresh DuckDB
  - rebuild geo context
  - refresh model artifacts only if prerequisites are satisfied
- No UI branch should ever bypass backend abstain logic.

## Test Plan
- Service lifecycle
  - World Monitor local app starts with analyst panel available.
  - Opening the panel starts or connects to the analyst service without affecting other panels.
  - Killing the analyst service and reopening the panel recovers cleanly.
- Settings integration
  - Changing Ollama model/endpoint in World Monitor settings is reflected in analyst requests without editing the finance repo `.env`.
  - Missing/invalid settings produce a visible degraded-state message, not a crash.
- Panel behavior
  - `analyst` panel appears in all variants and respects existing panel toggles/layout persistence.
  - Finance variant loads a brief with sectors/stocks/proof.
  - Full variant answers geo-finance questions with citations and analogs.
  - Existing finance premium panels still render and behave unchanged.
- Data/answer integrity
  - `brief` and `ask` responses are structured JSON and include freshness + citations.
  - stale or incomplete data forces `low_conviction` or `context_only`.
  - analogs are non-empty when similar periods exist and include dates plus outcome summaries.
  - no answer cites sources not present in the context pack.
- Failure modes
  - analyst service unavailable -> retry banner + degraded state
  - model unavailable -> proof and context still render, forecast suppressed
  - sync failure in one domain -> partial status shown, last-good brief preserved
- Regression
  - World Monitor lint/typecheck/e2e for panel registration and finance variant still pass
  - finance repo lint/tests still pass
  - local startup script still works with World Monitor + analyst stack together

## Assumptions and Defaults
- The analyst panel is enabled in all World Monitor variants from v1.
- The UI form is a new panel, not a full-screen page.
- Existing finance premium panels remain visible in v1.
- World Monitor settings own LLM/runtime config for the integrated analyst.
- The finance engine remains the source of truth for warehouse, ranking, analogs, and explanation assembly.
- The first release uses a local analyst API service plus World Monitor sidecar proxy, not a JS reimplementation of finance logic.
- The panel defaults to loading the latest cached brief on open, with manual `Sync now`; heavy sync is not forced on every app launch.

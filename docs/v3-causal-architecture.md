# V3 Causal Intelligence Backend

This document captures the current v3 implementation slice that now exists in code.

## What Was Added

- A new `ai_analyst.causal` package with:
  - composed `CausalState` types
  - version metadata
  - entity normalization helpers
  - YAML-backed causal graph assets
  - theme-regime materialization
  - pricing disagreement state
  - analog scoring
  - model interpretation packet scaffolding
- New warehouse relations:
  - `event_relations`
  - `dependency_markers`
  - `causal_state_daily`
  - `causal_chain_activations`
  - expanded `historical_analogs`
  - `forecast_outcomes`
  - `forecast_calibration_metrics`
  - `llm_override_log`
- `ContextPackBuilder` now includes:
  - `mode`
  - `causal_state`
  - `causal_chains`
  - `analog_matches`
  - `uncertainty_map`
  - `competing_hypotheses`
  - `missing_evidence`
  - `version_metadata`
- New v3 analyst surface:
  - `ai-analyst analyst research`
  - `ai-analyst analyst forecast` now uses the decision-mode orchestration path

## Current Implementation Boundary

This is a real foundation, but not the whole v3 roadmap yet.

Implemented now:

- canonical event entities and typed event relations
- seeded entity reference tables and dependency markers
- YAML causal graph loading and validation
- causal chain activation
- regime table materialization
- horizon-aware analog scoring
- research-mode role orchestration
- decision-mode synthesis plus adversarial critic

Still intentionally partial:

- calibration metric computation is not fully wired
- forecast outcome persistence is not yet populated
- override logging is not yet populated
- feature matrix does not yet include the full v3 family expansion
- World Monitor UI integration is still deferred

## Main Files

- `src/ai_analyst/causal/`
- `src/ai_analyst/llm/context_pack.py`
- `src/ai_analyst/llm/reasoning.py`
- `src/ai_analyst/events/normalization.py`
- `src/ai_analyst/warehouse/schema.py`

## Verification Status

- `ruff check src tests` passes
- `pytest -q` passes
- isolated CLI smoke run passes for:
  - `bootstrap`
  - `geo seed-defaults`
  - `snapshot`

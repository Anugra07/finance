#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
START_WORLDMONITOR=1
START_OLLAMA=1
RUN_BOOTSTRAP=1
CHECK_ONLY=0

log() {
  printf '[start] %s\n' "$*"
}

warn() {
  printf '[start] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[start] ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

port_open() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for_port() {
  local port="$1"
  local timeout="${2:-30}"
  local label="${3:-service}"
  local elapsed=0

  while (( elapsed < timeout )); do
    if port_open "$port"; then
      log "$label is listening on port $port."
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  warn "$label did not open port $port within ${timeout}s."
  return 1
}

worldmonitor_ready() {
  curl -fsS "http://localhost:3000/api/market/v1/get-sector-summary?period=1d" >/dev/null 2>&1
}

ollama_ready() {
  curl -fsS "http://localhost:11434/api/tags" >/dev/null 2>&1
}

wait_for_check() {
  local check_name="$1"
  local timeout="${2:-30}"
  local elapsed=0

  while (( elapsed < timeout )); do
    if "$check_name"; then
      log "$check_name passed."
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  warn "$check_name did not pass within ${timeout}s."
  return 1
}

usage() {
  cat <<'EOF'
Usage: ./start.sh [options]

Options:
  --skip-worldmonitor  Do not start the World Monitor Docker stack.
  --skip-ollama        Do not start the local Ollama server.
  --skip-bootstrap     Do not run the analyst bootstrap command.
  --check-only         Only print status checks; do not start anything.
  --help               Show this help text.
EOF
}

while (($#)); do
  case "$1" in
    --skip-worldmonitor)
      START_WORLDMONITOR=0
      ;;
    --skip-ollama)
      START_OLLAMA=0
      ;;
    --skip-bootstrap)
      RUN_BOOTSTRAP=0
      ;;
    --check-only)
      CHECK_ONLY=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

ensure_python_ready() {
  have "$PYTHON_BIN" || fail "Python interpreter not found: $PYTHON_BIN"
  if ! "$PYTHON_BIN" -c "import ai_analyst" >/dev/null 2>&1; then
    fail "Python cannot import ai_analyst. Install the repo dependencies first."
  fi
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    warn "No .env file found at $ROOT_DIR/.env"
  fi
}

bootstrap_workspace() {
  if (( RUN_BOOTSTRAP == 0 )); then
    log "Skipping bootstrap."
    return
  fi
  log "Bootstrapping analyst workspace."
  "$PYTHON_BIN" -m ai_analyst.cli.app bootstrap >/dev/null
}

start_worldmonitor() {
  if (( START_WORLDMONITOR == 0 )); then
    log "Skipping World Monitor."
    return
  fi
  if ! have docker; then
    warn "Docker is not installed; cannot start World Monitor."
    return
  fi
  if [[ ! -d "$ROOT_DIR/worldmonitor-forked" ]]; then
    warn "worldmonitor-forked is missing; cannot start World Monitor."
    return
  fi
  if worldmonitor_ready; then
    log "World Monitor is already responding on localhost:3000."
    return
  fi
  if port_open 3000; then
    warn "Port 3000 is already in use by another process; skipping World Monitor start."
    return
  fi

  log "Starting World Monitor Docker stack."
  (
    cd "$ROOT_DIR/worldmonitor-forked"
    docker compose up -d
  )
  wait_for_port 3000 30 "World Monitor" || true
  wait_for_check worldmonitor_ready 90 || true
}

start_ollama() {
  if (( START_OLLAMA == 0 )); then
    log "Skipping Ollama."
    return
  fi
  if ! have ollama; then
    warn "Ollama CLI is not installed; cannot start Ollama."
    return
  fi
  if ollama_ready; then
    log "Ollama is already responding on localhost:11434."
    return
  fi
  if port_open 11434; then
    warn "Port 11434 is already in use by another process; skipping Ollama start."
    return
  fi

  log "Starting Ollama server."
  if [[ -d "/Applications/Ollama.app" ]]; then
    open -ga Ollama
  else
    nohup ollama serve >/tmp/local-ai-analyst-ollama.log 2>&1 &
    disown || true
  fi
  wait_for_port 11434 10 "Ollama" || true
  wait_for_check ollama_ready 30 || true
}

print_status() {
  log "Status summary:"

  if worldmonitor_ready; then
    log "  World Monitor: up (http://localhost:3000)"
  elif port_open 3000; then
    warn "  Port 3000 is open, but World Monitor did not answer the expected endpoint."
  else
    log "  World Monitor: down"
  fi

  if ollama_ready; then
    log "  Ollama: up (http://localhost:11434)"
  elif port_open 11434; then
    warn "  Port 11434 is open, but Ollama did not answer /api/tags."
  else
    log "  Ollama: down"
  fi

  log "  DuckDB: $ROOT_DIR/data/warehouse/analyst.duckdb"
  log "Next commands:"
  log "  PYTHONPATH=src $PYTHON_BIN -m ai_analyst.cli.app db freshness"
  log "  PYTHONPATH=src $PYTHON_BIN -m ai_analyst.cli.app analyst ollama-status"
  log "  PYTHONPATH=src $PYTHON_BIN -m ai_analyst.cli.app collect worldmonitor"
}

main() {
  ensure_python_ready
  if (( CHECK_ONLY == 0 )); then
    bootstrap_workspace
    start_worldmonitor
    start_ollama
  fi
  print_status
}

main "$@"

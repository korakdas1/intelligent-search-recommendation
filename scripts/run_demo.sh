#!/usr/bin/env bash
# Start the local API and demo UI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8000}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
echo "Demo: http://${HOST}:${PORT}/demo"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"

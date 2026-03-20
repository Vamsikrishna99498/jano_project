#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "[1/6] Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found."
  exit 1
fi

echo "[2/6] Creating virtual environment (if missing)..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

echo "[3/6] Installing Python dependencies..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "[4/6] Preparing .env..."
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from template."
fi

echo "[5/6] Checking PostgreSQL availability..."
if command -v pg_isready >/dev/null 2>&1; then
  if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo "PostgreSQL is not reachable on localhost:5432."
    echo "Start PostgreSQL and update DATABASE_URL in .env, then rerun this script."
    exit 1
  fi
else
  echo "pg_isready not found. Skipping active PostgreSQL health check."
fi

echo "[6/6] Setup complete."
echo "Next steps:"
echo "  1) Open .env and set DATABASE_URL with your postgres password"
echo "  2) Run: source .venv/bin/activate"
echo "  3) Run: python -m streamlit run app.py"

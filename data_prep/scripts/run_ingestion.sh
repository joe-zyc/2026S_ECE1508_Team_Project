#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
ENV_FILE="${PROJECT_ROOT}/.env"

cd "$PROJECT_ROOT"

if [[ ! -f "amazon_products_with_main_category.csv" ]]; then
  echo "CSV file not found: ${PROJECT_ROOT}/amazon_products_with_main_category.csv" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not available on PATH. Start Docker Desktop and enable WSL integration if needed." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f "$ENV_FILE" && -f ".env.example" ]]; then
  cp .env.example .env
fi

docker compose up -d

python scripts/ingest_amazon_products.py "$@"

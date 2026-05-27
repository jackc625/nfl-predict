# NFL Prediction System Makefile
#
# Thin convenience wrapper over the canonical run sequence documented in PIPELINE.md.
# Each target's recipe is the LITERAL `uv run` command from PIPELINE.md so the Makefile
# and the documentation never drift. PIPELINE.md is the source of truth; this file is
# portfolio/convenience polish only.
#
# NOTE: Windows ships no native `make`. Correctness is measured against the documented
# `uv run` commands run directly in PowerShell, not against `make` execution. Run any
# recipe below by copying its `uv run ...` line into PowerShell.

.PHONY: train backtest backtest-blend predict build-cache serve friday-production test lint

# Season/week overrides for the predict target (e.g. `make predict SEASON=2024 WEEK=6`).
SEASON ?= 2024
WEEK ?= 1

# Stage 3 -- Train all models (WP, ATS, O/U) with walk-forward temporal validation.
train:
	uv run python scripts/train_models.py --target all

# Stage 4 -- Walk-forward backtest across 2021-2024 with interactive HTML report.
backtest:
	uv run python scripts/run_backtest.py

# Stage 4 (blended) -- Backtest with market blending applied.
backtest-blend:
	uv run python scripts/run_backtest.py --blend

# Stage 5 -- Generate predictions for a specific season/week (override SEASON / WEEK).
predict:
	uv run python scripts/generate_current_week_predictions.py --season $(SEASON) --week $(WEEK)

# Stage 6 -- (Re)build the read-only DuckDB web cache the API serves from.
build-cache:
	uv run python scripts/populate_cache.py

# Stage 7 -- Start the FastAPI app + web UI (single worker envelope).
serve:
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Automation subset -- the Friday orchestrator (current-week data + predictions).
friday-production:
	uv run python scripts/friday_pipeline.py --log-level INFO

# Run the test suite (unit + integration + api).
test:
	uv run pytest tests/unit tests/integration tests/api -q

# Lint: Ruff check + format check.
lint:
	uv run ruff check . && uv run ruff format --check .

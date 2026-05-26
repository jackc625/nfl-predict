# NFL Prediction System - Deployment (Windows Scheduling)

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. It is intentionally narrow: it covers only how the surviving Friday automation is
> scheduled on Windows. See `.planning/` for the authoritative project documentation, architecture
> decisions, and current development state. A full runbook is owned by a later phase.

This directory holds the Windows scheduling setup for the automated Friday pipeline. There is one
automation story: `scripts/friday_pipeline.py` (the weekly orchestrator) triggered by Windows Task
Scheduler. Nothing else lives here -- no container, hosting, or other-OS scheduling configuration.

## Files

- **`setup_scheduling.py`** - Windows Task Scheduler setup utility for the Friday run.
- **`windows_scheduler.xml`** - Windows Task Scheduler task definition for the Friday run.

## What the Friday automation does

The weekly run is the orchestrator `scripts/friday_pipeline.py`, which ingests the current week's
data, builds features, validates models, and generates predictions (WP, ATS, O/U). It is aligned to
the Friday 6:00 PM ET odds-snapshot timing. (Phase 21 verifies the orchestrator end-to-end and the
runbook is written in a later phase; this README only documents the scheduling step.)

## Setting up Windows Task Scheduler

```bash
# Dry run to see what would be created
python deployment/setup_scheduling.py --platform windows --dry-run

# Install the scheduled task (requires administrator privileges)
python deployment/setup_scheduling.py --platform windows --install
```

The task definition installed by the command above is `deployment/windows_scheduler.xml`.

For everything else (running each pipeline stage by hand, end-to-end ordering, troubleshooting), see
the project documentation in `.planning/`.

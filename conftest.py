"""Repo-root conftest: cap the thread pools before any numerical library loads.

WHY THIS FILE EXISTS
--------------------
The trainers set ``n_jobs=-1`` at 22 sites (``models/trainers/ats_trainer.py``,
``models/trainers/ou_trainer.py``, ``models/train_ats.py``, ``models/train_ou.py``
and siblings). ``-1`` means "use every core", which is correct for a real training
run and wrong for a test session: ``tests/unit/test_{wp,ats,ou}_trainer.py``,
``tests/unit/test_prediction_pipeline.py`` and ``tests/unit/test_tuning.py`` all
perform real XGBoost and scikit-learn fits, and the session-scoped
``p31_rehearsal_run`` fixture runs a fifteen-cell tune sweep. Each fit took all
twelve cores, so a suite run made the machine unusable for its whole duration.

Nothing in the project capped this -- a grep for any of the variables below
returned nothing outside ``.venv/``.

WHY THE REPO ROOT, AND NOT ``tests/conftest.py``
------------------------------------------------
BLAS reads its thread count ONCE, when numpy is first imported, and ignores later
changes. ``tests/conftest.py`` imports numpy at module scope, so setting these
there would already be too late. pytest loads the rootdir conftest before any
package conftest, which is early enough. ``_warn_if_too_late`` below asserts that
rather than assuming it -- if numpy is somehow already imported, the session says
so out loud instead of silently capping nothing.

This file deliberately imports nothing but the standard library.
"""

from __future__ import annotations

import os
import sys
import warnings

#: Cores the test session may use. Owner-chosen 2026-09-12 (8 of 12), leaving
#: four for the machine to stay usable while a suite runs.
DEFAULT_TEST_THREADS = 8

#: Every pool that would otherwise size itself to the whole machine.
#: ``OMP_NUM_THREADS`` covers XGBoost and most BLAS builds; ``LOKY_MAX_CPU_COUNT``
#: is the one scikit-learn's ``n_jobs=-1`` actually consults, via joblib/loky.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)

#: Set this to override the cap. ``0`` means "uncapped" -- use every core, the
#: pre-2026-09-12 behaviour -- which is what a benchmark run wants.
OVERRIDE_ENV_VAR = "NFL_TEST_THREADS"


def _resolve_thread_cap() -> int | None:
    """The cap to apply, or None to leave every pool alone.

    An explicit ``NFL_TEST_THREADS`` always wins, including ``0`` for uncapped.
    A non-numeric value is a warning and a fall back to the default, never a
    crash: a typo in an env var must not be able to fail a test session.
    """
    raw = os.environ.get(OVERRIDE_ENV_VAR)
    if raw is None:
        return DEFAULT_TEST_THREADS

    try:
        requested = int(raw)
    except ValueError:
        warnings.warn(
            f"{OVERRIDE_ENV_VAR}={raw!r} is not an integer; falling back to the "
            f"default cap of {DEFAULT_TEST_THREADS}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_TEST_THREADS

    if requested <= 0:
        return None
    return min(requested, os.cpu_count() or requested)


def _warn_if_too_late() -> None:
    """Say so if numpy beat us to it -- a silent no-op is the failure to avoid."""
    if "numpy" in sys.modules:
        warnings.warn(
            "numpy was imported before the repo-root conftest ran, so the BLAS "
            "thread cap did not take effect. The pytest rootdir may have moved, "
            "or a plugin imported numpy first. Check with: "
            "python -c \"import os; os.environ['OMP_NUM_THREADS']\"",
            RuntimeWarning,
            stacklevel=2,
        )


def _apply_thread_cap() -> None:
    """Set each pool's ceiling, without overriding a value already chosen.

    ``setdefault`` rather than assignment: a caller who exported
    ``OMP_NUM_THREADS=2`` for their own reasons meant it, and a test harness
    silently overruling a deliberate environment is its own bug.
    """
    cap = _resolve_thread_cap()
    if cap is None:
        return

    _warn_if_too_late()
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(cap))


_apply_thread_cap()

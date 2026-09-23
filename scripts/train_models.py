"""Train models -- thin wrapper for models.train, plus THE THREAD PIN.

Usage:
    python -m scripts.train_models --target all
    python -m scripts.train_models --all-targets --tune --gold-generation <digest>

The canonical invocation remains:
    python -m models.train --target all

All flags (--target, --all-targets, --tune, --thread-limit, --artifacts-dir, --no-clv,
--config-train-seasons, ...) are owned and parsed by models.train.build_parser; this
wrapper adds no argument parsing of its own beyond PEEKING at two of them.

WHY THE PIN LIVES HERE AND NOT IN models/train.py
-------------------------------------------------
MEASURED by Plan 33.2-22: the XGBoost legs return DIFFERENT answers at different OpenMP
thread counts, by enough to move a verdict (12 threads and the 8-thread pytest cap
disagreed). A number produced under an unrecorded, unpinned thread count is a number nobody
else can reproduce, so every number Plan 33.2-23 produces -- both search arms, the outer
comparison and the final fits -- runs pinned.

The pin has TWO halves and they must happen at two different moments:

  * the ENVIRONMENT VARIABLES (OMP_NUM_THREADS and its OpenBLAS / MKL / NumExpr siblings)
    are read by those libraries when they are first LOADED, so they must be set BEFORE
    numpy, sklearn or xgboost is imported. A module that already imported numpy cannot set
    them retroactively -- and ``models/train.py`` imports numpy at its top. That is the
    whole reason this wrapper is not a one-line re-export any more;
  * ``threadpoolctl.threadpool_limits`` reaches the ALREADY-LOADED pools at runtime and
    covers anything the environment missed.

Both are applied. The value is also passed through to ``models.train``, which records it in
each artifact's metadata -- a pin nobody can read afterwards is not reproducibility, it is
an unverifiable claim.

Omitting ``--thread-limit`` on a run that is not ``--tune`` leaves the pool exactly as it
was, so every existing caller of this wrapper is byte-identical.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import sys

#: The environment variables each numerical backend reads AT LOAD TIME.
_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_thread_limit(argv: list[str]) -> int | None:
    """PEEK at argv for the thread pin, without parsing the rest.

    Deliberately a peek and not a parse: ``models.train.build_parser`` owns the argv
    surface, and a second parser here would be a second contract that can drift. All this
    needs to know, before numpy is imported, is one integer.

    ``--tune`` implies the pre-registered pin
    (``config.tuning_preregistration.PINNED_THREAD_COUNT``) when no explicit value is
    given, so the re-fit cannot be run unpinned by forgetting a flag.

    Args:
        argv: The command line, without the program name.

    Returns:
        The thread count to pin, or None to leave the pools alone.
    """
    for index, token in enumerate(argv):
        if token == "--thread-limit" and index + 1 < len(argv):
            return int(argv[index + 1])
        if token.startswith("--thread-limit="):
            return int(token.split("=", 1)[1])
    if "--tune" in argv:
        # Imported HERE rather than at module scope: the pre-registration imports only the
        # standard library and conf.season_partition, so this cannot pull in numpy before
        # the environment variables below are set.
        from config.tuning_preregistration import PINNED_THREAD_COUNT

        return PINNED_THREAD_COUNT
    return None


def main() -> None:
    """Apply the thread pin, then hand over to ``models.train.main``."""
    thread_limit = resolve_thread_limit(sys.argv[1:])

    if thread_limit is not None:
        for variable in _THREAD_ENV_VARS:
            os.environ[variable] = str(thread_limit)

    # Imported AFTER the environment is set, which is the entire point of this wrapper.
    from threadpoolctl import threadpool_limits

    from models.train import main as train_main

    if thread_limit is None:
        train_main()
        return

    print(f"  Thread pool PINNED at {thread_limit} thread(s) for this run.")
    with threadpool_limits(limits=thread_limit):
        train_main()


if __name__ == "__main__":
    main()

"""The utils and api packages no longer drag the scientific stack or the app into every importer.

Phase 36 D-11 (SPEC R1). Every page request imports three small utils modules (``get_logger``,
``utils.date_utils``, ``utils.team_data``). The old eager ``utils/__init__.py`` imported fifteen
submodules, and with them pandas, numpy, pyarrow and ``conf``, into each of those imports. It now
resolves its re-exported names lazily (PEP 562). ``api/__init__.py`` used to import ``.main``, so
``import api.services`` from ``notify/`` built the whole app; it now imports nothing.

These tests check what a fresh interpreter ACTUALLY loads (``tests/api/test_import_guard_serving.py``
walks the whole request path; this file pins the two package-level changes), and that every name in
``utils.__all__`` still resolves to the same object it did before.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCIENTIFIC_ROOTS = frozenset({"pandas", "numpy", "pyarrow", "scipy"})


def _fresh_interpreter(probe: str) -> list[str]:
    """Run *probe* in a fresh interpreter at the repo root and return the JSON list it prints.

    Not ``-I``: isolated mode drops the working directory from ``sys.path``, so ``utils`` and
    ``api`` would not import at all.
    """
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_a_utils_import_loads_no_scientific_stack() -> None:
    """The three utils imports the request path makes load none of pandas, numpy, pyarrow, scipy."""
    probe = (
        "import json, sys\n"
        "from utils import get_logger\n"
        "import utils.date_utils, utils.team_data\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    loaded = set(_fresh_interpreter(probe))
    assert not loaded & SCIENTIFIC_ROOTS, (
        f"a request-path utils import loaded {sorted(loaded & SCIENTIFIC_ROOTS)}"
    )


def test_every_all_name_resolves_to_its_source_object() -> None:
    """The lazy map covers exactly ``__all__`` and each name is its submodule's own object."""
    import utils

    lazy_map: dict[str, str] = utils._LAZY  # pyright: ignore[reportPrivateUsage]
    assert set(lazy_map) == set(utils.__all__), (
        f"lazy map and __all__ differ: {sorted(set(lazy_map) ^ set(utils.__all__))}"
    )
    for name, module_name in lazy_map.items():
        source = importlib.import_module(module_name, "utils")
        assert getattr(utils, name) is getattr(source, name), (
            f"utils.{name} is not utils{module_name}.{name}"
        )
    assert set(utils.__all__) <= set(dir(utils))


def test_importing_api_services_does_not_build_the_app() -> None:
    """``import api.services`` (what notify/ does) no longer imports ``api.main``."""
    probe = "import json, sys\nimport api.services\nprint(json.dumps(sorted(sys.modules)))\n"
    loaded = set(_fresh_interpreter(probe))
    assert "api.main" not in loaded, (
        "importing api.services still builds the FastAPI app"
    )

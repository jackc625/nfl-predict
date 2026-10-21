"""Control for RFIT-01: the first-re-fit ancestry arm FAILS on a dirty or missing code commit.

The production arm skips until the first install_intent exists, so on its own it never proves
its dirty/null assertions can fire. This control feeds it synthetic intents (no ledger read).
"""

from __future__ import annotations

import sys

import pytest

from tests.unit import test_phase37_preregistration_ancestry as ancestry


@pytest.mark.parametrize("bad_commit", [None, "a" * 40 + "+dirty"])
def test_a_dirty_or_null_first_refit_commit_fails(
    monkeypatch: pytest.MonkeyPatch, bad_commit: str | None
) -> None:
    intent = {
        "event": "install_intent",
        "date": "2026-11-03",
        "code_commit": bad_commit,
    }
    monkeypatch.setattr(
        sys.modules[ancestry.__name__], "first_refit_intent", lambda _ledger_dir: intent
    )
    with pytest.raises(AssertionError):
        ancestry.test_the_first_refit_descends_from_the_record()

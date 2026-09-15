"""The cold-start derivation is a DETERMINISTIC, DIGEST-REFUSING, PROVENANCE-RECORDING program.

Phase 33, Plan 33-16 Task 2 (COLD-07, CLEAN-01).

WHY A COMMITTED DERIVATION PROGRAM NEEDS ITS OWN TESTS
--------------------------------------------------------
The point of ``scripts/derive_cold_start_constants.py`` is that a future reviewer can check
more than arithmetic: which artifacts were scored, which rows were eligible, which quantile
convention produced a threshold. That claim is only worth anything if the program is actually
deterministic and actually refuses on a moved input. Both are asserted here rather than
assumed.

THREE PROPERTIES, EACH WITH ITS OWN CLASS
-------------------------------------------
1. DETERMINISM -- two runs against the same inputs produce byte-identical output. A derivation
   that drifted between runs could not anchor anything, because the committed bytes would not
   be reproducible from the committed program.
2. THE DIGEST-MISMATCH REFUSAL -- a declared digest that does not match the file on disk stops
   the run BY NAME, naming the input and BOTH digests. An artifact that moved between the gate
   and the derivation is a DIFFERENT measurement, and the refusal is what keeps a threshold
   from being frozen against one.
3. THE PROVENANCE CONSTANTS -- the emitted module parses and carries the quantile convention,
   the input digests and the eligible counts. Emitting the numbers without them would leave a
   reviewer unable to re-run anything.

NO TEST HERE WRITES INTO ``data/``, ``outputs/`` OR ``artifacts/``. The two runs emit into a
pytest ``tmp_path``; the production tree is opened read-only.

Run this module:  uv run pytest tests/unit/test_cold_start_derivation_cli.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backtest import cold_start_constants as frozen
from scripts import derive_cold_start_constants as cli
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

# The END-STATE artifacts the committed pre-registration was derived from. Read from the
# emitted module rather than typed here, so this test cannot disagree with the rule it checks.
ARTIFACTS: dict[str, str] = dict(frozen.DERIVATION_ARTIFACTS)


def _declared_digest_args() -> list[str]:
    """The ``--digest`` arguments naming every canonical input at its CURRENT digest."""
    measured = cli.measure_input_digests(ARTIFACTS, REPO_ROOT)
    args: list[str] = []
    for key, value in measured.items():
        args.extend(["--digest", f"{key}={value}"])
    return args


def _base_args(destination: Path) -> list[str]:
    """The invocation every test shares, emitting into *destination* and never into the tree."""
    return [
        "--wp-artifact",
        ARTIFACTS["wp"],
        "--ats-artifact",
        ARTIFACTS["ats"],
        "--ou-artifact",
        ARTIFACTS["ou"],
        "--repo-root",
        str(REPO_ROOT),
        "--out-module",
        str(destination / "constants.py"),
        "--out-doc",
        str(destination / "document.md"),
    ]


class TestTheDerivationIsDeterministic:
    """Two runs, same inputs, byte-identical output."""

    def test_two_runs_emit_byte_identical_files(self, tmp_path: Path) -> None:
        digests = _declared_digest_args()
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()

        for destination in (first, second):
            assert cli.main([*_base_args(destination), *digests]) == 0

        for name in ("constants.py", "document.md"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), (
                f"two runs of the derivation emitted DIFFERENT bytes for {name}. A "
                "non-deterministic derivation cannot anchor a pre-registration: the committed "
                "bytes would not be reproducible from the committed program."
            )

    def test_the_second_run_reproduces_the_committed_module(
        self, tmp_path: Path
    ) -> None:
        """The committed rule file is what this program emits TODAY, byte for byte.

        This is the assertion that turns "the derivation is committed" into "the derivation
        produced what is committed". If it ever fails, either an input moved or somebody
        hand-edited a frozen rule -- and hand-editing it destroys the evidence rather than
        fixing anything.
        """
        assert cli.main([*_base_args(tmp_path), *_declared_digest_args()]) == 0
        emitted = (tmp_path / "constants.py").read_bytes()
        committed = (REPO_ROOT / cli.MODULE_PATH).read_bytes().replace(b"\r\n", b"\n")
        assert emitted == committed, (
            "the committed backtest/cold_start_constants.py is NOT what the derivation emits "
            "from today's inputs. Either an input moved since the freeze, or the frozen rule "
            "was edited in place -- and an edit in place destroys the evidence rather than "
            "fixing anything. The remedy is a NEW, visibly-later corrective commit that names "
            "the superseded commit sha."
        )


class TestADigestMismatchRefusesByName:
    """An input that moved between the gate and the derivation is a DIFFERENT measurement."""

    def test_a_wrong_artifact_digest_refuses(self, tmp_path: Path) -> None:
        digests = _declared_digest_args()
        key = f"artifact:{ARTIFACTS['wp']}"
        index = digests.index(
            next(entry for entry in digests if entry.startswith(f"{key}="))
        )
        digests[index] = f"{key}={'0' * 64}"

        with pytest.raises(cli.DigestMismatchError) as excinfo:
            cli.main([*_base_args(tmp_path), *digests])

        message = str(excinfo.value)
        assert key in message, message
        assert "0" * 64 in message, "the refusal must name the DECLARED digest"
        assert "declared" in message and "measured" in message, (
            "the refusal must name BOTH digests, or a reader cannot tell which side moved."
        )

    def test_the_refusal_happens_before_any_scoring(self, tmp_path: Path) -> None:
        """A run that is going to refuse refuses in a second, not after three model loads.

        Proven structurally rather than by timing: the emitted files must NOT exist after the
        refusal, so nothing downstream of the verification ran.
        """
        digests = _declared_digest_args()
        key = "config/phase33_gate_verdict.toml"
        index = digests.index(
            next(entry for entry in digests if entry.startswith(f"{key}="))
        )
        digests[index] = f"{key}={'f' * 64}"

        with pytest.raises(cli.DigestMismatchError):
            cli.main([*_base_args(tmp_path), *digests])
        assert not (tmp_path / "constants.py").exists()
        assert not (tmp_path / "document.md").exists()

    def test_an_undeclared_input_is_refused_rather_than_trusted(
        self, tmp_path: Path
    ) -> None:
        """Every canonical input must carry a declared digest. Silence is not consent."""
        with pytest.raises(SystemExit) as excinfo:
            cli.main(
                [*_base_args(tmp_path), "--digest", "artifacts/latest.json=" + "0" * 64]
            )
        assert "missing" in str(excinfo.value)


class TestTheEmittedModuleCarriesItsProvenance:
    """The emitted module parses, and carries all three provenance constants."""

    def test_the_committed_module_parses(self) -> None:
        source = (REPO_ROOT / cli.MODULE_PATH).read_text(encoding="utf-8")
        assert ast.parse(source) is not None

    def test_it_carries_the_quantile_convention_the_input_digests_and_the_counts(
        self,
    ) -> None:
        assert frozen.THRESHOLD_QUANTILE_CONVENTION.strip(), (
            "the quantile convention must be NAMED, not left to a library default that could "
            "change on an upgrade and silently move a frozen threshold."
        )
        assert "method=" in frozen.THRESHOLD_QUANTILE_CONVENTION
        assert len(frozen.DERIVATION_INPUT_DIGESTS) >= 9, (
            "every input the derivation read must be recorded with its digest."
        )
        assert set(frozen.DERIVATION_ELIGIBLE_COUNTS) == {"wp", "ats", "ou"}
        for target, counts in frozen.DERIVATION_ELIGIBLE_COUNTS.items():
            assert counts["threshold_rows"] > 0, target
            assert counts["bias_rows"] > 0, target

    def test_every_recorded_input_digest_still_matches_the_tree(self) -> None:
        """The recorded provenance is CROSS-CHECKED against the stores, never trusted alone.

        A provenance record nobody verifies is a comment. This is also the check that would
        catch gold or an artifact moving after the freeze -- which is not a defect in this
        plan, but IS something a later reader must be told about rather than left to assume.
        """
        measured = cli.measure_input_digests(frozen.DERIVATION_ARTIFACTS, REPO_ROOT)
        assert measured == dict(frozen.DERIVATION_INPUT_DIGESTS), (
            "an input recorded in DERIVATION_INPUT_DIGESTS no longer hashes to its recorded "
            "value. The frozen numbers were derived from something the tree no longer holds."
        )

    def test_the_manifest_digest_agrees_with_the_committed_end_state(self) -> None:
        """One file, one instrument: the derivation and the gate record agree BY VALUE."""
        assert (
            frozen.DERIVATION_INPUT_DIGESTS["artifacts/latest.json"]
            == phase33_state.POST_GATE_MANIFEST_DIGEST
        ), (
            "the derivation's digest for artifacts/latest.json does not equal "
            "tests.phase33_state.POST_GATE_MANIFEST_DIGEST. Either the manifest moved after "
            "the promotion, or the two records used different instruments -- and a content "
            "hash compared against a differently-computed one says nothing about the bytes."
        )

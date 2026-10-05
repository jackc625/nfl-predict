"""The ledger's own copies of every artifact a row references (D-02, D-11, LDGR-09).

WHY THE LEDGER KEEPS COPIES
---------------------------
The private backup must let the ledger be restored, verified AND replayed from the backup alone
(D-02). ``artifacts/`` is gitignored and its manifest is overwritten by every swap, so a row that
named a model id without carrying the model would replay only for as long as nobody pruned that
directory. The ledger directory is therefore SELF-CONTAINED: on first reference every artifact
directory a decision was scored by is copied into ``ledger/artifacts/<id>/`` -- the three models,
the blend, AND the converter the blend is bound to, because the WP second test reads that
converter's slope and a blend-only copy would break backup-only replay (34-RESEARCH Pitfall 16).
The chain-fit record the bet rule reads is gitignored generator output too, so it is kept as
``ledger/recipes/<sha256>.json``, addressed by the digest the recipe registry records.

VERIFIED BY TREE DIGEST, AT COPY TIME AND ON EVERY REUSE
--------------------------------------------------------
:func:`tree_digest` is sha256 over every file's sorted relative POSIX path and content sha256. A
copy is staged privately, proven equal to its source, and only then renamed into place, so a crash
never leaves a half copy under a real id. A later reference re-verifies the existing copy against
the source and refuses a drifted one by name (:class:`ArtifactCopyMismatchError`): the copy must
equal what scored the row. Copies are written BEFORE the row that references them and are
idempotent (34-RESEARCH F1).

READ-ONLY on ``artifacts/``. Every write is asked of ``data.write_sink.current_sink()`` first.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from data.write_sink import current_sink
from models.artifacts import ResolvedArtifacts

__all__ = [
    "ARTIFACT_COPIES_DIRNAME",
    "RECIPE_COPIES_DIRNAME",
    "ArtifactCopyMismatchError",
    "ensure_artifact_copies",
    "ensure_recipe_record_copy",
    "referenced_artifact_ids",
    "tree_digest",
]

ARTIFACT_COPIES_DIRNAME: str = "artifacts"
RECIPE_COPIES_DIRNAME: str = "recipes"

# Domain separation: a tree digest can never equal a file digest or a chain link by accident.
TREE_DOMAIN = b"nfl-ledger-tree-v1\n"


class ArtifactCopyMismatchError(Exception):
    """A ledger copy differs from the artifact (or record) it must equal; named in the message.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a drifted copy must never be
    degraded into "nothing to copy".
    """


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path | str) -> str:
    """sha256 over every file under *path*: sorted relative POSIX paths and per-file sha256.

    Raises:
        FileNotFoundError: *path* is not a directory.
    """
    root = Path(path)
    if not root.is_dir():
        msg = f"cannot digest {root.as_posix()}: not a directory"
        raise FileNotFoundError(msg)
    entries = sorted(
        (file.relative_to(root).as_posix(), _sha256(file))
        for file in root.rglob("*")
        if file.is_file()
    )
    body = "\n".join(f"{relative}={digest}" for relative, digest in entries)
    return hashlib.sha256(TREE_DOMAIN + body.encode("utf-8")).hexdigest()


def referenced_artifact_ids(resolved: ResolvedArtifacts) -> list[str]:
    """Every artifact id one decision was scored by: three models, the blend, its converter."""
    ids = [resolved.wp, resolved.ats, resolved.ou, resolved.blend]
    if resolved.converter is not None:
        ids.append(resolved.converter)
    return ids


def _require_plain_name(artifact_id: str) -> None:
    if (
        not artifact_id
        or artifact_id in (".", "..")
        or Path(artifact_id).name != artifact_id
    ):
        msg = f"{artifact_id!r} is not a plain artifact directory name"
        raise ArtifactCopyMismatchError(msg)


def ensure_artifact_copies(
    ledger_dir: Path | str,
    resolved: ResolvedArtifacts,
    artifacts_dir: Path = Path("artifacts"),
) -> list[str]:
    """Copy every artifact *resolved* names into ``<ledger_dir>/artifacts/<id>/``, once each.

    Args:
        ledger_dir: The ledger directory.
        resolved: The ids the decision was scored by (Plan 34-05).
        artifacts_dir: The production artifacts root (read only).

    Returns:
        The ids copied by THIS call; an id already present and verified is not repeated.

    Raises:
        FileNotFoundError: a referenced artifact directory is absent from *artifacts_dir*.
        ArtifactCopyMismatchError: an existing copy, or a fresh one, differs from its source.
    """
    copies_root = Path(ledger_dir) / ARTIFACT_COPIES_DIRNAME
    copied: list[str] = []
    for artifact_id in referenced_artifact_ids(resolved):
        _require_plain_name(artifact_id)
        source = Path(artifacts_dir) / artifact_id
        if not source.is_dir():
            msg = (
                f"artifact {artifact_id!r} is absent from {Path(artifacts_dir).as_posix()}; the "
                "ledger cannot keep a copy of an artifact it cannot read"
            )
            raise FileNotFoundError(msg)
        expected = tree_digest(source)
        destination = copies_root / artifact_id
        if destination.exists():
            if tree_digest(destination) != expected:
                msg = (
                    f"the ledger's copy of {artifact_id!r} differs from the artifact that scored "
                    "the row; it is refused, never overwritten"
                )
                raise ArtifactCopyMismatchError(msg)
            continue
        if not current_sink().authorize(str(destination), "ledger_artifact_copy"):
            continue

        copies_root.mkdir(parents=True, exist_ok=True)
        staging = copies_root / f".staging-{artifact_id}-{uuid.uuid4().hex}"
        try:
            shutil.copytree(source, staging)
            if tree_digest(staging) != expected:
                msg = f"the staged copy of {artifact_id!r} does not equal its source"
                raise ArtifactCopyMismatchError(msg)
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        copied.append(artifact_id)
    return copied


def ensure_recipe_record_copy(
    ledger_dir: Path | str, chain_fit_path: Path | str
) -> str:
    """Keep the chain-fit record as ``<ledger_dir>/recipes/<sha256>.json``; return the sha256.

    Raises:
        ArtifactCopyMismatchError: an existing copy under that name does not hash to it.
    """
    data = Path(chain_fit_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    destination = Path(ledger_dir) / RECIPE_COPIES_DIRNAME / f"{sha}.json"
    if destination.exists():
        if _sha256(destination) != sha:
            msg = f"the ledger's copy of the chain-fit record {sha} no longer hashes to its name"
            raise ArtifactCopyMismatchError(msg)
        return sha
    if not current_sink().authorize(str(destination), "ledger_recipe_copy"):
        return sha

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sha

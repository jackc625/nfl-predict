"""Publish an Elo GENERATION: five artifacts staged together, one pointer move.

WHY THIS IS ITS OWN MODULE
--------------------------
FIVE INDEPENDENTLY-ATOMIC FILES ARE NOT ONE ATOMIC STATE (T-33-16d). The old
``EloBuilder.save_results`` wrote five artifacts with five separate calls. Each
individual write was atomic; the SET was not, so a crash between any two of them left
snapshots from the new build sitting beside a rating history from the old one, with
nothing on disk saying so. Every consumer downstream would have read that mixture as a
single coherent Elo state, because from the outside it is indistinguishable from one.

Publishing a generation is a self-contained concern with its own vocabulary -- staging,
validation, a pointer -- and both write verbs in ``scripts/build_elo.py`` route through
it. It lives here rather than inside the builder so that the builder stays about
building ratings and this module stays about publishing them atomically.

THE SPLIT RULE LIVES HERE TOO, because validation is what enforces it: a ROW table
accumulates history and is UPSERTED, a STATE artifact is current-state by definition and
is REPLACED. ``scripts/build_elo`` re-exports both tuples, so callers and tests have one
name to import either way.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# The Elo artifact split rule.
#
# A ROW TABLE accumulates history: one row per game, keyed on ``game_id``, and a
# weekly run must UPSERT into it so that 24 seasons of burn-in survive the write.
# A STATE ARTIFACT is current-state by definition -- it describes where the ratings
# stand right now -- so it is REPLACED, and replacing it loses nothing.
#
# Getting this split wrong in either direction is a real failure: upserting a state
# artifact would leave stale teams behind forever, and replacing a row table would
# destroy the chain the deployed WP model reads.
# ---------------------------------------------------------------------------

ELO_ROW_TABLES: tuple[str, ...] = (
    "elo_game_snapshots",
    "games_with_elo",
    "elo_rating_history",
)

ELO_STATE_ARTIFACTS: tuple[str, ...] = ("elo_ratings_current", "elo_ratings")

# Every row table is ONE ROW PER GAME, so ``game_id`` alone is its identity and
# ``upsert_silver`` (latest-wins on a single key) is the right verb for all three.
# ``upsert_silver_composite`` exists for the ``odds_timeline`` trajectory grain, where
# a game legitimately carries many rows; none of these three does.
ELO_ROW_TABLE_KEY_COLUMN: str = "game_id"

# Elo burn-in starts in 2002: 16 seasons before the first backtest season (2018).
ELO_BURN_IN_START_SEASON: int = 2002

# The staged-generation tree and the single pointer that publishes one.
ELO_GENERATION_DIRNAME: str = "elo_generations"
ELO_GENERATION_POINTER_NAME: str = "elo_generation.json"
ELO_GENERATION_POINTER_PATH: Path = (
    Path("data") / "silver" / ELO_GENERATION_POINTER_NAME
)


class EloGenerationIncompleteError(RuntimeError):
    """A generation was missing an artifact, or its members disagreed.

    Raised BEFORE anything live is written and before the pointer moves, so the
    previous generation stays published and readable.
    """


# ---------------------------------------------------------------------------
# The generation publisher: five artifacts, staged together, one pointer move.
# ---------------------------------------------------------------------------


def new_generation_id() -> str:
    """A sortable, collision-resistant generation id.

    Microsecond resolution on purpose: two publishes inside the same second are
    ordinary in a test and possible in a rerun, and a colliding id would silently
    stage the second generation on top of the first.
    """
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def staged_artifact_filename(name: str) -> str:
    """The on-disk filename a staged artifact takes.

    ``elo_ratings`` is the Elo system's JSON state; the other four are tables.
    """
    return f"{name}.json" if name == "elo_ratings" else f"{name}.parquet"


def default_stage_writer(path: Path, payload: Any) -> None:
    """Write one staged artifact.

    PARQUET AND JSON ONLY -- deliberately NOT ``save_dataframe``. Staging must not
    touch the shared DuckDB store or the live parquet paths: the whole guarantee is
    that a failure during staging leaves the live artifacts untouched, and a stage
    writer that routed through ``save_dataframe`` (``save_to_db=True`` by default)
    would write the live database on every staged write and destroy that guarantee.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, pd.DataFrame):
        payload.to_parquet(path, engine="pyarrow", index=False)
        return
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def elo_generation_pointer_path(silver_root: Path) -> Path:
    """The generation pointer beneath *silver_root*."""
    return Path(silver_root) / ELO_GENERATION_POINTER_NAME


def read_elo_generation_pointer(silver_root: Path) -> dict[str, Any] | None:
    """Read the published generation pointer, or None when nothing is published."""
    path = elo_generation_pointer_path(silver_root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class EloGenerationPublisher:
    """Stage five Elo artifacts, validate them together, then move ONE pointer.

    FIVE INDEPENDENTLY-ATOMIC FILES ARE NOT ONE ATOMIC STATE. Each individual write in
    the old ``save_results`` was atomic; the SET was not, so a crash between any two of
    them left snapshots from one build beside a rating history from another, with
    nothing on disk saying so. Every consumer downstream would read that mixture as a
    single coherent Elo state, because from the outside it is indistinguishable from
    one.
    """

    generation_id: str
    silver_root: Path
    stage_writer: Callable[[Path, Any], None] = default_stage_writer
    staged: dict[str, Path] = field(default_factory=dict)

    @property
    def generation_dir(self) -> Path:
        return Path(self.silver_root) / ELO_GENERATION_DIRNAME / self.generation_id

    def stage(self, name: str, payload: Any) -> Path:
        """Write one artifact into this generation's staging directory."""
        path = self.generation_dir / staged_artifact_filename(name)
        self.stage_writer(path, payload)
        self.staged[name] = path
        return path

    def validate(self) -> None:
        """Refuse a generation whose members are missing or disagree.

        Two checks, both about the SET rather than about any one file:

        1. All five artifacts are staged and present on disk.
        2. Every NON-EMPTY row table ends on the same terminal season, and
           ``elo_ratings_current`` names that same terminal season. A season with zero
           completed games legitimately produces empty row tables (R3's explicit edge
           case), so empty members are skipped rather than treated as disagreement.
        """
        expected = (*ELO_ROW_TABLES, *ELO_STATE_ARTIFACTS)
        missing = [
            name
            for name in expected
            if name not in self.staged or not self.staged[name].exists()
        ]
        if missing:
            raise EloGenerationIncompleteError(
                f"Elo generation {self.generation_id} is incomplete: "
                f"{', '.join(missing)} was not staged. Nothing was published and the "
                "generation pointer did not move, so the previous generation is still "
                "the one that serves."
            )

        terminal_by_table: dict[str, int] = {}
        for name in ELO_ROW_TABLES:
            frame = pd.read_parquet(self.staged[name], engine="pyarrow")
            if len(frame) == 0 or "season" not in frame.columns:
                continue
            seasons = frame["season"].dropna()
            if len(seasons) == 0:
                continue
            terminal_by_table[name] = int(seasons.max())

        if not terminal_by_table:
            return

        terminals = set(terminal_by_table.values())
        if len(terminals) > 1:
            raise EloGenerationIncompleteError(
                f"Elo generation {self.generation_id} is inconsistent: the row tables "
                f"end on different terminal seasons ({terminal_by_table}). A generation "
                "whose members describe different points in time is a mixed state."
            )

        terminal = terminals.pop()
        current = pd.read_parquet(self.staged["elo_ratings_current"], engine="pyarrow")
        if len(current) > 0 and "season" in current.columns:
            current_seasons = current["season"].dropna()
            if len(current_seasons) > 0 and int(current_seasons.max()) != terminal:
                raise EloGenerationIncompleteError(
                    f"Elo generation {self.generation_id} is inconsistent: "
                    f"elo_ratings_current names terminal season "
                    f"{int(current_seasons.max())} while the row tables end on "
                    f"{terminal}."
                )

    def move_pointer(self, **extra: Any) -> Path:
        """Publish this generation with ONE atomic pointer write.

        Reuses ``models.artifacts._atomic_write_json``'s temp-then-``os.replace``
        idiom rather than writing a second one: ``Path.write_text`` truncates before it
        writes, so an interruption leaves a 0-byte pointer and every reader fails.
        """
        from models.artifacts import _atomic_write_json

        path = elo_generation_pointer_path(self.silver_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "generation_id": self.generation_id,
            "published_at": datetime.now(UTC).isoformat(),
            "generation_dir": str(self.generation_dir),
            "artifacts": {
                name: staged_path.name
                for name, staged_path in sorted(self.staged.items())
            },
            **extra,
        }
        _atomic_write_json(path, payload)
        return path


def publish_elo_generation(
    staged: Mapping[str, Any],
    generation_id: str,
    *,
    silver_root: Path,
    publish_live: Callable[[], None],
    stage_writer: Callable[[Path, Any], None] | None = None,
    **pointer_fields: Any,
) -> Path:
    """Stage, validate, publish and point -- in that order, with no shortcuts.

    Args:
        staged: Artifact name -> payload. DataFrames for the four tables, a dict for
            ``elo_ratings``.
        generation_id: This generation's id (see :func:`new_generation_id`).
        silver_root: The silver layer this generation belongs to.
        publish_live: Callable that performs the LIVE writes. It differs between the
            two verbs -- replace for a full rebuild, upsert for a live append -- which
            is exactly why the publisher takes it rather than deciding it.
        stage_writer: Override for the staged-write function (tests inject failures).
        **pointer_fields: Extra facts recorded in the pointer (mode, season, ...).

    Returns:
        Path to the generation pointer.

    Raises:
        EloGenerationIncompleteError: When a member is missing or the members
            disagree. Nothing live is written and the pointer does not move.
    """
    publisher = EloGenerationPublisher(
        generation_id=generation_id,
        silver_root=Path(silver_root),
        stage_writer=stage_writer or default_stage_writer,
    )
    for name in (*ELO_ROW_TABLES, *ELO_STATE_ARTIFACTS):
        if name in staged:
            publisher.stage(name, staged[name])

    publisher.validate()
    publish_live()
    return publisher.move_pointer(**pointer_fields)

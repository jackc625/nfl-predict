"""Rebuild gold with the builder the in-force recipe names (MIDSEASON-SWITCH-RECORD.md section 4).

The gold boundary's recovery command. ``scripts.build_features`` always builds v1 gold, so once
the owner install puts the v2 recipe in force, a hand-run that saves would write v1 gold under
the v2 models, which read their columns BY NAME. This command runs the pipeline's own gold step,
``pipeline.steps.step_build_features``, whose builder follows
``refit.generations.in_force_recipe``: the gold it writes always matches the models in force.
When the recipe in force cannot be read (no ``artifacts/latest.json``), it raises and writes
nothing.

Usage:
    uv run python -m scripts.rebuild_gold

ASCII only, no emoji.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """The CLI: no options; the recipe in force decides the builder."""
    return argparse.ArgumentParser(
        description=(
            "Rebuild and save the gold matrices with the builder the in-force recipe names "
            "(the pipeline's own build_features step)."
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Rebuild and save gold with the in-force recipe's builder; 0 when it is written."""
    build_parser().parse_args(argv)
    # Imported here, as the pipeline's step adapters import their scripts: loading this module
    # (for --help) builds nothing.
    from pipeline.steps import step_build_features

    step_build_features()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

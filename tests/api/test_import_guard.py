"""UIAP-01 compliance test: API must not import model classes.

Uses AST parsing to verify that no file in the api/ directory imports
any of the forbidden model classes. This ensures the API layer only
serves precomputed data from the DuckDB cache.
"""

import ast
import pathlib


def test_api_does_not_import_model_classes():
    """UIAP-01: API serves precomputed artifacts only -- no model inference."""
    forbidden = {
        "WPTrainer",
        "ATSTrainer",
        "OUTrainer",
        "MarketBlender",
        "BacktestEngine",
        "BettingSimulator",
    }
    api_dir = pathlib.Path("api")
    for py_file in api_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                overlap = forbidden.intersection(names)
                assert not overlap, (
                    f"{py_file} imports forbidden class: {overlap}"
                )

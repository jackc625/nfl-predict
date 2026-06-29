"""Gold-width tripwire parity test (IN-03, Plan 28-06 / SIG-06).

``scripts.data_qa.GOLD_FEATURE_MATRICES`` hardcodes the expected column count of each
gold feature matrix as a deliberate audit tripwire: any stray/dropped column trips it
rather than passing silently. This test asserts the tripwire is itself HONEST -- the
declared expected width equals the real on-disk column count of each rebuilt matrix.

If a future builder legitimately adds or removes a column, this test fails until the
operator UPDATES ``GOLD_FEATURE_MATRICES`` to the new empirically-counted width
(Pitfall 6 -- never silence the tripwire by widening tolerances).
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.data_qa import GOLD_FEATURE_MATRICES

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"


@pytest.mark.parametrize(
    ("table_name", "expected_width"), list(GOLD_FEATURE_MATRICES.items())
)
def test_gold_matrix_width_matches_tripwire(
    table_name: str, expected_width: int
) -> None:
    """Each rebuilt gold matrix's real column count equals its tripwire value."""
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    actual_width = pd.read_parquet(path).shape[1]
    assert actual_width == expected_width, (
        f"{table_name}: real column count {actual_width} != "
        f"GOLD_FEATURE_MATRICES expected {expected_width}. If the change is "
        f"intentional, update GOLD_FEATURE_MATRICES in scripts/data_qa.py to the "
        f"new empirically-counted width (IN-03, never silence the tripwire)."
    )

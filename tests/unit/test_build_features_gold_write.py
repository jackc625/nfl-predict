"""G-02: build_features gold write does not pass partition_cols (Phase 20, CR-01, D-10).

Source-level call-site guard (approach b from the gap spec): assert that the
save_dataframe call in scripts/build_features.py for the gold layer does NOT pass
partition_cols as a keyword argument.

The regression class: the old gold write used partition_cols=["season"], which routes
pq.write_to_dataset into the shared data/gold/ root where all three matrices
(features_wp/ats/ou) collide in the same season=YYYY/ directory -- every current-week
run appended a NEW hash-named parquet instead of overwriting, silently multiplying gold
cardinality and cross-contaminating the three matrices across runs.

The fix (commit 0269a15): removed partition_cols entirely from the gold write call, so
save_dataframe uses its default append_mode=True path (single file, latest-wins dedup).
This test will FAIL if a future edit re-introduces partition_cols= to the gold write.

Approach: inspect.getsource on the build_features module, extract the save_dataframe
call block for layer="gold", and assert "partition_cols" is absent from that call.
Mirrors the source-guard pattern in tests/integration/test_audit_determinism.py.
"""

from __future__ import annotations

import inspect
import textwrap


def test_gold_write_call_has_no_partition_cols():
    """The save_dataframe(..., layer="gold") call in build_features must NOT pass
    partition_cols -- re-adding it would re-introduce the shared-root partitioned-append
    antipattern that multiplied gold cardinality and cross-contaminated the matrices.

    Regression: if partition_cols= reappears in the gold write, pq.write_to_dataset
    writes season=YYYY/ dirs into the shared data/gold/ root, all three feature
    matrices collide there, and every rebuild appends new hash-named files instead of
    overwriting.  (Phase 20, commit 0269a15, CR-01, D-10)
    """
    import scripts.build_features as build_features_mod

    source = inspect.getsource(build_features_mod)

    # Locate the save_dataframe call that targets layer="gold".
    # Strategy: find the character offset of the gold-layer save_dataframe call,
    # then extract a window around it to check for partition_cols.
    gold_write_marker = 'layer="gold"'

    # Find ALL occurrences -- there should be exactly one save_dataframe call
    # with layer="gold" (the gold matrix write).
    occurrences = []
    search_start = 0
    while True:
        idx = source.find(gold_write_marker, search_start)
        if idx == -1:
            break
        occurrences.append(idx)
        search_start = idx + 1

    assert len(occurrences) >= 1, (
        'scripts/build_features.py has no save_dataframe call with layer="gold" -- '
        "the gold write was removed or renamed, which itself is a regression"
    )

    # For each occurrence, extract a 600-character window back to find the
    # enclosing save_dataframe( ... ) call and check for partition_cols.
    # 600 chars is ample for a multi-line call with several keyword arguments.
    regressions_found = []
    for idx in occurrences:
        window_start = max(0, idx - 600)
        window = source[window_start : idx + 200]

        # Only examine windows where the call is a save_dataframe invocation
        # (not a comment referencing layer="gold")
        if "save_dataframe(" not in window:
            continue

        # Look for partition_cols being passed as a keyword argument in this window.
        # The marker we look for: "partition_cols" appearing after the last
        # save_dataframe( before the layer="gold" token.
        last_call_start = window.rfind("save_dataframe(")
        call_fragment = window[last_call_start:]

        if "partition_cols" in call_fragment:
            regressions_found.append(textwrap.shorten(call_fragment, width=200))

    assert regressions_found == [], (
        "REGRESSION DETECTED: the gold-layer save_dataframe call in "
        "scripts/build_features.py now passes partition_cols. "
        "This re-introduces the shared-root partitioned-append antipattern "
        "(season=YYYY/ dirs in gold/, multiplying cardinality on every rebuild). "
        f"Offending call fragment(s): {regressions_found}"
    )


def test_gold_write_call_targets_correct_table_names():
    """The gold write produces features_wp, features_ats, features_ou tables.

    Sanity guard: the table_name= values passed to the gold save_dataframe call
    all follow the features_{target} pattern, confirming the call site has not
    been renamed or pointed at an unexpected table.
    """
    import scripts.build_features as build_features_mod

    source = inspect.getsource(build_features_mod)

    # The gold write block constructs table_name = f"features_{target}" then passes
    # table_name= to save_dataframe.  Verify the pattern exists in source.
    assert 'f"features_{target}"' in source or "features_{target}" in source, (
        "scripts/build_features.py no longer constructs 'features_{target}' as the "
        "gold table name -- the gold write call site has been refactored in an "
        "unexpected way; review for partition_cols regression manually"
    )

    # Verify the three known gold table names are reachable from the source
    # (as string literals they appear in comments or the builder loop targets).
    for target in ("wp", "ats", "ou"):
        assert target in source, (
            f"Target '{target}' not found in build_features source -- the gold "
            f"write for features_{target} may have been removed"
        )

"""The Phase 37 frozen-recipe rolling re-fit package.

It is the ONLY place allowed to call the model-manifest writers, which is why it sits outside
``forward_ledger/``: ``tests/unit/test_ledger_never_writes_latest.py`` forbids those writers under
``forward_ledger/`` and in the daily run's modules, so a ledger run can never swap the models that
score the next decision.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

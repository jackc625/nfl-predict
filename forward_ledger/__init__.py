"""The 2026 append-only forward bet ledger (Phase 34, LDGR-01..11).

WHY A PACKAGE NAMED ``forward_ledger`` AND NOT ``ledger``
--------------------------------------------------------
``ledger/`` is the DATA directory the store lives in (gitignored by the public repo, D-10). A code
package with the same top-level name would collide with it: an ``__init__``-less ``ledger/``
directory becomes an implicit namespace package, and ``import ledger.x`` would then resolve
unpredictably between code and data (34-RESEARCH Pitfall 4).

WHAT LIVES HERE
---------------
* :mod:`forward_ledger.canonical` -- the frozen v1 column lists, the one schema-typed canonical
  serialization, the published genesis constant and the chain-hash formula.
* :mod:`forward_ledger.store` -- the one-file JSON-lines store, its reader and atomic writer, and
  the chain verifier.
* :mod:`forward_ledger.schema` -- the ledger row key and the import-time proof that the four
  mutability classes of ``api.cache``'s bet-list schema are disjoint and covering.

No module under ``api/`` may import this package (D-18, UIAP-01): verification is a CLI, never a
request path.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

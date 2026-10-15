"""The owner notification sender: phone push, weekly email and the outside watcher (Phase 35).

WHY THIS PACKAGE EXISTS
-----------------------
Until Phase 35 every alert was a log line on the laptop, so a failed run reached the owner nowhere.
This package is the ONE delivery path every message takes: a provider receipt (a message id) is the
only proof that a message was delivered, failures are retried only when they are transient, and
every message leaves exactly one record in ``logs/notifications.jsonl``.

WHAT LIVES HERE
---------------
* :mod:`notify.secrets` -- the five secrets, read from ``conf.settings``, and the provider constants.
* :mod:`notify.transport` -- one bounded, classified retry loop over httpx; the test seams.
* :mod:`notify.channels` -- one function per provider call (ntfy here; Plan 35-05 adds the rest).
* :mod:`notify.log` -- the append-only notification log and its reader.
* :mod:`notify.events` -- the closed event-class vocabulary and the push router.

WHO MAY IMPORT IT (D-20)
------------------------
Only ``scripts/`` and the daily run. Nothing under ``api/`` imports it (UIAP-01): a request path
never sends a message.

This file imports nothing, so a broken submodule can never break an import of another.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

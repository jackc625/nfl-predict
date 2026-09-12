"""RED interface stub -- replaced wholesale in the GREEN commit of Plan 33-09 Task 2.

The names below exist only so the Task-2 test modules can IMPORT this module.
Without them pytest fails at COLLECTION, which this phase's own gate classifies as
INVALID_RED: a load failure proves nothing about behaviour.
"""

# RED-phase sentinel. The real archive endpoint lands in the GREEN commit.
ARCHIVE_ENDPOINT_URL = ""


def main() -> None:
    """RED stub -- replaced in the GREEN commit."""
    raise NotImplementedError("backfill_historical_weather is not implemented yet")

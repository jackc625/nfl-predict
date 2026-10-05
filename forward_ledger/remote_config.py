"""Where the ledger's head and its backup go, and the credentials that take them there (Plan 34-06).

Committed constants only -- no secret lives here. The private key FILES are named by path and
stay under the owner's profile, outside the repository (D-03); this module never reads them.

TWO DESTINATIONS, TWO KEYS (D-01, D-03)
--------------------------------------
* The PUBLIC repository receives the anchor: one commit per head move on ``ANCHOR_REF``, holding
  the head hash and the row count and nothing else (LDGR-05). It is pushed with the anchor deploy
  key and one explicit refspec; ``master`` is never named (D-12).
* The PRIVATE repository receives the whole ``ledger/`` directory, a nested git repository whose
  ``main`` branch is pushed after every run that changed anything in it (D-01, D-02).

GitHub refuses the same deploy key on two repositories, so each destination has its own key.

WHY ABSOLUTE PATHS
------------------
The daily run executes under a Task Scheduler S4U logon, whose profile and PATH must not be
guessed: the ssh executable and the key files are therefore absolute (34-RESEARCH D). The git
identity is copied here for the same reason -- the S4U profile's ``.gitconfig`` is not relied on.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ANCHOR_FILE_NAME",
    "ANCHOR_KEY_PATH",
    "ANCHOR_PUSHED_REF",
    "ANCHOR_REF",
    "ANCHOR_REMOTE_HTTPS_URL",
    "ANCHOR_REMOTE_URL",
    "ANCHOR_SPEC",
    "BACKUP_BRANCH_REF",
    "BACKUP_KEY_PATH",
    "BACKUP_PUSHED_REF",
    "BACKUP_REMOTE_URL",
    "BACKUP_REPO_SLUG",
    "BACKUP_SPEC",
    "CONNECT_TIMEOUT_SECONDS",
    "GIT_IDENTITY_EMAIL",
    "GIT_IDENTITY_NAME",
    "KNOWN_HOSTS_PATH",
    "PUBLIC_REPO_SLUG",
    "PUSH_TIMEOUT_SECONDS",
    "REMOTE_ANCHOR_FETCH_REF",
    "SSH_EXECUTABLE",
    "RemoteSpec",
]

PUBLIC_REPO_SLUG: str = "jackc625/nfl-predict"
BACKUP_REPO_SLUG: str = "jackc625/nfl-predict-ledger"

# Pushes always name the SSH URL explicitly, never a configured remote: ``origin`` stays HTTPS for
# the owner's own workflow, and no remote name can be mistyped into a push to ``master``.
ANCHOR_REMOTE_URL: str = "git@github.com:jackc625/nfl-predict.git"
# The public repository read anonymously, for the verify CLI's remote check (read-only fetch).
ANCHOR_REMOTE_HTTPS_URL: str = "https://github.com/jackc625/nfl-predict.git"
BACKUP_REMOTE_URL: str = "git@github.com:jackc625/nfl-predict-ledger.git"

# The anchor branch (D-12). Costly to rename after the first push: the published head history
# and the verify CLI's remote check both live under this name.
ANCHOR_REF: str = "refs/heads/ledger-anchor"
# Local bookkeeping refs outside refs/heads, so no branch listing or push of branches carries them.
ANCHOR_PUSHED_REF: str = "refs/ledger/anchor-pushed"
REMOTE_ANCHOR_FETCH_REF: str = "refs/ledger/remote-anchor"
BACKUP_BRANCH_REF: str = "refs/heads/main"
BACKUP_PUSHED_REF: str = "refs/ledger/backup-pushed"

# The one file in every anchor commit's tree.
ANCHOR_FILE_NAME: str = "ANCHOR"

# Windows OpenSSH, by absolute path: the machine PATH lists it before Git's own ssh, but the
# transport must not depend on resolution order (34-RESEARCH D).
SSH_EXECUTABLE: str = "C:/Windows/System32/OpenSSH/ssh.exe"

_SSH_DIR = Path("C:/Users/jackc/.ssh")
ANCHOR_KEY_PATH: Path = _SSH_DIR / "nfl_ledger_anchor_ed25519"
BACKUP_KEY_PATH: Path = _SSH_DIR / "nfl_ledger_backup_ed25519"
# GitHub's published host keys, pinned in a dedicated file (created and fingerprint-checked by
# the setup plan); StrictHostKeyChecking=yes reads only this file.
KNOWN_HOSTS_PATH: Path = _SSH_DIR / "nfl_ledger_github_known_hosts"

# Copied from ``git log -1 --format="%an|%ae"`` on master when this module was written.
GIT_IDENTITY_NAME: str = "Jack Cutrara"
GIT_IDENTITY_EMAIL: str = "jackhcutrara@gmail.com"

# A hung push or fetch must not hold the daily run (34-RESEARCH D).
PUSH_TIMEOUT_SECONDS: int = 90
CONNECT_TIMEOUT_SECONDS: int = 20


@dataclass(frozen=True)
class RemoteSpec:
    """One push destination: its SSH URL and the deploy key and known_hosts file that reach it."""

    url: str
    key_path: Path
    known_hosts_path: Path


ANCHOR_SPEC: RemoteSpec = RemoteSpec(
    url=ANCHOR_REMOTE_URL, key_path=ANCHOR_KEY_PATH, known_hosts_path=KNOWN_HOSTS_PATH
)
BACKUP_SPEC: RemoteSpec = RemoteSpec(
    url=BACKUP_REMOTE_URL, key_path=BACKUP_KEY_PATH, known_hosts_path=KNOWN_HOSTS_PATH
)

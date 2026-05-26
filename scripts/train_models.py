"""Train models -- thin wrapper for models.train module.

Usage:
    python scripts/train_models.py --target all
    python scripts/train_models.py --target wp

This is a convenience wrapper. The canonical invocation is:
    python -m models.train --target all

All flags (--target, --artifacts-dir, --no-clv, --config-train-seasons, ...) are
owned and parsed by models.train.main; this wrapper adds no argument parsing of
its own.
"""

from models.train import main

if __name__ == "__main__":
    main()

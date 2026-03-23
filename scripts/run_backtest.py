"""Run backtest -- thin wrapper for backtest.run module.

Usage:
    python scripts/run_backtest.py [--seasons 2021,2022,2023,2024] [--targets wp,ats,ou]

This is a convenience wrapper. The canonical invocation is:
    python -m backtest.run
"""

from backtest.run import main

if __name__ == "__main__":
    main()

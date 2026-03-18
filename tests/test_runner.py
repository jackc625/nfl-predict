#!/usr/bin/env python3
"""
Test runner for NFL Prediction System.

This script provides convenient test execution with different modes and configurations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(
    test_type="all", verbose=False, coverage=False, parallel=False, pattern=None
):
    """Run tests with specified configuration."""
    project_root = Path(__file__).parent.parent

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test paths based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "api":
        cmd.append("tests/api/")
    elif test_type == "ui":
        cmd.append("tests/ui/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}/")

    # Add pattern matching
    if pattern:
        cmd.extend(["-k", pattern])

    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add coverage
    if coverage:
        cmd.extend(
            [
                "--cov=scripts",
                "--cov=ratings",
                "--cov=models",
                "--cov=backtest",
                "--cov=api",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
            ]
        )

    # Add parallel execution
    if parallel:
        try:
            import pytest_xdist  # noqa: F401

            cmd.extend(["-n", "auto"])
        except ImportError:
            pass

    # Run tests

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="NFL Prediction System Test Runner")

    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "api", "ui", "fast"],
        help="Type of tests to run",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "-c", "--coverage", action="store_true", help="Run with coverage reporting"
    )

    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )

    parser.add_argument("-k", "--pattern", help="Only run tests matching this pattern")

    parser.add_argument(
        "--install-deps", action="store_true", help="Install test dependencies first"
    )

    args = parser.parse_args()

    # Install dependencies if requested
    if args.install_deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[test]"])

    # Handle 'fast' test type (unit tests only, no coverage)
    if args.test_type == "fast":
        args.test_type = "unit"
        args.coverage = False

    # Run tests
    exit_code = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        pattern=args.pattern,
    )

    if exit_code == 0:
        if args.coverage:
            pass
    else:
        pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

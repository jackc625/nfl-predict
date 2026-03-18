#!/usr/bin/env python3
"""
Production server runner for NFL Prediction API.

This script provides a robust way to run the API in production with
proper configuration, health checks, and monitoring.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from conf.settings import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ProductionRunner:
    """Production server runner with monitoring and graceful shutdown."""

    def __init__(self, config_path: str = "deployment/production.env"):
        self.config_path = config_path
        self.process = None
        self.shutdown_requested = False
        self.settings = get_settings()
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._handle_reload)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        if self.process:
            self._stop_server()

    def _handle_reload(self, signum, frame):
        """Handle reload signal."""
        logger.info("Received SIGHUP, reloading configuration...")
        if self.process:
            self._restart_server()

    def load_environment(self):
        """Load production environment variables."""
        env_file = Path(self.config_path)
        if not env_file.exists():
            logger.warning(f"Environment file {env_file} not found, using defaults")
            return

        logger.info(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
                    except ValueError:
                        logger.warning(f"Skipping invalid environment line: {line}")

    def validate_environment(self) -> bool:
        """Validate required environment variables."""
        required_vars = [
            "SECRET_KEY",
            "ENVIRONMENT",
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        # Validate SECRET_KEY length
        secret_key = os.getenv("SECRET_KEY", "")
        if len(secret_key) < 32:
            logger.error("SECRET_KEY must be at least 32 characters long")
            return False

        return True

    def pre_flight_checks(self) -> bool:
        """Perform pre-flight checks before starting server."""
        logger.info("Performing pre-flight checks...")

        # Check Python version

        # Check required directories
        required_dirs = ["logs", "data", "outputs", "artifacts"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                logger.info(f"Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)

        # Check disk space
        if not self._check_disk_space():
            return False

        # Test database connection
        if not self._test_database():
            return False

        # Test critical imports
        try:
            import fastapi  # noqa: F401
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import uvicorn  # noqa: F401

            logger.info("Critical imports verified")
        except ImportError as e:
            logger.error(f"Missing critical dependency: {e}")
            return False

        logger.info("Pre-flight checks completed successfully")
        return True

    def _check_disk_space(self, min_gb: float = 1.0) -> bool:
        """Check available disk space."""
        try:
            import shutil

            _total, _used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)

            if free_gb < min_gb:
                logger.error(
                    f"Insufficient disk space: {free_gb:.2f}GB available, {min_gb}GB required"
                )
                return False

            logger.info(f"Disk space check passed: {free_gb:.2f}GB available")
            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Don't fail startup for this

    def _test_database(self) -> bool:
        """Test database connectivity."""
        try:
            from data.storage import get_db_connection

            db = get_db_connection()
            db.execute("SELECT 1").fetchone()
            logger.info("Database connection test passed")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_server_command(self, use_gunicorn: bool = True) -> list[str]:
        """Get server command based on configuration."""
        if use_gunicorn and self._should_use_gunicorn():
            return self._get_gunicorn_command()
        return self._get_uvicorn_command()

    def _should_use_gunicorn(self) -> bool:
        """Determine if we should use Gunicorn."""
        # Use Gunicorn in production with multiple workers
        return (
            os.getenv("ENVIRONMENT") == "production"
            and int(os.getenv("GUNICORN_WORKERS", "1")) > 1
            and os.name != "nt"  # Gunicorn doesn't work on Windows
        )

    def _get_gunicorn_command(self) -> list[str]:
        """Get Gunicorn command."""
        cmd = [
            sys.executable,
            "-m",
            "gunicorn",
            "--config",
            "deployment/gunicorn.conf.py",
            "api.main:app",
        ]

        # Add environment-specific options
        if os.getenv("GUNICORN_RELOAD") == "true":
            cmd.append("--reload")

        return cmd

    def _get_uvicorn_command(self) -> list[str]:
        """Get Uvicorn command."""
        host = os.getenv("API_HOST", "0.0.0.0")
        port = os.getenv("API_PORT", "8000")
        workers = int(os.getenv("UVICORN_WORKERS", "1"))

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            host,
            "--port",
            port,
        ]

        # Add workers if specified
        if workers > 1:
            cmd.extend(["--workers", str(workers)])

        # Add environment-specific options
        if os.getenv("DEBUG") == "true":
            cmd.append("--reload")

        log_level = os.getenv("LOG_LEVEL", "info").lower()
        cmd.extend(["--log-level", log_level])

        # SSL configuration
        ssl_keyfile = os.getenv("SSL_KEYFILE")
        ssl_certfile = os.getenv("SSL_CERTFILE")
        if ssl_keyfile and ssl_certfile:
            cmd.extend(["--ssl-keyfile", ssl_keyfile])
            cmd.extend(["--ssl-certfile", ssl_certfile])

        return cmd

    def start_server(self, use_gunicorn: bool = True) -> bool:
        """Start the production server."""
        if not self.validate_environment():
            return False

        if not self.pre_flight_checks():
            return False

        command = self.get_server_command(use_gunicorn)
        logger.info(f"Starting server with command: {' '.join(command)}")

        try:
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root)

            # Start the server process
            self.process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Monitor the process
            self._monitor_process()
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def _monitor_process(self):
        """Monitor the server process."""
        logger.info(f"Server started with PID: {self.process.pid}")

        # Wait for process to finish or shutdown signal
        try:
            while not self.shutdown_requested:
                # Check if process is still running
                return_code = self.process.poll()
                if return_code is not None:
                    if return_code == 0:
                        logger.info("Server process completed successfully")
                    else:
                        logger.error(f"Server process exited with code: {return_code}")
                    break

                # Read and log output
                try:
                    line = self.process.stdout.readline()
                    if line:
                        pass
                except Exception:
                    pass

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            if self.process and self.process.poll() is None:
                self._stop_server()

    def _stop_server(self):
        """Stop the server gracefully."""
        if not self.process:
            return

        logger.info("Stopping server gracefully...")

        # Send SIGTERM for graceful shutdown
        try:
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=30)
                logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, forcing shutdown...")
                self.process.kill()
                self.process.wait(timeout=10)
                logger.info("Server force stopped")

        except Exception as e:
            logger.error(f"Error stopping server: {e}")

        self.process = None

    def _restart_server(self):
        """Restart the server."""
        logger.info("Restarting server...")
        self._stop_server()
        time.sleep(2)  # Brief pause
        self.start_server()

    def health_check(self) -> bool:
        """Perform health check on running server."""
        try:
            import httpx

            host = os.getenv("API_HOST", "0.0.0.0")
            port = os.getenv("API_PORT", "8000")

            # Use localhost for health check if binding to 0.0.0.0
            check_host = "localhost" if host == "0.0.0.0" else host
            url = f"http://{check_host}:{port}/health"

            response = httpx.get(url, timeout=10)
            if response.status_code == 200:
                logger.info("Health check passed")
                return True
            logger.error(f"Health check failed with status: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def status(self) -> dict[str, Any]:
        """Get server status."""
        return {
            "running": self.process is not None and self.process.poll() is None,
            "pid": self.process.pid if self.process else None,
            "config_path": self.config_path,
            "environment": os.getenv("ENVIRONMENT", "unknown"),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NFL Prediction API Production Runner")
    parser.add_argument(
        "--config",
        default="deployment/production.env",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--server",
        choices=["gunicorn", "uvicorn", "auto"],
        default="auto",
        help="Server type to use",
    )
    parser.add_argument(
        "--health-check", action="store_true", help="Perform health check and exit"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate configuration and exit"
    )

    args = parser.parse_args()

    # Initialize runner
    runner = ProductionRunner(args.config)
    runner.load_environment()

    if args.validate_only:
        success = runner.validate_environment() and runner.pre_flight_checks()
        sys.exit(0 if success else 1)

    if args.health_check:
        success = runner.health_check()
        sys.exit(0 if success else 1)

    # Determine server type
    use_gunicorn = args.server == "gunicorn" or (
        args.server == "auto" and runner._should_use_gunicorn()
    )

    # Start server
    success = runner.start_server(use_gunicorn)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

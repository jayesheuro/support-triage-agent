"""
app/core/logging.py

Responsibility: Configure structured logging for the application. Outputs JSON
lines in production mode; human-readable in development. Never logs secrets or
full customer message bodies.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Carry through any extra fields added via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName", "taskName",
            }:
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure root logger. Call once at application startup.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, emit JSON lines; otherwise human-readable.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers (e.g. from uvicorn)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use __name__ as the name in each module."""
    return logging.getLogger(name)

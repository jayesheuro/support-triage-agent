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


class _ContextFilter(logging.Filter):
    """
    Injects ticket_id, run_id, and node from active tracing context variables
    into all log records so every log during graph execution is correlated.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from app.services.tracing.tracer import (
                get_current_node_name,
                get_current_run_id,
                get_current_ticket_id,
            )

            if not getattr(record, "ticket_id", None):
                record.ticket_id = get_current_ticket_id()
            if not getattr(record, "run_id", None):
                record.run_id = get_current_run_id()
            if not getattr(record, "node", None):
                record.node = get_current_node_name()
        except Exception:
            pass
        return True


class _HumanFormatter(logging.Formatter):
    """Formats log records for console readability with correlated context tags."""

    def format(self, record: logging.LogRecord) -> str:
        ctx_parts: list[str] = []
        if getattr(record, "ticket_id", None):
            ctx_parts.append(str(record.ticket_id))
        if getattr(record, "node", None):
            ctx_parts.append(str(record.node))
        ctx_tag = f" [{':'.join(ctx_parts)}]" if ctx_parts else ""

        orig_msg = record.getMessage()
        formatted_time = self.formatTime(record, self.datefmt)
        return f"{formatted_time} | {record.levelname:<8} | {record.name}{ctx_tag} | {orig_msg}"


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
    handler.addFilter(_ContextFilter())

    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_HumanFormatter(datefmt="%H:%M:%S"))
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use __name__ as the name in each module."""
    return logging.getLogger(name)

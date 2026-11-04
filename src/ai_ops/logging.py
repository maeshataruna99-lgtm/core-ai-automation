"""Structured JSON logging with a request id on every line.

Why: One JSON object per line can be searched and filtered by field, and the request id
links all lines of one request. The standard library is enough for this, so no logging
dependency is needed.

Used by: `ai_ops.api.app.create_app` calls `configure_logging`.
`ai_ops.api.middleware` sets `request_id_var` and writes the request log.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

LOGGER_NAME = "ai_ops"

# A ContextVar is like a thread-local that also works with async code: each request sees
# its own value, even when many requests run on the same thread.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Extra fields copied into the log line when a caller passes them through `extra=`.
_EXTRA_FIELDS = (
    "method",
    "path",
    "status",
    "duration_ms",
    "attempt",
    "delay_s",
    "error",
    "tool",
    "tool_input",
    "outcome",
    "approval_id",
    "ticket_id",
    "step",
)


class JsonFormatter(logging.Formatter):
    """Format each log record as one line of JSON.

    Why: The roadmap requires timestamp, severity, and request id on each line. Adding them
    here means no call site can forget them.

    Used by: `configure_logging`.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record to a JSON string.

        Why: This overrides the method that `logging` calls for every record.

        Used by: the `logging` module, through the handler in `configure_logging`.

        Args:
            record: The record to format.

        Returns:
            One line of JSON.
        """
        line = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "request_id": request_id_var.get(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        line.update({key: getattr(record, key) for key in _EXTRA_FIELDS if hasattr(record, key)})
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line)


def configure_logging(level: str) -> None:
    """Send `ai_ops` logs to stderr as JSON.

    Why: Only the `ai_ops` logger is changed, so uvicorn and libraries keep their own
    logging. Replacing the handlers keeps this safe to call more than once, which happens
    when tests build many apps.

    Used by: `ai_ops.api.app.create_app`.

    Args:
        level: Minimum severity name, such as `INFO`.

    Side effects:
        Replaces the handlers of the `ai_ops` logger and stops propagation to the root
        logger, so a line is not written twice.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False

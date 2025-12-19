from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """
    A structured JSON formatter for production-grade logging systems.

    Ensures logs are machine-readable and easy to index in systems such as
    Datadog, Splunk, CloudWatch, and ELK.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Additional metadata passed via logger.info(..., extra={})
        for attr in ("event", "user_id", "shape", "step", "exception_type"):
            if hasattr(record, attr):
                log_payload[attr] = getattr(record, attr)

        # Include exception details when available
        if record.exc_info:
            log_payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, ensure_ascii=False)


def configure_logger(name: str = "recommender.similarity", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with JSON formatting.

    Prevents duplicate handlers and ensures clean structured logs.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


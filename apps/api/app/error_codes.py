# apps/api/app/error_codes.py
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """
    Stable error codes for clients.

    These values are part of the public API contract and must remain stable.
    """

    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    HTTP_ERROR = "HTTP_ERROR"


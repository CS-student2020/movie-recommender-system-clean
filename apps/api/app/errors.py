# apps/api/app/errors.py
from __future__ import annotations

from typing import Any, Optional

from fastapi import Request

from apps.api.app.schemas.errors import ErrorResponse


UNKNOWN_REQUEST_ID = "unknown"


def get_request_id(request: Request) -> str:
    """
    Return the request correlation id if present.

    This helper expects a middleware to set `request.state.request_id`.
    If missing, it returns a stable sentinel value rather than generating a new id.
    """
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and rid.strip():
        return rid
    return UNKNOWN_REQUEST_ID


def make_error(
    *,
    code: str,
    message: str,
    request_id: str,
    details: Optional[Any] = None,
) -> ErrorResponse:
    """
    Build the canonical error response envelope used by the API.

    Notes:
    - `request_id` must match the `X-Request-ID` response header.
    - `details` is optional structured metadata (dict/list/str), or None.
    """
    return ErrorResponse(
        error={
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details,
        }
    )

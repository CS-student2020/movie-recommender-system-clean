from __future__ import annotations

from typing import Any, Dict, Optional

from .error_codes import ErrorCode


def _error_example(
    *,
    code: ErrorCode,
    message: str,
    request_id: str = "7b2b5a2c4f3a4e1fb7f4f44c9c1c2c9a",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a canonical error example matching the runtime error envelope:
    {"error": {"code": "...", "message": "...", "request_id": "...", "details": {...}}}
    """
    payload: Dict[str, Any] = {
        "error": {
            "code": code.value,
            "message": message,
            "request_id": request_id,
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return payload


def standard_error_responses() -> Dict[int, Dict[str, Any]]:
    """
    Reusable, BigTech-style error response docs for FastAPI routes.
    - Central source of truth for error documentation
    - Examples are generated from ErrorCode (no scattered raw strings)
    """
    return {
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": _error_example(
                        code=ErrorCode.BAD_REQUEST,
                        message="Bad request",
                    )
                }
            },
        },
        404: {
            "description": "Not found",
            "content": {
                "application/json": {
                    "example": _error_example(
                        code=ErrorCode.NOT_FOUND,
                        message="No recommendations found for user 404",
                    )
                }
            },
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": _error_example(
                        code=ErrorCode.VALIDATION_ERROR,
                        message="Request validation failed",
                        details={
                            "errors": [
                                {
                                    "loc": ["query", "limit"],
                                    "msg": "Input should be greater than 0",
                                    "type": "value_error",
                                }
                            ]
                        },
                    )
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": _error_example(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="Internal Server Error",
                    )
                }
            },
        },
    }


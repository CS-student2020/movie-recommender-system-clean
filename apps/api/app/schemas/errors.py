from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str = Field(
        ...,
        description="Stable error code for clients",
        json_schema_extra={"example": "NOT_FOUND"},
    )
    message: str = Field(
        ...,
        json_schema_extra={"example": "No recommendations found for user 404"},
    )
    request_id: Optional[str] = Field(
        None,
        json_schema_extra={"example": "7b2b5a2c4f3a4e1fb7f4f44c9c1c2c9a"},
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional structured metadata",
    )


class ErrorResponse(BaseModel):
    error: ErrorInfo

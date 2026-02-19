"""
HTTP routes for recommendation-related operations.
No business logic lives here.
"""

from __future__ import annotations

import inspect

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from apps.api.app.openapi_examples import standard_error_responses
from apps.api.app.schemas.errors import ErrorResponse
from apps.api.app.schemas.recommendations import (
    RecommendationOut,
    RecommendationQueryParams,
)
from src.recommender.bootstrap import get_recommender_service
from src.recommender.logging_utils import configure_logger
from src.recommender.service.recommender_service import RecommenderService

logger = configure_logger(__name__)

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


def _is_ready(request: Request) -> bool:
    return bool(getattr(request.app.state, "is_ready", False))


def get_service(request: Request) -> RecommenderService:
    """
    Retrieve the recommender service from application state.

    In production, the service is initialized in background bootstrap and stored on
    `app.state.recommender_service`. If the app is not ready yet, callers should
    receive a 503 (handled in the route).

    For dev/test contexts where startup hooks may not have executed, we fall back
    to a cached bootstrap.
    """
    service = getattr(request.app.state, "recommender_service", None)
    if service is not None:
        return service

    # Dev/Test fallback only (e.g., TestClient contexts)
    return get_recommender_service()


@router.get(
    "/{user_id}",
    status_code=status.HTTP_200_OK,
    summary="Get movie recommendations for a user",
    response_model=list[RecommendationOut],
    response_model_exclude_none=True,
    responses={
        **{
            code: {"model": ErrorResponse, **spec}
            for code, spec in standard_error_responses().items()
        },
        503: {
            "model": ErrorResponse,
            "description": "Service not ready",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "SERVICE_UNAVAILABLE",
                            "message": "Service is warming up. Please retry shortly.",
                            "request_id": "7b2b5a2c4f3a4e1fb7f4f44c9c1c2c9a",
                        }
                    }
                }
            },
        },
    },
)
def get_recommendations(
    request: Request,
    user_id: int,
    params: RecommendationQueryParams = Depends(),
    service: RecommenderService = Depends(get_service),
):
    # FAANG-grade: reject traffic until bootstrap is complete
    if not _is_ready(request):
        # Keep response shape aligned with ErrorResponse schema:
        # {"error": {"code": "...", "message": "...", "request_id": "...", ...}}
        request_id = request.headers.get("X-Request-ID") or getattr(
            request.state, "request_id", None
        )
        payload = {
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": "Service is warming up. Please retry shortly.",
                "request_id": request_id,
            }
        }
        return JSONResponse(status_code=503, content=payload)

    logger.info(
        "recommendation_request",
        extra={
            "event": "recommendations.request",
            "user_id": user_id,
            **params.model_dump(exclude_none=True),
        },
    )

    kwargs = {
        "user_id": user_id,
        **params.model_dump(exclude_none=True),
    }

    # Contract guard: only pass parameters supported by the service
    sig = inspect.signature(service.get_recommendations_for_user)
    supported = set(sig.parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in supported}

    try:
        recommendations = service.get_recommendations_for_user(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user {user_id}",
        )

    return recommendations

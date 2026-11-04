"""HTTP routes.

Why: An `APIRouter` keeps endpoints separate from app setup in `ai_ops.api.app`.

Used by: `ai_ops.api.app.create_app`, which includes this router.
"""

from fastapi import APIRouter

from ai_ops.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health")
async def health() -> HealthResponse:
    """Report that the API process is running.

    Why: Gives people, scripts, and the demo a cheap way to check the API is up. It calls
    no external service, so it still answers when an LLM provider is down.

    Used by: API clients over `GET /health`.

    Returns:
        A response with status `ok`.
    """
    return HealthResponse(status="ok")

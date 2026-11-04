"""HTTP routes.

Why: An `APIRouter` keeps endpoints separate from app setup in `ai_ops.api.app`. Routes only
translate between HTTP and services. They hold no business logic.

Used by: `ai_ops.api.app.create_app`, which includes this router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ai_ops.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    SourceResponse,
    UsageResponse,
)
from ai_ops.services.rag import RagService

router = APIRouter()


def get_rag_service(request: Request) -> RagService:
    """Return the app's RAG service.

    Why: With FastAPI's `Depends`, a route declares what it needs and does not have to know
    where it comes from. Routes stay short, and the service is built in one place.

    Used by: FastAPI, for the routes that declare `RagDependency`.

    Args:
        request: The current request.

    Returns:
        The service created by `create_app`.
    """
    return request.app.state.rag_service


RagDependency = Annotated[RagService, Depends(get_rag_service)]


@router.get("/health")
async def health(rag: RagDependency) -> HealthResponse:
    """Report that the API process is running and how many documents loaded.

    Why: Gives people, scripts, and the demo a cheap way to check the API is up. It calls no
    external service, so it still answers when an LLM provider is down.

    Used by: API clients over `GET /health`.

    Args:
        rag: The RAG service.

    Returns:
        Status `ok` and the document count.
    """
    return HealthResponse(status="ok", documents=rag.document_count)


@router.post(
    "/ask",
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ask(body: AskRequest, rag: RagDependency) -> AskResponse:
    """Answer a question from the knowledge base.

    Why: This is the main feature of the assistant. The extra `responses` entries put the
    real error format in the generated docs, in place of FastAPI's default.

    Used by: API clients over `POST /ask`.

    Args:
        body: The validated request.
        rag: The RAG service.

    Returns:
        The answer with its sources, token usage, and latency.

    Raises:
        LLMError: A model call failed. `ai_ops.api.errors` maps it to 502 or 504.
    """
    result = await rag.ask(body.question)
    return AskResponse(
        answer=result.answer,
        found=result.found,
        sources=[
            SourceResponse(
                id=source.chunk.source_id,
                document_id=source.chunk.document_id,
                position=source.chunk.position,
                score=source.score,
                text=source.chunk.text,
            )
            for source in result.sources
        ],
        usage=UsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        ),
        latency_ms=result.latency_ms,
    )

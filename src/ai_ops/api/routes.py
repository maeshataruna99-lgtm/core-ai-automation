"""HTTP routes.

Why: An `APIRouter` keeps endpoints separate from app setup in `ai_ops.api.app`. Routes only
translate between HTTP and services. They hold no business logic.

Used by: `ai_ops.api.app.create_app`, which includes this router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ai_ops.agents.loop import Agent
from ai_ops.api.schemas import (
    ApprovalDecisionRequest,
    ApprovalResponse,
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    SourceResponse,
    TicketRequest,
    TicketResponse,
    ToolCallResponse,
    UsageResponse,
)
from ai_ops.services.approval import ApprovalService
from ai_ops.services.rag import RagService
from ai_ops.services.workflow import TicketWorkflow

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


def get_agent(request: Request) -> Agent:
    """Return the app's agent.

    Why: See `get_rag_service`.

    Used by: FastAPI, for `chat`.

    Args:
        request: The current request.

    Returns:
        The agent created by `create_app`.
    """
    return request.app.state.agent


def get_ticket_workflow(request: Request) -> TicketWorkflow:
    """Return the app's ticket workflow.

    Why: See `get_rag_service`.

    Used by: FastAPI, for `run_ticket_workflow`.

    Args:
        request: The current request.

    Returns:
        The workflow created by `create_app`.
    """
    return request.app.state.ticket_workflow


def get_approvals(request: Request) -> ApprovalService:
    """Return the app's approval service.

    Why: See `get_rag_service`.

    Used by: FastAPI, for `decide_approval`.

    Args:
        request: The current request.

    Returns:
        The service created by `create_app`.
    """
    return request.app.state.approvals


RagDependency = Annotated[RagService, Depends(get_rag_service)]
AgentDependency = Annotated[Agent, Depends(get_agent)]
WorkflowDependency = Annotated[TicketWorkflow, Depends(get_ticket_workflow)]
ApprovalsDependency = Annotated[ApprovalService, Depends(get_approvals)]


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


@router.post(
    "/chat",
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat(body: ChatRequest, agent: AgentDependency) -> ChatResponse:
    """Run the agent for one message.

    Why: This is how staff use the assistant's tools. Each call starts a new conversation, because
    keeping history is not needed for the MVP.

    Used by: API clients over `POST /chat`.

    Args:
        body: The validated request.
        agent: The agent.

    Returns:
        The reply, why the agent stopped, its tool calls, and any actions awaiting approval.

    Raises:
        LLMError: A model call failed. `ai_ops.api.errors` maps it to 502 or 504.
    """
    result = await agent.run(body.message)
    return ChatResponse(
        reply=result.reply,
        stop_reason=result.stop_reason,
        tool_calls=[
            ToolCallResponse(
                name=call.name,
                arguments=call.arguments,
                status=call.status,
                approval_id=call.approval_id,
            )
            for call in result.tool_calls
        ],
        pending_approvals=[call.approval_id for call in result.tool_calls if call.approval_id],
        usage=UsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        ),
        latency_ms=result.latency_ms,
    )


@router.post(
    "/workflows/ticket",
    responses={
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def run_ticket_workflow(body: TicketRequest, workflow: WorkflowDependency) -> TicketResponse:
    """Run the ticket workflow, or report its state if it already ran.

    Why: Posting the same ticket again is safe, and it continues after a failure. A retry can use
    the same request, so a client needs no extra logic.

    Used by: API clients over `POST /workflows/ticket`.

    Args:
        body: The validated request.
        workflow: The workflow.

    Returns:
        The ticket's state, with the draft and the approval id.

    Raises:
        WorkflowConflictError: The id was already used with different text. Mapped to a 409.
        LLMError: A model call failed. Mapped to a 502 or 504.
    """
    state = await workflow.run(body.ticket_id, body.message)
    return TicketResponse(
        ticket_id=state.ticket_id,
        status=state.status,
        category=state.category,
        record_id=state.record_id,
        draft=state.draft,
        approval_id=state.approval_id,
    )


@router.post(
    "/approvals/{approval_id}",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def decide_approval(
    approval_id: str, body: ApprovalDecisionRequest, approvals: ApprovalsDependency
) -> ApprovalResponse:
    """Approve or reject a pending action.

    Why: This is the only way to run a write action, and it is not available to the model as a
    tool. The MVP has no login, so this relies on the API being bound to localhost.

    Used by: API clients over `POST /approvals/{approval_id}`.

    Args:
        approval_id: The approval id.
        body: The validated decision.
        approvals: The approval service.

    Returns:
        The approval in its new state.

    Raises:
        ApprovalNotFoundError: No approval has this id. Mapped to a 404.
        ApprovalConflictError: The decision does not fit the state. Mapped to a 409.
        ActionFailedError: The action failed, and can be retried. Mapped to a 502.
    """
    approval = await approvals.decide(approval_id, approve=body.decision == "approve")
    return ApprovalResponse(
        id=approval.id,
        kind=approval.kind,
        payload=approval.payload,
        status=approval.status,
        result=approval.result,
    )

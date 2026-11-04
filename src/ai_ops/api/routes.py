"""HTTP routes.

Why: An `APIRouter` keeps endpoints separate from app setup in `ai_ops.api.app`. Routes only
translate between HTTP and services. They hold no business logic.

Used by: `ai_ops.api.app.create_app`, which includes this router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi import Path as FastAPIPath
from fastapi.responses import RedirectResponse

from ai_ops.agents.loop import Agent
from ai_ops.api.schemas import (
    ApprovalDecisionRequest,
    ApprovalResponse,
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ConnectionCheckRequest,
    ConnectionCheckResponse,
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    LlmSettingsResponse,
    RecordDraftResponse,
    RecordRequest,
    RecordResponse,
    SelectModelRequest,
    SourceResponse,
    TicketRequest,
    TicketResponse,
    ToolCallResponse,
    UsageResponse,
)
from ai_ops.integrations.records import (
    Record,
    RecordInserter,
    RecordNotFoundError,
    RecordRepository,
)
from ai_ops.services.approval import Approval, ApprovalService
from ai_ops.services.documents import DocumentService
from ai_ops.services.extraction import RecordExtractor
from ai_ops.services.llm_settings import LlmSettingsService
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


def get_records(request: Request) -> RecordRepository:
    """Return the app's record repository.

    Why: See `get_rag_service`.

    Used by: FastAPI, for `get_record`.

    Args:
        request: The current request.

    Returns:
        The repository created by `create_app`.
    """
    return request.app.state.records


def get_documents(request: Request) -> DocumentService:
    """Return the app's document service.

    Why: See `get_rag_service`.

    Used by: FastAPI, for the document routes.

    Args:
        request: The current request.

    Returns:
        The service created by `create_app`.
    """
    return request.app.state.documents


def get_extractor(request: Request) -> RecordExtractor:
    """Return the app's order extractor.

    Why: See `get_rag_service`.

    Used by: FastAPI, for `extract_record`.

    Args:
        request: The current request.

    Returns:
        The extractor created by `create_app`.
    """
    return request.app.state.extractor


def get_llm_settings(request: Request) -> LlmSettingsService:
    """Return the app's LLM settings service.

    Why: See `get_rag_service`.

    Used by: FastAPI, for the settings routes.

    Args:
        request: The current request.

    Returns:
        The service created by `create_app`.
    """
    return request.app.state.llm_settings


def get_record_inserter(request: Request) -> RecordInserter:
    """Return the storage's inserting interface.

    Why: See `get_rag_service`. The agent's tools get a narrower interface, with no insert.

    Used by: FastAPI, for `add_record` and `list_records`.

    Args:
        request: The current request.

    Returns:
        The repository created by `create_app`.
    """
    return request.app.state.records


DocumentsDependency = Annotated[DocumentService, Depends(get_documents)]
ExtractorDependency = Annotated[RecordExtractor, Depends(get_extractor)]
LlmSettingsDependency = Annotated[LlmSettingsService, Depends(get_llm_settings)]
InserterDependency = Annotated[RecordInserter, Depends(get_record_inserter)]
RagDependency = Annotated[RagService, Depends(get_rag_service)]
RecordsDependency = Annotated[RecordRepository, Depends(get_records)]
AgentDependency = Annotated[Agent, Depends(get_agent)]
WorkflowDependency = Annotated[TicketWorkflow, Depends(get_ticket_workflow)]
ApprovalsDependency = Annotated[ApprovalService, Depends(get_approvals)]


@router.get("/", include_in_schema=False)
async def open_interface() -> RedirectResponse:
    """Send a browser to the web interface.

    Why: Opening `http://127.0.0.1:8000` should show the app, not a 404. This is left out of the
    API docs because it is not part of the API.

    Used by: people opening the root address in a browser.

    Returns:
        A redirect to `/ui/`.
    """
    return RedirectResponse("/ui/")


@router.get("/health")
async def health(request: Request, rag: RagDependency) -> HealthResponse:
    """Report that the API process is running, how many documents loaded, and the provider.

    Why: Gives people, scripts, and the demo a cheap way to check the API is up. It calls no
    external service, so it still answers when an LLM provider is down.

    Used by: API clients over `GET /health`, and the header of the web interface.

    Args:
        request: The current request.
        rag: The RAG service.

    Returns:
        Status `ok`, the document count, and the configured provider's name.
    """
    return HealthResponse(
        status="ok", documents=rag.document_count, provider=request.app.state.provider_name
    )


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


def _record_response(record: Record) -> RecordResponse:
    """Convert a record to its API model.

    Why: Several routes return the same shape.

    Used by: the record routes.

    Args:
        record: The record.

    Returns:
        The response model.
    """
    return RecordResponse(
        id=record.id, customer=record.customer, item=record.item, status=record.status
    )


@router.get("/documents")
async def list_documents(documents: DocumentsDependency) -> list[DocumentResponse]:
    """List the documents in the knowledge base.

    Why: A person needs to see what the assistant can use.

    Used by: the Insert Data page, and other API clients.

    Args:
        documents: The document service.

    Returns:
        The documents, sorted by id.
    """
    return [DocumentResponse(**item) for item in documents.list_all()]


@router.post(
    "/documents",
    status_code=201,
    responses={
        409: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def add_document(body: DocumentRequest, documents: DocumentsDependency) -> DocumentResponse:
    """Add a document, which can be found at once.

    Why: A person can extend the knowledge base with no restart. The model has no tool for this.

    Used by: the Insert Data page, and other API clients.

    Args:
        body: The validated request.
        documents: The document service.

    Returns:
        The new document.

    Raises:
        DocumentExistsError: The name is already used. Mapped to a 409.
        LLMError: Embedding failed. Mapped to a 502 or 504, and nothing is kept.
    """
    document_id, chunks = await documents.add(body.name, body.text)
    return DocumentResponse(id=document_id, chunks=chunks, removable=True)


@router.delete(
    "/documents/{name}",
    status_code=204,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def remove_document(
    name: Annotated[str, FastAPIPath(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.(md|txt)$")],
    documents: DocumentsDependency,
) -> None:
    """Remove an uploaded document.

    Why: A person must be able to take back what they added. Repository documents are out of reach.

    Used by: the Insert Data page, and other API clients.

    Args:
        name: The file name.
        documents: The document service.

    Raises:
        DocumentNotFoundError: No uploaded document has this name. Mapped to a 404.
    """
    documents.remove(name)


@router.get("/records")
async def list_records(records: InserterDependency) -> list[RecordResponse]:
    """List all orders.

    Why: A person needs to see what was added.

    Used by: the Records and Insert Data pages, and other API clients.

    Args:
        records: The repository.

    Returns:
        The orders, sorted by id.
    """
    return [_record_response(record) for record in records.list_all()]


@router.post(
    "/records",
    status_code=201,
    responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def add_record(body: RecordRequest, records: InserterDependency) -> RecordResponse:
    """Add a new order.

    Why: Only a person can do this. The agent has no insert tool, and any change to an existing
    order
    still needs an approval.

    Used by: the Insert Data page, and other API clients.

    Args:
        body: The validated request.
        records: The repository.

    Returns:
        The new order.

    Raises:
        RecordExistsError: The id is already used. Mapped to a 409.
    """
    record = Record(id=body.id, customer=body.customer, item=body.item, status=body.status)
    records.add(record)
    return _record_response(record)


@router.post(
    "/records/extract",
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def extract_record(body: ExtractRequest, extractor: ExtractorDependency) -> ExtractResponse:
    """Read a draft order from free text with the model. This saves nothing.

    Why: It saves typing, while a person stays in control of what is stored.

    Used by: the Insert Data page, and other API clients.

    Args:
        body: The validated request.
        extractor: The extractor.

    Returns:
        The draft, and whether anything could be read.

    Raises:
        LLMError: The model call failed. Mapped to a 502 or 504.
    """
    result = await extractor.extract(body.text)
    draft = result.draft
    return ExtractResponse(
        found=result.found,
        draft=RecordDraftResponse(
            id=draft.id, customer=draft.customer, item=draft.item, status=draft.status
        ),
        message=result.message,
        usage=UsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        ),
    )


@router.get("/settings/llm")
async def get_llm_settings_status(settings: LlmSettingsDependency) -> LlmSettingsResponse:
    """Show the provider and model, with no secrets.

    Why: A person must be able to see what answers their questions.

    Used by: the Settings page and the header, and other API clients.

    Args:
        settings: The settings service.

    Returns:
        The current state.
    """
    return LlmSettingsResponse(**settings.status())


@router.put(
    "/settings/llm",
    responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def select_model(
    body: SelectModelRequest, settings: LlmSettingsDependency
) -> LlmSettingsResponse:
    """Choose the model. The next model call uses it, and the choice survives a restart.

    Why: Models differ in price and strength, and changing one should not need a restart.

    Used by: the Settings page, and other API clients.

    Args:
        body: The validated request.
        settings: The settings service.

    Returns:
        The new state.

    Raises:
        ModelNotSelectableError: This provider's model cannot be changed. Mapped to a 409.
        UnknownModelError: The model is not on the list. Mapped to a 422.
    """
    settings.select_model(body.model)
    return LlmSettingsResponse(**settings.status())


@router.post("/settings/llm/check")
async def check_connection(
    body: ConnectionCheckRequest, settings: LlmSettingsDependency
) -> ConnectionCheckResponse:
    """Check that the provider really works.

    Why: A failed connection is the answer to the question, so it is a 200 with `connected` false.

    Used by: the Settings page and the header, and other API clients.

    Args:
        body: The validated request. `send_message` also sends a short message, which costs tokens.
        settings: The settings service.

    Returns:
        The result.
    """
    report = await settings.check(body.send_message)
    usage = report.usage
    return ConnectionCheckResponse(
        connected=report.connected,
        detail=report.detail,
        remote_models=list(report.remote_models),
        message_sent=report.message_sent,
        model_answered=report.model_answered,
        usage=UsageResponse(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        if usage
        else None,
    )


@router.get(
    "/records/{record_id}",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def get_record(
    record_id: Annotated[str, FastAPIPath(pattern=r"^ORD-[0-9]+$")], records: RecordsDependency
) -> RecordResponse:
    """Read one order.

    Why: Staff can check a record directly, for example after approving a change. This reads
    customer data with no login, which is another reason the API stays bound to localhost.

    Used by: API clients over `GET /records/{record_id}`.

    Args:
        record_id: The order id, such as `ORD-1001`.
        records: The repository.

    Returns:
        The order.

    Raises:
        RecordNotFoundError: The order does not exist. Mapped to a 404.
    """
    record = records.get(record_id)
    if record is None:
        raise RecordNotFoundError(f"Record {record_id} does not exist")
    return RecordResponse(
        id=record.id, customer=record.customer, item=record.item, status=record.status
    )


def _approval_response(approval: Approval) -> ApprovalResponse:
    """Convert an approval to its API model.

    Why: Both approval routes return the same shape.

    Used by: `list_approvals` and `decide_approval`.

    Args:
        approval: The approval.

    Returns:
        The response model.
    """
    return ApprovalResponse(
        id=approval.id,
        kind=approval.kind,
        payload=approval.payload,
        status=approval.status,
        result=approval.result,
        created_at=approval.created_at,
    )


@router.get("/approvals")
async def list_approvals(approvals: ApprovalsDependency) -> list[ApprovalResponse]:
    """List all approvals, newest first.

    Why: This gives a person one place to see what is waiting. It has no login, like the rest of
    the API, which is another reason it stays bound to localhost.

    Used by: the Approvals page of the web interface, and other API clients.

    Args:
        approvals: The approval service.

    Returns:
        The approvals.
    """
    return [_approval_response(approval) for approval in approvals.list_all()]


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
    return _approval_response(approval)

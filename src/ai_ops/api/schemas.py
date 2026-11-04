"""Request and response models for the HTTP API.

Why: Pydantic models validate the data and generate the OpenAPI docs, so the documented
contract and the actual responses cannot drift apart.

Used by: `ai_ops.api.routes` for requests and responses, `ai_ops.api.errors` for error bodies.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Body of `GET /health`.

    Why: Clients get a typed liveness answer. `documents` shows whether the knowledge base
    loaded, because a wrong directory only logs a warning and does not stop the API.

    Used by: `ai_ops.api.routes.health`.
    """

    status: Literal["ok"]
    documents: int


class AskRequest(BaseModel):
    """Body of `POST /ask`.

    Why: The limits apply before any model call. A blank question has nothing to search for,
    and a very long one costs tokens. Whitespace is removed first, so "   " counts as blank.

    Used by: `ai_ops.api.routes.ask`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)


class SourceResponse(BaseModel):
    """One retrieved chunk used for an answer.

    Why: Showing the text and score lets a person check the answer against its source. Clients
    must render `text` as plain text, because it comes from documents.

    Used by: `AskResponse`.
    """

    id: str
    document_id: str
    position: int
    score: float
    text: str


class UsageResponse(BaseModel):
    """Token counts for one API call.

    Why: Tokens are the main cost of an LLM application, so each answer reports them.

    Used by: `AskResponse`.
    """

    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    """Body of a successful `POST /ask`.

    Why: `found` is false when the knowledge base has no relevant text. That is a normal result,
    so it is a 200 and not an error.

    Used by: `ai_ops.api.routes.ask`.
    """

    answer: str
    found: bool
    sources: list[SourceResponse]
    usage: UsageResponse
    latency_ms: float


class ChatRequest(BaseModel):
    """Body of `POST /chat`.

    Why: The limit matters more than for `/ask`, because the message is sent again on every step
    of the agent's loop.

    Used by: `ai_ops.api.routes.chat`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)


class ToolCallResponse(BaseModel):
    """One tool request the agent made, and what happened to it.

    Why: A person should be able to see what the agent actually did, and not only what its reply
    claims it did.

    Used by: `ChatResponse`.
    """

    name: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "pending_approval"]
    approval_id: str | None


class ChatResponse(BaseModel):
    """Body of a successful `POST /chat`.

    Why: `stop_reason` shows whether the agent finished, and `pending_approvals` lists actions that
    are waiting for a person. Reaching a limit is a normal result, so it is still a 200.

    Used by: `ai_ops.api.routes.chat`.
    """

    reply: str
    stop_reason: Literal["completed", "max_steps", "timeout"]
    tool_calls: list[ToolCallResponse]
    pending_approvals: list[str]
    usage: UsageResponse
    latency_ms: float


class TicketRequest(BaseModel):
    """Body of `POST /workflows/ticket`.

    Why: The id is an idempotency key and ends up in logs, so it is limited to safe characters.

    Used by: `ai_ops.api.routes.run_ticket_workflow`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    message: str = Field(min_length=1, max_length=4000)


class TicketResponse(BaseModel):
    """Body of a successful `POST /workflows/ticket`.

    Why: Holds what a person needs to review the draft, and the id to approve or reject it.

    Used by: `ai_ops.api.routes.run_ticket_workflow`.
    """

    ticket_id: str
    status: Literal["pending_approval", "sending", "sent", "rejected", "send_failed"]
    category: str
    record_id: str | None
    draft: str
    approval_id: str


class ApprovalDecisionRequest(BaseModel):
    """Body of `POST /approvals/{id}`.

    Why: Only these two exact words are accepted, so an unclear value can never count as approval.

    Used by: `ai_ops.api.routes.decide_approval`.
    """

    decision: Literal["approve", "reject"]


class ApprovalResponse(BaseModel):
    """An approval and its current state.

    Why: `kind` and `payload` show exactly what is being approved.

    Used by: `ai_ops.api.routes.decide_approval`.
    """

    id: str
    kind: str
    payload: dict[str, Any]
    status: Literal["pending", "executing", "executed", "rejected", "failed"]
    result: dict[str, Any] | None


class RecordResponse(BaseModel):
    """Body of `GET /records/{id}`.

    Why: A separate model decides exactly what leaves the API, independent of the storage.

    Used by: `ai_ops.api.routes.get_record`.
    """

    id: str
    customer: str
    item: str
    status: Literal["processing", "shipped", "delivered", "cancelled"]


class ErrorDetail(BaseModel):
    """One invalid field in a validation error.

    Why: Clients can show the problem next to the right input.

    Used by: `ai_ops.api.errors`.
    """

    field: str
    message: str


class ErrorBody(BaseModel):
    """The content of an error response.

    Why: `code` is stable for programs, `message` is for people, and `request_id` links the
    error to the logs.

    Used by: `ErrorResponse`.
    """

    code: str
    message: str
    request_id: str | None
    details: list[ErrorDetail]


class ErrorResponse(BaseModel):
    """The body of every error response.

    Why: Clients parse a single shape for all errors.

    Used by: `ai_ops.api.errors.error_response`.
    """

    error: ErrorBody

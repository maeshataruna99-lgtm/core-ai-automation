"""Request and response models for the HTTP API.

Why: Pydantic models validate the data and generate the OpenAPI docs, so the documented
contract and the actual responses cannot drift apart.

Used by: `ai_ops.api.routes` for requests and responses, `ai_ops.api.errors` for error bodies.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """Body of `GET /health`.

    Why: Clients get a typed liveness answer. `documents` shows whether the knowledge base
    loaded, because a wrong directory only logs a warning and does not stop the API. `provider`
    lets the interface show when answers come from the mock.

    Used by: `ai_ops.api.routes.health`.
    """

    status: Literal["ok"]
    documents: int
    provider: str


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
    created_at: str


class RecordResponse(BaseModel):
    """Body of `GET /records/{id}`.

    Why: A separate model decides exactly what leaves the API, independent of the storage.

    Used by: `ai_ops.api.routes.get_record`.
    """

    id: str
    customer: str
    item: str
    status: Literal["processing", "shipped", "delivered", "cancelled"]


# Windows treats these names as devices, whatever the file type.
_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{device}{number}" for device in ("com", "lpt") for number in range(1, 10)
}


class DocumentName(BaseModel):
    """A safe file name for an uploaded document.

    Why: The name becomes a file path, so it may only hold letters, digits, dashes, and underscores,
    with one `.md` or `.txt` ending. That rules out folders, `..`, and hidden files such as `.env`.

    Used by: `DocumentRequest`.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.(md|txt)$")

    @field_validator("name")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        """Reject names that Windows reserves.

        Why: Writing to `con.md` can fail or behave oddly on Windows.

        Used by: Pydantic.

        Args:
            value: The name.

        Returns:
            The name, unchanged.

        Raises:
            ValueError: The name is reserved.
        """
        if value.rsplit(".", 1)[0].lower() in _RESERVED_NAMES:
            raise ValueError("this name is reserved by Windows")
        return value


class DocumentRequest(DocumentName):
    """Body of `POST /documents`.

    Why: The text is kept exactly as sent, so whitespace is only checked, and is not trimmed.

    Used by: `ai_ops.api.routes.add_document`.
    """

    text: str = Field(max_length=50_000)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject text with nothing in it.

        Why: A blank document has no chunks, so it could never be found.

        Used by: Pydantic.

        Args:
            value: The text.

        Returns:
            The text, unchanged.

        Raises:
            ValueError: The text is blank.
        """
        if not value.strip():
            raise ValueError("the text must not be empty")
        return value


class DocumentResponse(BaseModel):
    """One document in the knowledge base.

    Why: `removable` tells the page which ones may offer a delete button.

    Used by: `ai_ops.api.routes`.
    """

    id: str
    chunks: int
    removable: bool


class RecordRequest(BaseModel):
    """Body of `POST /records`.

    Why: The same rules apply to any new order. Whitespace is trimmed first, so "   " counts as
    empty.

    Used by: `ai_ops.api.routes.add_record`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(pattern=r"^ORD-[0-9]+$", max_length=32)
    customer: str = Field(min_length=1, max_length=100)
    item: str = Field(min_length=1, max_length=100)
    status: Literal["processing", "shipped", "delivered", "cancelled"]


class ExtractRequest(BaseModel):
    """Body of `POST /records/extract`.

    Why: Each call costs tokens, and sends the text to the provider, so it has a low limit.

    Used by: `ai_ops.api.routes.extract_record`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=2000)


class RecordDraftResponse(BaseModel):
    """A draft order that a model read. Any field can be empty.

    Why: These only fill in a form. Nothing is saved until a person sends `POST /records`.

    Used by: `ExtractResponse`.
    """

    id: str | None
    customer: str | None
    item: str | None
    status: str | None


class ExtractResponse(BaseModel):
    """Body of a successful `POST /records/extract`.

    Why: `found` is false when no order could be read, which is a normal result.

    Used by: `ai_ops.api.routes.extract_record`.
    """

    found: bool
    draft: RecordDraftResponse
    message: str
    usage: UsageResponse


class LlmSettingsResponse(BaseModel):
    """Body of `GET /settings/llm`.

    Why: This says whether a key exists, and never holds its value, because the API has no login.

    Used by: `ai_ops.api.routes`.
    """

    provider: str
    model: str | None
    models: list[str]
    selectable: bool
    key_configured: bool
    embeddings: str
    base_url: str | None


class SelectModelRequest(BaseModel):
    """Body of `PUT /settings/llm`.

    Why: The pattern keeps out line breaks and other text before the name is looked up.

    Used by: `ai_ops.api.routes.select_model`.
    """

    model: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")


class ConnectionCheckRequest(BaseModel):
    """Body of `POST /settings/llm/check`.

    Why: Sending a message costs tokens, so it must be asked for on purpose.

    Used by: `ai_ops.api.routes.check_connection`.
    """

    send_message: bool = False


class ConnectionCheckResponse(BaseModel):
    """Body of `POST /settings/llm/check`.

    Why: A failed connection is the answer to the question, so this is a 200 with `connected` false.

    Used by: `ai_ops.api.routes.check_connection`.
    """

    connected: bool
    detail: str
    remote_models: list[str]
    message_sent: bool
    model_answered: str | None
    usage: UsageResponse | None


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

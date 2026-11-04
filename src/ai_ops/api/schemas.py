"""Request and response models for the HTTP API.

Why: Pydantic models validate the data and generate the OpenAPI docs, so the documented
contract and the actual responses cannot drift apart.

Used by: `ai_ops.api.routes` for requests and responses, `ai_ops.api.errors` for error bodies.
"""

from typing import Literal

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

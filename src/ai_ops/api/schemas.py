"""Request and response models for the HTTP API.

Why: Pydantic models validate the data and generate the OpenAPI docs, so the documented
contract and the actual responses cannot drift apart.

Used by: `ai_ops.api.routes` for responses and `ai_ops.api.errors` for error bodies.
"""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Body of `GET /health`.

    Why: Clients get a typed liveness answer. Milestone 3 adds the loaded document count.

    Used by: `ai_ops.api.routes.health`.
    """

    status: Literal["ok"]


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

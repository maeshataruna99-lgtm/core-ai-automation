"""Map exceptions to one consistent JSON error format.

Why: FastAPI's default errors have different shapes for validation and HTTP errors, which
forces clients to handle both. The default validation error also repeats the rejected
input, and request bodies can contain personal data.

Used by: `ai_ops.api.app.create_app` calls `register_error_handlers`.
`ai_ops.api.middleware` calls `error_response` for unexpected exceptions.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ai_ops.api.schemas import ErrorBody, ErrorDetail, ErrorResponse
from ai_ops.integrations.records import RecordNotFoundError
from ai_ops.llm.provider import LLMError, LLMTimeoutError
from ai_ops.logging import LOGGER_NAME, request_id_var
from ai_ops.services.approval import (
    ActionFailedError,
    ApprovalConflictError,
    ApprovalNotFoundError,
)
from ai_ops.services.workflow import WorkflowConflictError

logger = logging.getLogger(f"{LOGGER_NAME}.api")


def error_response(
    status: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the JSON response for an error.

    Why: Every error path goes through here, so the body shape and the request id cannot
    differ between them.

    Used by: the handlers in this module and `RequestContextMiddleware`.

    Args:
        status: HTTP status code.
        code: Stable machine-readable error code.
        message: Text that is safe to show to the caller.
        details: Per-field problems, for validation errors.
        headers: Extra response headers, such as `Allow` on a 405.

    Returns:
        The response to send.
    """
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=request_id_var.get(),
            details=details or [],
        )
    )
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert a request validation failure to a 422 response.

    Why: This keeps the field location and the reason, and drops the rejected value.

    Used by: FastAPI, through `register_error_handlers`.

    Args:
        _: The request, required by the handler signature but not used.
        exc: The validation failure.

    Returns:
        A 422 response with one detail per invalid field.
    """
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in error["loc"]),
            message=error["msg"],
        )
        for error in exc.errors()
    ]
    return error_response(422, "validation_error", "The request is invalid.", details)


async def _handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    """Convert an HTTP error, such as 404 or 405, to the shared format.

    Why: The framework raises these itself for unknown routes and wrong methods, so they
    need the same shape as our own errors.

    Used by: FastAPI, through `register_error_handlers`.

    Args:
        _: The request, required by the handler signature but not used.
        exc: The HTTP error.

    Returns:
        A response with the same status and headers. The code comes from the status name,
        for example 404 becomes `not_found`.
    """
    code = HTTPStatus(exc.status_code).name.lower()
    return error_response(exc.status_code, code, str(exc.detail), headers=exc.headers)


async def _handle_llm_error(_: Request, exc: LLMError) -> JSONResponse:
    """Convert an LLM failure to a 504 (timeout) or 502 (anything else).

    Why: Gateway statuses tell the client the problem is upstream, not in their request. The
    client gets a fixed message. The internal one goes to the log, where it helps debugging.

    Used by: FastAPI, through `register_error_handlers`.

    Args:
        _: The request, required by the handler signature but not used.
        exc: The failure. `LLMError` messages are safe to log by design.

    Returns:
        A 502 or 504 response.

    Side effects:
        Writes one error log line.
    """
    logger.error("llm call failed", extra={"error": str(exc)})
    if isinstance(exc, LLMTimeoutError):
        return error_response(504, "upstream_timeout", "The LLM provider did not respond in time.")
    return error_response(502, "upstream_error", "The LLM provider failed.")


# Service errors with their HTTP status and code. Services do not know about HTTP, so this table
# keeps the translation in the API layer. Each of these messages is safe to show to a client.
_SERVICE_ERRORS: dict[type[Exception], tuple[int, str]] = {
    ApprovalNotFoundError: (404, "approval_not_found"),
    ApprovalConflictError: (409, "approval_conflict"),
    ActionFailedError: (502, "action_failed"),
    WorkflowConflictError: (409, "ticket_conflict"),
    RecordNotFoundError: (404, "record_not_found"),
}


async def _handle_service_error(_: Request, exc: Exception) -> JSONResponse:
    """Convert a service error using the table above.

    Why: One handler and one table are easier to read than a handler per error type.

    Used by: FastAPI, through `register_error_handlers`.

    Args:
        _: The request, required by the handler signature but not used.
        exc: The service error.

    Returns:
        The response for this error type.
    """
    status, code = _SERVICE_ERRORS[type(exc)]
    return error_response(status, code, str(exc))


def register_error_handlers(app: FastAPI) -> None:
    """Attach the error handlers to an app.

    Why: Keeps `create_app` short and all error mapping in this module. Unexpected exceptions
    are not handled here, because the handler for `Exception` runs outside the middleware
    stack, where the request id is no longer available. See `RequestContextMiddleware`.

    Used by: `ai_ops.api.app.create_app`.

    Args:
        app: The app to change.
    """
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(HTTPException, _handle_http_error)
    app.add_exception_handler(LLMError, _handle_llm_error)
    for error_type in _SERVICE_ERRORS:
        app.add_exception_handler(error_type, _handle_service_error)

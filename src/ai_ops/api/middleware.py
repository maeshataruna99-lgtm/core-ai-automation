"""Middleware for request ids, request logs, safe failures, and the body size limit.

Why: This is plain ASGI middleware, not `BaseHTTPMiddleware` or `@app.middleware("http")`.
Plain ASGI gives direct access to the outgoing response headers and the incoming body, and it
can catch an unexpected exception while the request id is still set.

Used by: `ai_ops.api.app.create_app`, which adds both classes to the app.
"""

import logging
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_ops.api.errors import error_response
from ai_ops.logging import LOGGER_NAME, request_id_var

logger = logging.getLogger(f"{LOGGER_NAME}.api")


class BodySizeLimitMiddleware:
    """Reject request bodies larger than a fixed limit with a 413.

    Why: Pydantic's length limits only run after the whole body is in memory, so a huge request
    needs stopping earlier. Bytes are counted as they arrive, because a chunked request has no
    `Content-Length` and a header can lie anyway.

    Used by: `ai_ops.api.app.create_app`, which adds it to the app.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        """Store the next ASGI app and the limit.

        Why: This is the standard ASGI middleware constructor.

        Used by: Starlette, when it builds the middleware stack.

        Args:
            app: The next app to call.
            max_bytes: The largest body allowed.
        """
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request on, counting body bytes as the app reads them.

        Why: Raising `HTTPException` from `receive` works because the app reads the body inside
        FastAPI's exception handling, so it becomes a normal error response.

        Used by: the ASGI server, through the middleware stack.

        Args:
            scope: Connection details.
            receive: Reads request messages.
            send: Writes response messages.

        Raises:
            HTTPException: The body is over the limit.
        """
        received = 0

        async def counting_receive() -> Message:
            """Read one message and enforce the limit."""
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise HTTPException(413, "The request body is too large.")
            return message

        await self.app(scope, counting_receive, send)


class RequestContextMiddleware:
    """Track one HTTP request from start to finish.

    Why: Request ids, request logs, and the last-resort 500 all need the same start and end
    points, so they live together.

    Used by: `ai_ops.api.app.create_app`. ASGI servers call it for every connection.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Store the next ASGI app in the chain.

        Why: This is the standard ASGI middleware constructor.

        Used by: Starlette, when it builds the middleware stack.

        Args:
            app: The next app to call.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection.

        Why: The id is generated here and any incoming `X-Request-ID` is ignored, so a
        client cannot put arbitrary text into the logs.

        Used by: the ASGI server, through the middleware stack.

        Args:
            scope: Connection details.
            receive: Reads request messages.
            send: Writes response messages.

        Raises:
            Exception: Re-raised only when the failure happens after the response started,
                because the status code has already been sent by then.

        Side effects:
            Writes one log line per request, and one more for an unexpected exception. The
            query string is not logged, because it can contain personal data.
        """
        # Lifespan and websocket messages are not HTTP requests.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started_at = time.perf_counter()
        status: int | None = None

        async def send_with_request_id(message: Message) -> None:
            """Add the request id header and remember the status code.

            Why: `http.response.start` is the only message that carries the headers and status.

            Used by: the inner app and the failure response, in place of `send`.

            Args:
                message: The ASGI message to send.
            """
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception("unhandled error")
            if status is not None:
                raise
            # The exception text stays in the log. The caller only gets the request id.
            response = error_response(500, "internal_error", "An internal error occurred.")
            await response(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request completed",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
                },
            )
            request_id_var.reset(token)

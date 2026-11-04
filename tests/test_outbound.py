"""Behaviour tests for the HTTP adapter of the outbound service.

Why: The adapter is the boundary with a third party, so its request format and each way the service
can fail must be pinned. `httpx2.MockTransport` runs the real request and parsing code offline.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import asyncio
import json

import httpx2
import pytest

from ai_ops.integrations.outbound import HttpOutboundAdapter, OutboundError, StubOutboundService

API_KEY = "outbound-test-secret"


def send_with(handler) -> dict:
    """Send one message through an adapter that talks to an in-memory handler.

    Why: Each test only changes how the fake service answers.

    Used by: the tests in this module.

    Args:
        handler: Function that turns a request into a response.

    Returns:
        The adapter's result.
    """

    async def scenario() -> dict:
        """Send and close the client in one event loop."""
        adapter = HttpOutboundAdapter(
            base_url="https://messages.example.test/v1",
            api_key=API_KEY,
            timeout_seconds=5,
            transport=httpx2.MockTransport(handler),
        )
        try:
            return await adapter.send("T-1", "Hello customer")
        finally:
            await adapter.aclose()

    return asyncio.run(scenario())


def test_send_uses_the_format_the_service_expects():
    """Check the outgoing request and the parsed result.

    Why: The idempotency key must travel as a header, or a retry could send a message twice.

    Used by: pytest.
    """
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Record the request, then accept it."""
        seen.update(
            url=str(request.url),
            auth=request.headers["Authorization"],
            key=request.headers["Idempotency-Key"],
            body=json.loads(request.content),
        )
        return httpx2.Response(201, json={"id": "msg-42"})

    result = send_with(handler)

    assert seen == {
        "url": "https://messages.example.test/v1/messages",
        "auth": f"Bearer {API_KEY}",
        "key": "T-1",
        "body": {"reference": "T-1", "body": "Hello customer"},
    }
    assert result == {"delivery_id": "msg-42", "delivered": True}


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_error_status_is_surfaced_with_its_code(status):
    """Check that a rejected message is reported.

    Why: The roadmap requires adapter failures to be surfaced. The status tells staff whether to
    retry or to fix the configuration.

    Used by: pytest.

    Args:
        status: The status the fake service returns.
    """
    with pytest.raises(OutboundError, match=str(status)):
        send_with(lambda request: httpx2.Response(status, json={"error": "nope"}))


def test_timeout_is_surfaced():
    """Check a service that does not answer.

    Why: The message may have gone out, so this must be reported, and retried with the same key.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Fail the way a slow server does."""
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(OutboundError, match="^Timeout calling"):
        send_with(handler)


def test_connection_failure_is_surfaced():
    """Check an unreachable service.

    Why: Callers should not need to know the HTTP library's exception types.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Fail the way an unreachable server does."""
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(OutboundError, match="ConnectError"):
        send_with(handler)


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, content=b"not json"),
        httpx2.Response(200, json={"unexpected": True}),
        httpx2.Response(200, json=["a", "list"]),
    ],
)
def test_unusable_success_response_is_surfaced(response):
    """Check a 200 that does not confirm delivery.

    Why: Treating it as sent would hide a failure from the person who approved the reply.

    Used by: pytest.

    Args:
        response: The response from the fake service.
    """
    with pytest.raises(OutboundError):
        send_with(lambda request: response)


def test_key_and_response_body_stay_out_of_the_error():
    """Check an error response that echoes the key and a customer's message.

    Why: This error text reaches a client and the logs, so it must hold neither.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Reject the call and echo everything back."""
        return httpx2.Response(401, json={"error": f"bad key {API_KEY}", "echo": "Hello customer"})

    with pytest.raises(OutboundError) as error:
        send_with(handler)

    assert API_KEY not in str(error.value)
    assert "Hello customer" not in str(error.value)


def test_stub_can_be_closed_like_the_adapter():
    """Check the shared interface.

    Why: The app closes whichever one it has at shutdown, with no special case.

    Used by: pytest.
    """
    asyncio.run(StubOutboundService().aclose())

"""Behaviour tests for the HTTP API.

Why: These tests send real HTTP requests through the full middleware stack, so they
keep passing when internals are refactored and only fail when API behaviour changes.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import json
from pathlib import Path

import httpx2
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from ai_ops.api.app import create_app
from ai_ops.config import Settings
from ai_ops.llm.mock import MockProvider
from ai_ops.llm.openai_compatible import OpenAICompatibleProvider
from ai_ops.llm.provider import ChatMessage, GenerationResult, LLMProviderError, LLMTimeoutError

FIXTURES = Path(__file__).parent / "fixtures"
# Tests that are not about RAG use no documents, so startup never calls the provider's `embed`.
NO_DOCS_SETTINGS = Settings(llm_provider="mock", knowledge_dir=FIXTURES / "does-not-exist")
DOCS_SETTINGS = Settings(llm_provider="mock", knowledge_dir=FIXTURES / "knowledge")
REFUND_QUESTION = "How many days do I have for refunds after purchase?"


@pytest.fixture
def client():
    """Provide a client for an app with the two fixture documents.

    Why: `create_app` takes settings as an argument, so tests do not touch environment
    variables. The `with` block also runs app startup and shutdown.

    Used by: tests in this module that need only the real routes.
    """
    with TestClient(create_app(DOCS_SETTINGS)) as test_client:
        yield test_client


class EchoRequest(BaseModel):
    """Body of the test-only validation route.

    Why: Milestone 1 has no endpoint that takes a body yet. A route defined in the test
    proves the error handling without shipping a fake endpoint.

    Used by: the `client_with_test_routes` fixture.
    """

    message: str = Field(min_length=1)


def build_app_with_test_routes() -> FastAPI:
    """Build an app with extra routes that trigger each failure.

    Why: The real API cannot fail on demand yet.

    Used by: the `client_with_test_routes` fixture and the crash logging test.

    Returns:
        The app, with `POST /test/echo` and `GET /test/crash` added.
    """
    app = create_app(NO_DOCS_SETTINGS)

    @app.post("/test/echo")
    async def echo(body: EchoRequest) -> EchoRequest:
        """Return the body, so only validation can fail."""
        return body

    @app.get("/test/crash")
    async def crash() -> None:
        """Fail with text that must never reach the caller."""
        raise RuntimeError("database password is hunter2")

    return app


@pytest.fixture
def client_with_test_routes():
    """Provide a client for the app with failure routes.

    Why: `raise_server_exceptions=False` makes the client return the 500 response a real
    caller would see, instead of re-raising the exception in the test.

    Used by: the error mapping tests in this module.
    """
    with TestClient(build_app_with_test_routes(), raise_server_exceptions=False) as test_client:
        yield test_client


def test_invalid_input_returns_422_with_clear_message(client_with_test_routes):
    """Check the validation error body.

    Why: The caller needs to see which field is wrong and why. The rejected value is left
    out, because request bodies can contain personal data.

    Used by: pytest.
    """
    response = client_with_test_routes.post("/test/echo", json={"message": ""})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"] == [
        {"field": "body.message", "message": "String should have at least 1 character"}
    ]


def test_unknown_route_returns_404_in_the_same_format(client):
    """Check that framework errors use the shared error format.

    Why: Clients should parse one error shape, whatever produced the error.

    Used by: pytest.
    """
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_unhandled_error_returns_500_without_leaking_details(client_with_test_routes):
    """Check the response for an unexpected exception.

    Why: Exception text can hold secrets or internals, so the caller gets a fixed message.

    Used by: pytest.
    """
    response = client_with_test_routes.get("/test/crash")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "hunter2" not in response.text


def read_log_lines(capsys) -> list[dict]:
    """Parse the JSON log lines written to stderr so far.

    Why: Reading stderr tests what an operator sees, not how the logger is built.

    Used by: the logging tests in this module.

    Args:
        capsys: Pytest's output capture fixture.

    Returns:
        One dict per log line.
    """
    return [json.loads(line) for line in capsys.readouterr().err.splitlines()]


def test_request_log_has_timestamp_severity_and_request_id(capsys):
    """Check the per-request log line against the response.

    Why: The request id in the header must match the log, otherwise a user report cannot
    be traced to its log lines. The app is built inside the test because the log handler
    binds to `sys.stderr` in `create_app`, and only the test body sees the `capsys` stream.

    Used by: pytest.
    """
    with TestClient(create_app(NO_DOCS_SETTINGS)) as client:
        response = client.get("/health?email=person@example.com")

    request_id = response.headers["X-Request-ID"]
    line = next(log for log in read_log_lines(capsys) if log["message"] == "request completed")
    assert line["request_id"] == request_id
    assert line["severity"] == "INFO"
    assert line["timestamp"]
    assert line["method"] == "GET"
    assert line["path"] == "/health"
    assert line["status"] == 200
    assert "person@example.com" not in str(line)


def test_each_request_gets_its_own_id(client):
    """Check that request ids differ.

    Why: A repeated id would merge the logs of different requests.

    Used by: pytest.
    """
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]

    assert first != second


def test_crash_is_logged_with_the_id_returned_to_the_caller(capsys):
    """Check that a 500 can be traced from the response to the log.

    Why: The caller only gets a fixed message, so the request id is the only link to the
    real exception in the logs.

    Used by: pytest.
    """
    with TestClient(build_app_with_test_routes(), raise_server_exceptions=False) as client:
        response = client.get("/test/crash")

    request_id = response.json()["error"]["request_id"]
    assert response.headers["X-Request-ID"] == request_id
    line = next(log for log in read_log_lines(capsys) if log["severity"] == "ERROR")
    assert line["request_id"] == request_id
    assert "RuntimeError" in line["exception"]


class BrokenProvider(MockProvider):
    """Fake provider whose `generate` always fails with a given error.

    Why: This proves the HTTP mapping without any network code.

    Used by: `build_app_with_llm_route`.
    """

    def __init__(self, error: Exception) -> None:
        """Store the error to raise and start with the provider open.

        Why: Each test needs a different failure. `closed` lets a test check shutdown.

        Used by: the LLM error tests in this module.

        Args:
            error: The exception `generate` raises.
        """
        self.error = error
        self.closed = False

    async def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Raise the stored error.

        Why: This is the failure under test.

        Used by: the `/test/llm` route.

        Args:
            messages: Ignored.

        Raises:
            Exception: The stored error, always.
        """
        raise self.error

    async def aclose(self) -> None:
        """Record that the app closed the provider.

        Why: Lets a test check the shutdown step.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
        self.closed = True


def build_app_with_llm_route(provider) -> FastAPI:
    """Build an app with a test route that calls the given provider.

    Why: No real endpoint calls the LLM until Milestone 3.

    Used by: the LLM error tests in this module.

    Args:
        provider: The provider to inject into `create_app`.

    Returns:
        The app, with `GET /test/llm` added.
    """
    app = create_app(NO_DOCS_SETTINGS, llm_provider=provider)

    @app.get("/test/llm")
    async def call_llm(request: Request) -> dict[str, str]:
        """Call the provider stored on the app."""
        result = await request.app.state.llm_provider.generate(
            [ChatMessage(role="user", content="Hello")]
        )
        return {"text": result.text}

    return app


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMProviderError("HTTP 500 from /chat/completions"), 502, "upstream_error"),
        (LLMTimeoutError("Timeout calling /chat/completions"), 504, "upstream_timeout"),
    ],
)
def test_llm_failures_map_to_gateway_statuses(error, status, code):
    """Check the HTTP status for each kind of provider failure.

    Why: 502 and 504 tell the client the problem is upstream, not in their request. The
    message is fixed, because the internal one can name endpoints.

    Used by: pytest.

    Args:
        error: The failure the provider raises.
        status: The expected HTTP status.
        code: The expected error code.
    """
    with TestClient(build_app_with_llm_route(BrokenProvider(error))) as client:
        response = client.get("/test/llm")

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "/chat/completions" not in response.text


def test_llm_failure_log_never_contains_the_api_key(capsys):
    """Check the logs of a failing real provider, from HTTP call to log line.

    Why: This is the full path a key could leak through: the provider echoes the key in its
    401 body, and the failure is then logged and returned to the client.

    Used by: pytest.
    """
    secret = "sk-test-secret-key"

    def reject(request: httpx2.Request) -> httpx2.Response:
        """Reject the call and echo the key, as some real APIs do."""
        return httpx2.Response(401, json={"error": f"Incorrect API key: {secret}"})

    provider = OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        api_key=secret,
        chat_model="chat-model",
        embedding_model="embed-model",
        timeout_seconds=5,
        transport=httpx2.MockTransport(reject),
    )

    with TestClient(build_app_with_llm_route(provider)) as client:
        response = client.get("/test/llm")

    assert response.status_code == 502
    assert secret not in response.text
    logs = capsys.readouterr().err
    assert "HTTP 401" in logs
    assert secret not in logs


def test_default_app_uses_the_configured_mock_provider():
    """Check the whole app works with the mock provider and no network.

    Why: Local runs, tests, and the demo depend on this.

    Used by: pytest.
    """
    app = create_app(NO_DOCS_SETTINGS)

    @app.get("/test/llm")
    async def call_llm(request: Request) -> dict[str, str]:
        """Call the provider that `create_app` built from the settings."""
        result = await request.app.state.llm_provider.generate(
            [ChatMessage(role="user", content="Hello")]
        )
        return {"text": result.text}

    with TestClient(app) as client:
        assert client.get("/test/llm").json() == {"text": "[mock] Hello"}


def test_provider_is_closed_when_the_app_stops():
    """Check the shutdown step.

    Why: A real provider holds open connections that must be released.

    Used by: pytest.
    """
    provider = BrokenProvider(RuntimeError("unused"))

    with TestClient(build_app_with_llm_route(provider)):
        assert provider.closed is False

    assert provider.closed is True


def test_health_reports_ok(client):
    """Check the liveness response that clients and the demo rely on.

    Why: `GET /health` is the first thing anyone calls to see the API is up.

    Used by: pytest.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 2, "provider": "mock"}


def test_ask_returns_answer_with_sources_and_metrics(client):
    """Check the full `POST /ask` response for an answerable question.

    Why: This is the contract that clients and the demo rely on. Documents are loaded from
    files at startup, so this also covers ingestion end to end.

    Used by: pytest.
    """
    response = client.post("/ask", json={"question": REFUND_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["answer"] == f"[mock] {REFUND_QUESTION}"
    top_source = body["sources"][0]
    assert top_source["id"] == "refunds.md#0"
    assert top_source["document_id"] == "refunds.md"
    assert top_source["position"] == 0
    assert "within 30 days" in top_source["text"]
    assert 0 < top_source["score"] <= 1
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0
    assert body["latency_ms"] >= 0


def test_ask_without_relevant_context_says_not_found(client):
    """Check that an unanswerable question is a normal answer, not an error.

    Why: "Not found" is a correct result, so it is a 200 with `found` set to false.

    Used by: pytest.
    """
    response = client.post("/ask", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []
    assert body["answer"] == "I could not find this in the knowledge base."


@pytest.mark.parametrize("question", ["", "    ", "x" * 1001])
def test_ask_rejects_invalid_questions(client, question):
    """Check the question limits at the API boundary.

    Why: A blank question has nothing to search for, and a long one costs tokens.

    Used by: pytest.

    Args:
        client: The test client.
        question: The invalid question.
    """
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "body.question"


def test_oversized_body_is_rejected_with_413(client):
    """Check the request size limit, with a declared length.

    Why: Validation only runs after the whole body is read, so a size limit must stop a huge
    request before it uses up memory.

    Used by: pytest.
    """
    response = client.post("/ask", content=b"x" * 70_000)

    assert response.status_code == 413
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


def test_oversized_chunked_body_is_rejected_with_413(client):
    """Check the request size limit, without a declared length.

    Why: A chunked request has no `Content-Length`, so the limit must count the bytes received
    instead of trusting a header.

    Used by: pytest.
    """

    def body_parts():
        """Yield a body larger than the limit, in pieces."""
        for _ in range(8):
            yield b"x" * 10_000

    response = client.post("/ask", content=body_parts())

    assert response.status_code == 413


@pytest.mark.parametrize(
    ("error", "status"),
    [(LLMProviderError("HTTP 500"), 502), (LLMTimeoutError("Timeout"), 504)],
)
def test_ask_maps_model_failures_to_gateway_statuses(error, status):
    """Check provider failures through the real endpoint.

    Why: Earlier tests used a test route, because no real endpoint called the LLM yet.

    Used by: pytest.

    Args:
        error: The failure the provider raises.
        status: The expected HTTP status.
    """
    app = create_app(DOCS_SETTINGS, llm_provider=BrokenProvider(error))

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": REFUND_QUESTION})

    assert response.status_code == status


def test_openapi_documents_our_error_format_for_ask(client):
    """Check that the generated docs describe the real error responses.

    Why: FastAPI documents its own validation error format by default, which is not what this
    API returns.

    Used by: pytest.
    """
    responses = client.get("/openapi.json").json()["paths"]["/ask"]["post"]["responses"]

    for status in ("422", "502", "504"):
        schema = responses[status]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/ErrorResponse")

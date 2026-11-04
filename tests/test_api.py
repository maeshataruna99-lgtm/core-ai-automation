"""Behaviour tests for the HTTP API.

Why: These tests send real HTTP requests through the full middleware stack, so they
keep passing when internals are refactored and only fail when API behaviour changes.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from ai_ops.api.app import create_app
from ai_ops.config import Settings


@pytest.fixture
def client():
    """Provide a client for an app built with test settings.

    Why: `create_app` takes settings as an argument, so tests do not touch environment
    variables. The `with` block also runs app startup and shutdown.

    Used by: tests in this module that need only the real routes.
    """
    with TestClient(create_app(Settings(llm_provider="mock"))) as test_client:
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
    app = create_app(Settings(llm_provider="mock"))

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
    with TestClient(create_app(Settings(llm_provider="mock"))) as client:
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


def test_health_reports_ok(client):
    """Check the liveness response that clients and the demo rely on.

    Why: `GET /health` is the first thing anyone calls to see the API is up.

    Used by: pytest.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

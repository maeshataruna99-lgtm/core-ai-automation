"""HTTP tests for the chat, workflow, and approval endpoints.

Why: These run the full path a client uses, with the same mock provider the demo uses, including
the approval that a person gives over HTTP.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_ops.api.app import create_app
from ai_ops.config import Settings
from ai_ops.integrations.outbound import OutboundError, StubOutboundService

SETTINGS = Settings(
    llm_provider="mock", knowledge_dir=Path(__file__).parent / "fixtures" / "knowledge"
)


@pytest.fixture
def client():
    """Provide a client for an app with fixture documents and the sample orders.

    Why: Each test gets a fresh app, so one test's data changes cannot affect another.

    Used by: the tests in this module.
    """
    with TestClient(create_app(SETTINGS)) as test_client:
        yield test_client


def chat(client, message: str) -> dict:
    """Send a chat message and return the response body.

    Why: Keeps each test to the part that differs.

    Used by: the tests in this module.

    Args:
        client: The test client.
        message: The message.

    Returns:
        The parsed response.
    """
    response = client.post("/chat", json={"message": message})
    assert response.status_code == 200
    return response.json()


def test_chat_reads_a_record_through_a_tool(client):
    """Check a read-only question.

    Why: The reply is based on real data, and the response shows which tool produced it.

    Used by: pytest.
    """
    body = chat(client, "What is the status of ORD-1001?")

    assert body["stop_reason"] == "completed"
    assert "shipped" in body["reply"]
    assert body["pending_approvals"] == []
    call = body["tool_calls"][0]
    assert call["name"] == "lookup_record"
    assert call["arguments"] == {"record_id": "ORD-1001"}
    assert call["status"] == "ok"
    assert body["usage"]["input_tokens"] > 0
    assert body["latency_ms"] >= 0


def test_chat_uses_the_knowledge_base_tool(client):
    """Check a policy question.

    Why: The agent reaches documents through the same retrieval as `POST /ask`.

    Used by: pytest.
    """
    body = chat(client, "How many days for refunds after purchase?")

    assert body["tool_calls"][0]["name"] == "search_knowledge"
    assert "refunds.md#0" in body["reply"]


def test_chat_write_needs_approval_before_the_record_changes(client):
    """Check the whole write path over HTTP.

    Why: This is the demo sequence, and the core safety promise of the project: chat proposes,
    the data stays the same, a person approves, and only then does the data change.

    Used by: pytest.
    """
    proposal = chat(client, "Please update ORD-1002 to shipped")

    assert proposal["tool_calls"][0]["status"] == "pending_approval"
    approval_id = proposal["pending_approvals"][0]
    assert "processing" in chat(client, "Status of ORD-1002?")["reply"]

    response = client.post(f"/approvals/{approval_id}", json={"decision": "approve"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "executed"
    assert body["kind"] == "update_record"
    assert body["payload"] == {"record_id": "ORD-1002", "status": "shipped"}
    assert body["result"] == {"id": "ORD-1002", "status": "shipped"}
    assert "shipped" in chat(client, "Status of ORD-1002?")["reply"]


def test_rejected_chat_write_leaves_the_record_unchanged(client):
    """Check the rejection path over HTTP.

    Why: The roadmap requires that a rejected action changes no data.

    Used by: pytest.
    """
    approval_id = chat(client, "Please update ORD-1002 to shipped")["pending_approvals"][0]

    response = client.post(f"/approvals/{approval_id}", json={"decision": "reject"})

    assert response.json()["status"] == "rejected"
    assert "processing" in chat(client, "Status of ORD-1002?")["reply"]


def test_approving_an_action_on_a_missing_record_fails_safely(client):
    """Check an approved action that cannot be carried out.

    Why: The failure must be visible to the caller, without exposing internals.

    Used by: pytest.
    """
    approval_id = chat(client, "Please update ORD-9999 to shipped")["pending_approvals"][0]

    response = client.post(f"/approvals/{approval_id}", json={"decision": "approve"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "action_failed"
    assert "RecordNotFoundError" not in response.text


def test_unknown_approval_returns_404(client):
    """Check an id that does not exist.

    Why: A client must be able to tell this apart from a failed action.

    Used by: pytest.
    """
    response = client.post("/approvals/missing", json={"decision": "approve"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "approval_not_found"


def test_reversing_a_decision_returns_409(client):
    """Check rejecting an action that was already approved.

    Why: The action has already run, so it cannot be taken back.

    Used by: pytest.
    """
    approval_id = chat(client, "Please update ORD-1002 to shipped")["pending_approvals"][0]
    client.post(f"/approvals/{approval_id}", json={"decision": "approve"})

    response = client.post(f"/approvals/{approval_id}", json={"decision": "reject"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "approval_conflict"


def test_invalid_decision_returns_422(client):
    """Check a decision that is neither approve nor reject.

    Why: An unclear decision must never count as an approval.

    Used by: pytest.
    """
    approval_id = chat(client, "Please update ORD-1002 to shipped")["pending_approvals"][0]

    response = client.post(f"/approvals/{approval_id}", json={"decision": "yes please"})

    assert response.status_code == 422
    assert "processing" in chat(client, "Status of ORD-1002?")["reply"]


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_chat_rejects_invalid_messages(client, message):
    """Check the message limits at the API boundary.

    Why: A long message costs tokens on every step of the loop.

    Used by: pytest.

    Args:
        client: The test client.
        message: The invalid message.
    """
    assert client.post("/chat", json={"message": message}).status_code == 422


def test_ticket_workflow_end_to_end(client):
    """Check the workflow over HTTP, from ticket to sent reply.

    Why: This is the end-to-end workflow that the specification asks for. Posting the ticket again
    is how a client reads its current state.

    Used by: pytest.
    """
    ticket = {"ticket_id": "T-100", "message": "Where is my delivery for ORD-1001?"}

    response = client.post("/workflows/ticket", json=ticket)

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "T-100"
    assert body["status"] == "pending_approval"
    assert body["category"] == "delivery"
    assert body["record_id"] == "ORD-1001"
    assert body["draft"]
    assert client.app.state.outbound.sent == {}

    decision = client.post(f"/approvals/{body['approval_id']}", json={"decision": "approve"})

    assert decision.status_code == 200
    assert decision.json()["kind"] == "send_ticket_reply"
    assert client.post("/workflows/ticket", json=ticket).json()["status"] == "sent"
    assert list(client.app.state.outbound.sent) == ["T-100"]


def test_ticket_id_reused_with_different_text_returns_409(client):
    """Check the idempotency key rule over HTTP.

    Why: Returning the old result would silently ignore the new ticket.

    Used by: pytest.
    """
    client.post("/workflows/ticket", json={"ticket_id": "T-1", "message": "First ticket"})

    response = client.post("/workflows/ticket", json={"ticket_id": "T-1", "message": "Second"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ticket_conflict"


@pytest.mark.parametrize(
    "ticket",
    [
        {"ticket_id": "bad id with spaces", "message": "Hello"},
        {"ticket_id": "", "message": "Hello"},
        {"ticket_id": "T-1", "message": ""},
        {"ticket_id": "T-1", "message": "x" * 4001},
    ],
)
def test_ticket_input_limits(client, ticket):
    """Check the ticket limits at the API boundary.

    Why: The id ends up in logs and in an idempotency key, so it is limited to safe characters.

    Used by: pytest.

    Args:
        client: The test client.
        ticket: The invalid request body.
    """
    assert client.post("/workflows/ticket", json=ticket).status_code == 422


def test_get_record_returns_a_seeded_order(client):
    """Check the read endpoint.

    Why: The operator needs to see a record without going through the agent.

    Used by: pytest.
    """
    response = client.get("/records/ORD-1001")

    assert response.status_code == 200
    assert response.json() == {
        "id": "ORD-1001",
        "customer": "Sample Customer A",
        "item": "Bolts",
        "status": "shipped",
    }


def test_get_missing_record_returns_404(client):
    """Check an order that does not exist.

    Why: Clients must get the shared error format.

    Used by: pytest.
    """
    response = client.get("/records/ORD-404")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "record_not_found"


@pytest.mark.parametrize("record_id", ["not-an-order", "ORD-1'OR'1'='1", "ORD-"])
def test_get_record_rejects_malformed_ids(client, record_id):
    """Check ids that could not be an order.

    Why: Bad input stops at the API boundary, before it reaches SQL.

    Used by: pytest.

    Args:
        client: The test client.
        record_id: The malformed id.
    """
    assert client.get(f"/records/{record_id}").status_code == 422


def test_approved_change_survives_an_application_restart():
    """Check persistence through the whole application.

    Why: This is the difference SQL makes for the user: a second app on the same database file sees
    the change, and the seed data at startup does not undo it.

    Used by: pytest.
    """
    with TestClient(create_app(SETTINGS)) as first:
        approval_id = chat(first, "Please update ORD-1002 to shipped")["pending_approvals"][0]
        first.post(f"/approvals/{approval_id}", json={"decision": "approve"})

    with TestClient(create_app(SETTINGS)) as second:
        assert second.get("/records/ORD-1002").json()["status"] == "shipped"


def test_outbound_failure_is_surfaced_and_can_be_retried():
    """Check an approved reply when the messaging service is down.

    Why: The person who approved it must see that it did not go out and why, and the ticket must
    show the failure. Approving again retries it.

    Used by: pytest.
    """

    class DownOnce(StubOutboundService):
        """Outbound service that is down for the first attempt."""

        def __init__(self) -> None:
            """Start in the failing state."""
            super().__init__()
            self.down = True

        async def send(self, idempotency_key: str, body: str) -> dict:
            """Fail once, then behave like the stub."""
            if self.down:
                self.down = False
                raise OutboundError("HTTP 503 from the outbound service")
            return await super().send(idempotency_key, body)

    outbound = DownOnce()
    ticket = {"ticket_id": "T-5", "message": "Where is my delivery for ORD-1001?"}

    with TestClient(create_app(SETTINGS, outbound=outbound)) as client:
        approval_id = client.post("/workflows/ticket", json=ticket).json()["approval_id"]

        failed = client.post(f"/approvals/{approval_id}", json={"decision": "approve"})

        assert failed.status_code == 502
        assert failed.json()["error"]["code"] == "action_failed"
        assert "HTTP 503" in failed.json()["error"]["message"]
        assert client.post("/workflows/ticket", json=ticket).json()["status"] == "send_failed"

        retried = client.post(f"/approvals/{approval_id}", json={"decision": "approve"})

        assert retried.status_code == 200
        assert client.post("/workflows/ticket", json=ticket).json()["status"] == "sent"
    assert outbound.send_count == 1


def test_outbound_service_is_closed_when_the_app_stops():
    """Check the shutdown step.

    Why: The HTTP adapter holds open connections that must be released.

    Used by: pytest.
    """

    class Closable(StubOutboundService):
        """Stub that records being closed."""

        closed = False

        async def aclose(self) -> None:
            """Record the call."""
            self.closed = True

    outbound = Closable()

    with TestClient(create_app(SETTINGS, outbound=outbound)):
        assert outbound.closed is False

    assert outbound.closed is True


def test_action_on_a_missing_record_says_why_it_failed(client):
    """Check that a known failure reaches the caller.

    Why: Hiding this reason would leave a person retrying an action that can never work. Only
    errors written to be safe are shown. Other exceptions stay hidden.

    Used by: pytest.
    """
    approval_id = chat(client, "Please update ORD-9999 to shipped")["pending_approvals"][0]

    response = client.post(f"/approvals/{approval_id}", json={"decision": "approve"})

    assert response.status_code == 502
    assert "ORD-9999" in response.json()["error"]["message"]


def test_tool_calls_in_logs_contain_inputs_and_outcome(capsys):
    """Check the audit trail for a request through the real API.

    Why: The log line must carry the request id, so that a tool call can be traced to a request.
    The app is built here because the log handler binds to the `capsys` stream at that point.

    Used by: pytest.

    Args:
        capsys: Pytest's output capture fixture.
    """
    with TestClient(create_app(SETTINGS)) as client:
        response = client.post("/chat", json={"message": "Please update ORD-1002 to shipped"})

    logs = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    line = next(log for log in logs if log["message"] == "tool call")
    assert line["request_id"] == response.headers["X-Request-ID"]
    assert line["tool"] == "update_record"
    assert line["outcome"] == "pending_approval"


def test_openapi_documents_the_new_endpoints_and_errors(client):
    """Check the generated docs.

    Why: The API contract must match what clients receive.

    Used by: pytest.
    """
    paths = client.get("/openapi.json").json()["paths"]

    approval = paths["/approvals/{approval_id}"]["post"]["responses"]
    for status in ("404", "409", "422", "502"):
        schema = approval[status]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/ErrorResponse")
    assert "/chat" in paths
    assert "/workflows/ticket" in paths

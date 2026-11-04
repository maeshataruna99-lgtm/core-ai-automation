"""Tests for the web interface and the API endpoints it needs.

Why: These cover what can be checked without a browser: that the page is served and packaged, that
it works offline, and that document text can never become HTML. Behaviour in a browser was checked
by hand, as listed in `docs/UI_UX_DESIGN.md`.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ai_ops.ui
from ai_ops.api.app import create_app
from ai_ops.config import Settings

# Read from the installed package, so these tests also prove that the files were packaged.
UI_DIRECTORY = Path(ai_ops.ui.__file__).parent
SETTINGS = Settings(
    llm_provider="mock", knowledge_dir=Path(__file__).parent / "fixtures" / "knowledge"
)


@pytest.fixture
def client():
    """Provide a client for an app with fixture documents.

    Why: Each test gets a fresh app, so no approval carries over.

    Used by: the tests in this module.
    """
    with TestClient(create_app(SETTINGS)) as test_client:
        yield test_client


def propose(client, record_id: str) -> str:
    """Create a pending approval through the agent.

    Why: Several tests need approvals to list.

    Used by: the approval list tests.

    Args:
        client: The test client.
        record_id: The order to change.

    Returns:
        The approval id.
    """
    body = client.post("/chat", json={"message": f"Please update {record_id} to shipped"}).json()
    return body["pending_approvals"][0]


def test_root_sends_people_to_the_interface(client):
    """Check the address a person types first.

    Why: Opening `http://127.0.0.1:8000` should show the app, not a 404.

    Used by: pytest.
    """
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


@pytest.mark.parametrize(
    ("path", "content_type"),
    [("/ui/", "text/html"), ("/ui/app.js", "javascript"), ("/ui/style.css", "text/css")],
)
def test_interface_files_are_served(client, path, content_type):
    """Check each file of the interface.

    Why: A missing file would leave a blank or unstyled page.

    Used by: pytest.

    Args:
        client: The test client.
        path: The URL path.
        content_type: Part of the expected content type.
    """
    response = client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_unknown_api_route_still_uses_the_error_format(client):
    """Check that serving the interface did not take over other routes.

    Why: Clients of the API rely on the shared error format.

    Used by: pytest.
    """
    assert client.get("/does-not-exist").json()["error"]["code"] == "not_found"


def test_script_never_builds_html_from_strings():
    """Check for the browser features that turn text into markup.

    Why: Answers and documents are untrusted, so a document could otherwise run a script in the
    operator's browser and approve its own actions. Text is only ever set with `textContent`.

    Used by: pytest.
    """
    script = (UI_DIRECTORY / "app.js").read_text(encoding="utf-8")

    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in script


def test_page_loads_nothing_from_other_sites():
    """Check for outside addresses, and for the policy that enforces it.

    Why: The interface must work offline, and a third-party script could read everything on the
    page. The Content-Security-Policy makes the browser enforce this as well.

    Used by: pytest.
    """
    page = (UI_DIRECTORY / "index.html").read_text(encoding="utf-8")
    styles = (UI_DIRECTORY / "style.css").read_text(encoding="utf-8")
    script = (UI_DIRECTORY / "app.js").read_text(encoding="utf-8")

    assert "default-src 'self'" in page
    for text in (page, styles, script):
        assert not re.search(r"https?://", text)


def test_health_names_the_provider(client):
    """Check the field that the header of the interface shows.

    Why: A person must be able to see that answers come from a mock.

    Used by: pytest.
    """
    assert client.get("/health").json()["provider"] == "mock"


def test_approvals_are_listed_newest_first(client):
    """Check the list, with its order and fields.

    Why: Without a list, an approval could only be found through the response that created it.

    Used by: pytest.
    """
    first = propose(client, "ORD-1001")
    second = propose(client, "ORD-1002")

    response = client.get("/approvals")

    assert response.status_code == 200
    approvals = response.json()
    assert [approval["id"] for approval in approvals] == [second, first]
    newest = approvals[0]
    assert newest["kind"] == "update_record"
    assert newest["payload"] == {"record_id": "ORD-1002", "status": "shipped"}
    assert newest["status"] == "pending"
    assert newest["created_at"]


def test_approval_list_shows_current_status(client):
    """Check the list after a decision.

    Why: The interface shows only what the API says, so the list must reflect each decision.

    Used by: pytest.
    """
    approval_id = propose(client, "ORD-1002")
    client.post(f"/approvals/{approval_id}", json={"decision": "reject"})

    assert client.get("/approvals").json()[0]["status"] == "rejected"


def test_listing_approvals_on_a_new_app_is_empty(client):
    """Check the empty state.

    Why: The interface has a message for this, and it must not be an error.

    Used by: pytest.
    """
    assert client.get("/approvals").json() == []

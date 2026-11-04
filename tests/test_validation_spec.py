"""Specification of what input is valid, as tables of accepted and rejected examples.

Why: Boundaries are where validation usually breaks, so each limit is tested on both sides. A table
also reads as a specification: a reader can see the exact rules without reading the code. Rejected
values must never appear in an error, because an input can hold personal data or a key.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_ops.api.app import create_app
from ai_ops.config import ConfigError, Settings, read_env_file

FIXTURE_KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge"
CANARY = "private-value-7731"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Provide one app for all the input tables in this module.

    Why: These tests only send invalid or read-only requests, so they can share an app. That keeps a
    table of many cases fast.

    Used by: the API tests in this module.

    Args:
        tmp_path_factory: Pytest's temporary directory factory.
    """
    settings = Settings(
        llm_provider="mock",
        knowledge_dir=FIXTURE_KNOWLEDGE,
        database_path=tmp_path_factory.mktemp("db") / "spec.db",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


ASK_ACCEPTED = [
    pytest.param({"question": "x"}, id="shortest"),
    pytest.param({"question": "x" * 1000}, id="longest"),
    pytest.param({"question": "  padded  "}, id="whitespace-is-trimmed"),
    pytest.param({"question": "x" * 1000 + "   "}, id="trimmed-before-the-limit"),
    pytest.param({"question": "Berapa lama refund? 退款 🙂"}, id="unicode"),
    pytest.param({"question": "'; DROP TABLE records; --"}, id="sql-text-is-only-text"),
    pytest.param({"question": "<script>alert(1)</script>"}, id="markup-is-only-text"),
    pytest.param({"question": "Hello", "unknown": 1}, id="unknown-field-is-ignored"),
]
ASK_REJECTED = [
    pytest.param({}, id="missing"),
    pytest.param({"question": ""}, id="empty"),
    pytest.param({"question": " \t\n "}, id="only-whitespace"),
    pytest.param({"question": "x" * 1001}, id="one-over-the-limit"),
    pytest.param({"question": None}, id="null"),
    pytest.param({"question": 123}, id="number"),
    pytest.param({"question": True}, id="boolean"),
    pytest.param({"question": ["a"]}, id="list"),
    pytest.param({"question": {"a": 1}}, id="object"),
    pytest.param(["question"], id="body-is-a-list"),
    pytest.param("question", id="body-is-a-string"),
]


@pytest.mark.parametrize("body", ASK_ACCEPTED)
def test_ask_accepts(client, body):
    """Check valid questions, including both ends of each limit.

    Why: Rejecting valid input is also a bug.

    Used by: pytest.

    Args:
        client: The test client.
        body: The request body.
    """
    assert client.post("/ask", json=body).status_code == 200


@pytest.mark.parametrize("body", ASK_REJECTED)
def test_ask_rejects(client, body):
    """Check invalid questions, with the shared error format.

    Why: Bad input must stop at the API boundary, before it costs a model call.

    Used by: pytest.

    Args:
        client: The test client.
        body: The request body.
    """
    response = client.post("/ask", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    ("message", "status"),
    [
        pytest.param("x", 200, id="shortest"),
        pytest.param("x" * 2000, 200, id="longest"),
        pytest.param("x" * 2001, 422, id="one-over-the-limit"),
        pytest.param("   ", 422, id="only-whitespace"),
    ],
)
def test_chat_message_limits(client, message, status):
    """Check both sides of the chat message limit.

    Why: The message is sent on every step of the agent's loop, so its limit controls cost.

    Used by: pytest.

    Args:
        client: The test client.
        message: The message.
        status: The expected status.
    """
    assert client.post("/chat", json={"message": message}).status_code == status


TICKET_ID_ACCEPTED = ["a", "T-1", "t_1", "A" * 64, "0"]
TICKET_ID_REJECTED = [
    pytest.param("", id="empty"),
    pytest.param("A" * 65, id="one-over-the-limit"),
    pytest.param("a b", id="space"),
    pytest.param("a/b", id="slash"),
    pytest.param("../etc", id="path-traversal"),
    pytest.param("T-1\nfake log line", id="newline-could-forge-a-log-line"),
    pytest.param("T-١", id="non-ascii-digit"),
    pytest.param("tïcket", id="non-ascii-letter"),
]


@pytest.mark.parametrize("ticket_id", TICKET_ID_ACCEPTED)
def test_ticket_id_accepts(client, ticket_id):
    """Check valid ticket ids.

    Why: The id must allow what real ticket systems use.

    Used by: pytest.

    Args:
        client: The test client.
        ticket_id: The id.
    """
    body = {"ticket_id": ticket_id, "message": "A ticket"}

    assert client.post("/workflows/ticket", json=body).status_code == 200


@pytest.mark.parametrize("ticket_id", TICKET_ID_REJECTED)
def test_ticket_id_rejects(client, ticket_id):
    """Check unsafe ticket ids.

    Why: The id ends up in logs and in an idempotency key sent to another service.

    Used by: pytest.

    Args:
        client: The test client.
        ticket_id: The id.
    """
    body = {"ticket_id": ticket_id, "message": "A ticket"}

    assert client.post("/workflows/ticket", json=body).status_code == 422


@pytest.mark.parametrize(
    ("record_id", "status"),
    [
        pytest.param("ORD-1001", 200, id="exists"),
        pytest.param("ORD-0", 404, id="valid-but-missing"),
        pytest.param("ord-1001", 422, id="lower-case"),
        pytest.param("ORD-", 422, id="no-number"),
        pytest.param("ORD-1a", 422, id="not-a-number"),
        pytest.param("ORD-١٢٣", 422, id="non-ascii-digits"),
        pytest.param("ORD-1%0A", 422, id="newline"),
    ],
)
def test_record_id_rules(client, record_id, status):
    """Check the order id format.

    Why: Only exact ids reach SQL. Python and Rust treat other scripts' digits as digits unless told
    otherwise, so that case is pinned here.

    Used by: pytest.

    Args:
        client: The test client.
        record_id: The id, as it appears in the URL.
        status: The expected status.
    """
    assert client.get(f"/records/{record_id}").status_code == status


@pytest.mark.parametrize(
    "decision",
    ["APPROVE", "approved", "yes", "true", "", " approve", None, True, 1],
)
def test_approval_decision_must_be_exact(client, decision):
    """Check that only the exact words count as a decision.

    Why: An approval runs a real action, so anything unclear must be refused. The check runs before
    the approval lookup, so this holds for any id.

    Used by: pytest.

    Args:
        client: The test client.
        decision: The invalid decision.
    """
    assert client.post("/approvals/any", json={"decision": decision}).status_code == 422


def test_malformed_json_is_a_validation_error(client):
    """Check a body that is not JSON.

    Why: A parser error must not become a 500.

    Used by: pytest.
    """
    response = client.post(
        "/ask", content=b'{"question": ', headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_validation_errors_never_repeat_the_input(client):
    """Check that a rejected value stays out of the response.

    Why: FastAPI's default error repeats the input, and it could hold personal data or a key.

    Used by: pytest.
    """
    response = client.post("/ask", json={"question": [CANARY]})

    assert response.status_code == 422
    assert CANARY not in response.text


REAL = {
    "AI_OPS_LLM_PROVIDER": "openai_compatible",
    "AI_OPS_LLM_BASE_URL": "https://llm.example.test/v1",
    "AI_OPS_LLM_API_KEY": "test-key-value",
    "AI_OPS_LLM_CHAT_MODEL": "chat",
    "AI_OPS_LLM_EMBEDDING_MODEL": "embed",
}
MOCK = {"AI_OPS_LLM_PROVIDER": "mock"}

SETTINGS_ACCEPTED = [
    pytest.param(MOCK, id="mock-needs-nothing-else"),
    pytest.param(REAL, id="real-provider-complete"),
    pytest.param({**MOCK, "AI_OPS_PORT": "1"}, id="lowest-port"),
    pytest.param({**MOCK, "AI_OPS_PORT": "65535"}, id="highest-port"),
    pytest.param({**MOCK, "AI_OPS_LLM_MAX_RETRIES": "0"}, id="no-retries"),
    pytest.param({**MOCK, "AI_OPS_LLM_MAX_RETRIES": "5"}, id="most-retries"),
    pytest.param({**MOCK, "AI_OPS_AGENT_MAX_STEPS": "1"}, id="fewest-steps"),
    pytest.param({**MOCK, "AI_OPS_AGENT_MAX_STEPS": "20"}, id="most-steps"),
    pytest.param({**MOCK, "AI_OPS_RAG_MIN_SCORE": "-1"}, id="lowest-score"),
    pytest.param({**MOCK, "AI_OPS_RAG_MIN_SCORE": "1"}, id="highest-score"),
    pytest.param(
        {**MOCK, "AI_OPS_RAG_CHUNK_SIZE": "2", "AI_OPS_RAG_CHUNK_OVERLAP": "1"}, id="small-chunks"
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "http://127.0.0.1:8080/v1"},
        id="plain-http-to-this-computer",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "http://localhost:8080"}, id="plain-http-to-localhost"
    ),
]
SETTINGS_REJECTED = [
    pytest.param({}, "AI_OPS_LLM_PROVIDER", id="provider-missing"),
    pytest.param({"AI_OPS_LLM_PROVIDER": "gpt"}, "AI_OPS_LLM_PROVIDER", id="provider-unknown"),
    pytest.param({**MOCK, "AI_OPS_PORT": "0"}, "AI_OPS_PORT", id="port-too-low"),
    pytest.param({**MOCK, "AI_OPS_PORT": "65536"}, "AI_OPS_PORT", id="port-too-high"),
    pytest.param({**MOCK, "AI_OPS_PORT": "abc"}, "AI_OPS_PORT", id="port-not-a-number"),
    pytest.param({**MOCK, "AI_OPS_HOST": ""}, "AI_OPS_HOST", id="host-empty"),
    pytest.param({**MOCK, "AI_OPS_LOG_LEVEL": "TRACE"}, "AI_OPS_LOG_LEVEL", id="log-level-unknown"),
    pytest.param(
        {**MOCK, "AI_OPS_LLM_MAX_RETRIES": "-1"}, "AI_OPS_LLM_MAX_RETRIES", id="retries-negative"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_LLM_MAX_RETRIES": "6"}, "AI_OPS_LLM_MAX_RETRIES", id="retries-too-many"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_LLM_TIMEOUT_SECONDS": "0"}, "AI_OPS_LLM_TIMEOUT_SECONDS", id="timeout-zero"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_LLM_TIMEOUT_SECONDS": "inf"},
        "AI_OPS_LLM_TIMEOUT_SECONDS",
        id="timeout-infinite",
    ),
    pytest.param(
        {**MOCK, "AI_OPS_LLM_TIMEOUT_SECONDS": "nan"},
        "AI_OPS_LLM_TIMEOUT_SECONDS",
        id="timeout-not-a-number",
    ),
    pytest.param(
        {**MOCK, "AI_OPS_AGENT_TIMEOUT_SECONDS": "inf"},
        "AI_OPS_AGENT_TIMEOUT_SECONDS",
        id="agent-timeout-infinite",
    ),
    pytest.param(
        {**MOCK, "AI_OPS_OUTBOUND_TIMEOUT_SECONDS": "inf"},
        "AI_OPS_OUTBOUND_TIMEOUT_SECONDS",
        id="outbound-timeout-infinite",
    ),
    pytest.param(
        {**MOCK, "AI_OPS_AGENT_MAX_STEPS": "0"}, "AI_OPS_AGENT_MAX_STEPS", id="steps-zero"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_AGENT_MAX_STEPS": "21"}, "AI_OPS_AGENT_MAX_STEPS", id="steps-too-many"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_RAG_MIN_SCORE": "1.01"}, "AI_OPS_RAG_MIN_SCORE", id="score-too-high"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_RAG_MIN_SCORE": "-1.01"}, "AI_OPS_RAG_MIN_SCORE", id="score-too-low"
    ),
    pytest.param({**MOCK, "AI_OPS_RAG_TOP_K": "0"}, "AI_OPS_RAG_TOP_K", id="top-k-zero"),
    pytest.param(
        {**MOCK, "AI_OPS_RAG_CHUNK_SIZE": "0"}, "AI_OPS_RAG_CHUNK_SIZE", id="chunk-size-zero"
    ),
    pytest.param(
        {**MOCK, "AI_OPS_RAG_CHUNK_SIZE": "100", "AI_OPS_RAG_CHUNK_OVERLAP": "100"},
        "AI_OPS_RAG_CHUNK_OVERLAP",
        id="overlap-equals-size",
    ),
    pytest.param(
        {**MOCK, "AI_OPS_OUTBOUND_PROVIDER": "email"},
        "AI_OPS_OUTBOUND_PROVIDER",
        id="outbound-unknown",
    ),
    pytest.param(
        {k: v for k, v in REAL.items() if k != "AI_OPS_LLM_API_KEY"},
        "AI_OPS_LLM_API_KEY",
        id="real-without-key",
    ),
    pytest.param({**REAL, "AI_OPS_LLM_API_KEY": ""}, "AI_OPS_LLM_API_KEY", id="real-empty-key"),
    pytest.param(
        {**REAL, "AI_OPS_LLM_API_KEY": "replace-me"},
        "AI_OPS_LLM_API_KEY",
        id="key-is-a-placeholder",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_API_KEY": "your-api-key"},
        "AI_OPS_LLM_API_KEY",
        id="key-is-another-placeholder",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_API_KEY": "abc def"}, "AI_OPS_LLM_API_KEY", id="key-has-a-space"
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_API_KEY": "abc\nX-Evil: 1"},
        "AI_OPS_LLM_API_KEY",
        id="key-could-inject-a-header",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "http://llm.example.test/v1"},
        "AI_OPS_LLM_BASE_URL",
        id="key-over-plain-http",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "ftp://llm.example.test"},
        "AI_OPS_LLM_BASE_URL",
        id="url-wrong-scheme",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "llm.example.test"},
        "AI_OPS_LLM_BASE_URL",
        id="url-no-scheme",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "https://user:pass@llm.example.test"},
        "AI_OPS_LLM_BASE_URL",
        id="url-holds-a-password",
    ),
    pytest.param(
        {**REAL, "AI_OPS_LLM_BASE_URL": "http://localhost.evil.test"},
        "AI_OPS_LLM_BASE_URL",
        id="url-only-looks-local",
    ),
    pytest.param(
        {
            **MOCK,
            "AI_OPS_OUTBOUND_PROVIDER": "http",
            "AI_OPS_OUTBOUND_API_KEY": "test-key-value",
            "AI_OPS_OUTBOUND_BASE_URL": "http://messages.example.test",
        },
        "AI_OPS_OUTBOUND_BASE_URL",
        id="outbound-key-over-plain-http",
    ),
    pytest.param(
        {"AI_OPS_LLM_PROVIDER": "deepseek"}, "AI_OPS_LLM_API_KEY", id="deepseek-without-key"
    ),
    pytest.param(
        {"AI_OPS_LLM_PROVIDER": "deepseek", "AI_OPS_LLM_API_KEY": "replace-me"},
        "AI_OPS_LLM_API_KEY",
        id="deepseek-placeholder-key",
    ),
]


@pytest.mark.parametrize("env", SETTINGS_ACCEPTED)
def test_settings_accept(env):
    """Check valid settings, including both ends of each range.

    Why: A limit that is off by one would reject a documented value.

    Used by: pytest.

    Args:
        env: The environment variables.
    """
    assert Settings.from_env(env)


@pytest.mark.parametrize(("env", "named_variable"), SETTINGS_REJECTED)
def test_settings_reject_and_name_the_variable(env, named_variable):
    """Check invalid settings, and that the message says which variable to fix.

    Why: The app must fail at startup with a message a person can act on. A setting such as an
    infinite timeout would otherwise only show up as a request that never ends.

    Used by: pytest.

    Args:
        env: The environment variables.
        named_variable: The variable the message must name.
    """
    with pytest.raises(ConfigError, match=named_variable):
        Settings.from_env(env)


@pytest.mark.parametrize(
    "secret_variable",
    ["AI_OPS_LLM_API_KEY", "AI_OPS_OUTBOUND_API_KEY"],
)
def test_config_errors_never_repeat_a_key(secret_variable):
    """Check that a rejected key stays out of the message.

    Why: This message is printed to a terminal and can end up in CI logs.

    Used by: pytest.

    Args:
        secret_variable: The variable holding the key.
    """
    env = {**REAL, "AI_OPS_OUTBOUND_PROVIDER": "http", "AI_OPS_OUTBOUND_BASE_URL": "https://x.test"}
    env["AI_OPS_OUTBOUND_API_KEY"] = "test-key-value"
    env[secret_variable] = f"{CANARY} with a space"

    with pytest.raises(ConfigError) as error:
        Settings.from_env(env)

    assert CANARY not in str(error.value)
    assert error.value.__cause__ is None


def test_deepseek_needs_only_a_key_and_uses_documented_defaults():
    """Check the DeepSeek preset against its documentation.

    Why: Setting one key should be enough. The values come from the DeepSeek API docs, read on
    2026-11-04. It has no embeddings API, so the embedding model is marked as local.

    Used by: pytest.
    """
    settings = Settings.from_env(
        {"AI_OPS_LLM_PROVIDER": "deepseek", "AI_OPS_LLM_API_KEY": "test-key-value"}
    )

    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_chat_model == "deepseek-flash"
    assert settings.llm_embedding_model == "local-word-hash"


def test_deepseek_model_can_be_changed():
    """Check the override.

    Why: A person may want the larger model, and names change over time.

    Used by: pytest.
    """
    env = {
        "AI_OPS_LLM_PROVIDER": "deepseek",
        "AI_OPS_LLM_API_KEY": "test-key-value",
        "AI_OPS_LLM_CHAT_MODEL": "deepseek-v4-pro",
    }

    assert Settings.from_env(env).llm_chat_model == "deepseek-v4-pro"


ENV_FILES = [
    pytest.param("AI_OPS_PORT=9000\n", {"AI_OPS_PORT": "9000"}, id="plain"),
    pytest.param("AI_OPS_PORT=9000", {"AI_OPS_PORT": "9000"}, id="no-final-newline"),
    pytest.param("AI_OPS_PORT=9000\r\n", {"AI_OPS_PORT": "9000"}, id="windows-line-ending"),
    pytest.param(
        "\ufeffAI_OPS_PORT=9000\n", {"AI_OPS_PORT": "9000"}, id="byte-order-mark-from-notepad"
    ),
    pytest.param(
        "# note\n\n  \nAI_OPS_PORT=9000\n", {"AI_OPS_PORT": "9000"}, id="comments-and-blanks"
    ),
    pytest.param('AI_OPS_PORT="9000"\n', {"AI_OPS_PORT": "9000"}, id="double-quotes"),
    pytest.param("AI_OPS_PORT='9000'\n", {"AI_OPS_PORT": "9000"}, id="single-quotes"),
    pytest.param("  AI_OPS_PORT = 9000  \n", {"AI_OPS_PORT": "9000"}, id="spaces-around"),
    pytest.param("export AI_OPS_PORT=9000\n", {"AI_OPS_PORT": "9000"}, id="export-prefix"),
    pytest.param(
        "AI_OPS_LLM_API_KEY=a#b=c\n",
        {"AI_OPS_LLM_API_KEY": "a#b=c"},
        id="value-keeps-hash-and-equals",
    ),
    pytest.param("AI_OPS_LLM_API_KEY=\n", {}, id="empty-value-counts-as-not-set"),
    pytest.param("OTHER_SECRET=abc\n", {}, id="only-this-project-variables-are-read"),
]


@pytest.mark.parametrize(("content", "expected"), ENV_FILES)
def test_env_file_parsing(tmp_path, content, expected):
    """Check the formats that an env file can have.

    Why: A person edits this file by hand, often in Notepad, which can add a byte order mark and
    Windows line endings.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
        content: The file's content.
        expected: The variables it should give.
    """
    path = tmp_path / ".env"
    path.write_bytes(content.encode("utf-8"))

    assert read_env_file(path) == expected


def test_missing_env_file_gives_nothing(tmp_path):
    """Check a project with no env file.

    Why: This is normal: the mock provider needs none, and CI has none.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    assert read_env_file(tmp_path / ".env") == {}


def test_malformed_env_line_names_the_line_but_not_its_content(tmp_path):
    """Check a line that is not `NAME=value`.

    Why: A person might paste a bare key on its own line. The error must help find it without
    printing it.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    path = tmp_path / ".env"
    path.write_text(f"AI_OPS_PORT=9000\n{CANARY}\n", encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        read_env_file(path)

    assert "line 2" in str(error.value)
    assert CANARY not in str(error.value)


def test_real_environment_overrides_the_env_file(tmp_path, monkeypatch):
    """Check the order of priority.

    Why: This is the usual rule, and it lets CI or a one-off command override the local file.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
        monkeypatch: Pytest's tool for changes that are undone after the test.
    """
    path = tmp_path / "local.env"
    path.write_text("AI_OPS_LLM_PROVIDER=mock\nAI_OPS_PORT=9000\n", encoding="utf-8")
    monkeypatch.setenv("AI_OPS_ENV_FILE", str(path))
    monkeypatch.setenv("AI_OPS_PORT", "9001")

    settings = Settings.from_env()

    assert settings.llm_provider == "mock"
    assert settings.port == 9001


def test_env_file_can_be_switched_off(tmp_path, monkeypatch):
    """Check the switch that the demo script uses.

    Why: The demo promises never to contact a real service, so it must ignore a person's real key.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
        monkeypatch: Pytest's tool for changes that are undone after the test.
    """
    (tmp_path / ".env").write_text("AI_OPS_PORT=9000\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_OPS_ENV_FILE", "")
    monkeypatch.setenv("AI_OPS_LLM_PROVIDER", "mock")

    assert Settings.from_env().port == 8000


def test_named_env_file_must_exist(tmp_path, monkeypatch):
    """Check a file that was asked for by name and is missing.

    Why: Ignoring this would silently run with the wrong settings.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
        monkeypatch: Pytest's tool for changes that are undone after the test.
    """
    monkeypatch.setenv("AI_OPS_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("AI_OPS_LLM_PROVIDER", "mock")

    with pytest.raises(ConfigError, match="AI_OPS_ENV_FILE"):
        Settings.from_env()

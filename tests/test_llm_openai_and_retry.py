"""Behaviour tests for the OpenAI-compatible provider and the retry wrapper.

Why: These cover failure handling that is hard to trigger against a real API. They run
offline: `httpx2.MockTransport` answers requests in memory, so no socket is opened and the
real request and response parsing code still runs.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import asyncio
import json

import httpx2
import pytest

from ai_ops.llm.mock import MockProvider
from ai_ops.llm.openai_compatible import OpenAICompatibleProvider
from ai_ops.llm.provider import (
    ChatMessage,
    EmbeddingResult,
    GenerationResult,
    LLMProviderError,
    LLMTimeoutError,
    ToolCall,
    ToolSchema,
    Usage,
)
from ai_ops.llm.retry import RetryingProvider

API_KEY = "sk-test-secret-key"
MESSAGES = [ChatMessage(role="user", content="Hello")]


def fake_api(request: httpx2.Request) -> httpx2.Response:
    """Answer like an OpenAI-compatible API.

    Why: A real API call would need a key and a network connection.

    Used by: `build_provider` as its default transport handler.

    Args:
        request: The request the provider sent.

    Returns:
        A successful response for the requested endpoint.
    """
    if request.url.path.endswith("/embeddings"):
        return httpx2.Response(
            200,
            json={
                "model": "embed-model",
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )
    return httpx2.Response(
        200,
        json={
            "model": "chat-model",
            "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def build_provider(handler=fake_api) -> OpenAICompatibleProvider:
    """Build a provider that talks to an in-memory handler.

    Why: Every test needs the same settings, with only the fake response changing.

    Used by: the tests in this module.

    Args:
        handler: Function that turns a request into a response.

    Returns:
        A provider using `httpx2.MockTransport`.
    """
    return OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        api_key=API_KEY,
        chat_model="chat-model",
        embedding_model="embed-model",
        timeout_seconds=5,
        transport=httpx2.MockTransport(handler),
    )


async def generate_and_close(provider) -> GenerationResult:
    """Call `generate` and close the provider in one event loop.

    Why: `asyncio.run` starts a new event loop each time, and an HTTP client must be closed
    in the loop that used it.

    Used by: the tests in this module.

    Args:
        provider: Any `LLMProvider`.

    Returns:
        The generation result.
    """
    try:
        return await provider.generate(MESSAGES)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("build", [MockProvider, build_provider], ids=["mock", "openai_compatible"])
def test_providers_are_interchangeable(build):
    """Run the same calls against each provider.

    Why: This is what "swap the provider without changing service code" means in practice:
    a caller gets the same result types from both.

    Used by: pytest.

    Args:
        build: Function that creates the provider.
    """

    async def scenario():
        """Run both calls and close the provider in one event loop."""
        provider = build()
        try:
            return await provider.generate(MESSAGES), await provider.embed(["a", "b"])
        finally:
            await provider.aclose()

    generated, embedded = asyncio.run(scenario())

    assert isinstance(generated, GenerationResult)
    assert generated.text
    assert isinstance(embedded, EmbeddingResult)
    assert len(embedded.vectors) == 2


def test_generate_sends_the_request_a_compatible_api_expects():
    """Check the outgoing request and the parsed response.

    Why: This wire format is the contract with the vendor, so it is pinned with literals.

    Used by: pytest.
    """
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Record the outgoing request, then answer normally."""
        seen.update(
            url=str(request.url),
            auth=request.headers["Authorization"],
            body=json.loads(request.content),
        )
        return fake_api(request)

    result = asyncio.run(generate_and_close(build_provider(handler)))

    assert seen["url"] == "https://llm.example.test/v1/chat/completions"
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert seen["body"] == {
        "model": "chat-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    assert result == GenerationResult(
        text="Hi there", model="chat-model", usage=Usage(input_tokens=3, output_tokens=2)
    )


def test_embeddings_follow_input_order_even_if_the_response_does_not():
    """Check that vectors are sorted by their `index`.

    Why: The API does not promise response order, and a wrong order would silently attach
    vectors to the wrong document chunks.

    Used by: pytest.
    """

    async def scenario():
        """Embed and close the provider in one event loop."""
        provider = build_provider()
        try:
            return await provider.embed(["first", "second"])
        finally:
            await provider.aclose()

    assert asyncio.run(scenario()).vectors == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (503, True), (400, False), (401, False)],
)
def test_error_status_maps_to_provider_error(status, retryable):
    """Check which HTTP failures are worth retrying.

    Why: Rate limits and server errors are temporary. A bad request or bad key will fail the
    same way every time, so retrying only wastes time.

    Used by: pytest.

    Args:
        status: The status the fake API returns.
        retryable: The expected retry decision.
    """
    provider = build_provider(lambda request: httpx2.Response(status, json={"error": "nope"}))

    with pytest.raises(LLMProviderError) as error:
        asyncio.run(generate_and_close(provider))

    assert error.value.retryable is retryable
    assert str(status) in str(error.value)


def test_timeout_raises_timeout_error():
    """Check that a slow provider becomes `LLMTimeoutError`.

    Why: The API layer turns this type into a 504.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Fail the way a slow server does."""
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMTimeoutError):
        asyncio.run(generate_and_close(build_provider(handler)))


def test_connection_failure_is_retryable_provider_error():
    """Check that a network failure becomes a retryable provider error.

    Why: Callers should not need to know the HTTP library's exception types.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Fail the way an unreachable server does."""
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LLMProviderError) as error:
        asyncio.run(generate_and_close(build_provider(handler)))

    assert error.value.retryable is True


def test_malformed_response_raises_provider_error():
    """Check that an unexpected response body is handled.

    Why: A compatible API can differ from the format we expect, and a raw `KeyError` would
    surface as a 500 with no useful meaning.

    Used by: pytest.
    """
    provider = build_provider(lambda request: httpx2.Response(200, json={"unexpected": True}))

    with pytest.raises(LLMProviderError):
        asyncio.run(generate_and_close(provider))


def test_api_key_echoed_by_provider_never_reaches_the_error_message():
    """Check that the provider's response body stays out of our errors.

    Why: Real APIs can echo part of a rejected key in a 401 body. Error messages reach logs,
    so the body must never be copied into them.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Reject the call and echo the key, as some real APIs do."""
        return httpx2.Response(401, json={"error": f"Incorrect API key: {API_KEY}"})

    with pytest.raises(LLMProviderError) as error:
        asyncio.run(generate_and_close(build_provider(handler)))

    assert API_KEY not in str(error.value)
    assert API_KEY not in repr(error.value)


LOOKUP_TOOL = ToolSchema(
    name="lookup_record",
    description="Look up a record.",
    parameters={"type": "object", "properties": {"record_id": {"type": "string"}}},
)


def test_generate_sends_tools_and_parses_tool_calls():
    """Check the tool calling wire format in both directions.

    Why: This format is the contract with the vendor. Arguments arrive as a JSON string, and
    the rest of the code should only see a dict.

    Used by: pytest.
    """
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Record the tools sent, then answer with a tool call."""
        seen.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "chat-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_record",
                                        "arguments": '{"record_id": "ORD-1"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    async def scenario() -> GenerationResult:
        """Call the provider and close it in one event loop."""
        provider = build_provider(handler)
        try:
            return await provider.generate(MESSAGES, tools=[LOOKUP_TOOL])
        finally:
            await provider.aclose()

    result = asyncio.run(scenario())

    assert seen["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_record",
                "description": "Look up a record.",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}},
            },
        }
    ]
    assert result.text == ""
    assert result.tool_calls == (ToolCall("call_1", "lookup_record", {"record_id": "ORD-1"}),)


def test_tool_call_history_uses_the_vendor_format():
    """Check how earlier tool calls and results are sent back to the model.

    Why: The model only sees its earlier tool calls through this history, and an API rejects a
    tool result that does not point at a matching call id.

    Used by: pytest.
    """
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Record the messages, then answer normally."""
        seen.update(json.loads(request.content))
        return fake_api(request)

    call = ToolCall("call_1", "lookup_record", {"record_id": "ORD-1"})
    history = [
        ChatMessage(role="assistant", content="", tool_calls=(call,)),
        ChatMessage(role="tool", content="found", tool_call_id="call_1"),
    ]

    async def scenario() -> None:
        """Call the provider and close it in one event loop."""
        provider = build_provider(handler)
        try:
            await provider.generate(history)
        finally:
            await provider.aclose()

    asyncio.run(scenario())

    assert seen["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_record", "arguments": '{"record_id": "ORD-1"}'},
                }
            ],
        },
        {"role": "tool", "content": "found", "tool_call_id": "call_1"},
    ]


def test_invalid_tool_arguments_json_raises_provider_error():
    """Check a tool call whose arguments are not valid JSON.

    Why: Models can produce broken JSON. A raw `JSONDecodeError` would surface as a 500.

    Used by: pytest.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Answer with arguments that cannot be parsed."""
        function = {"name": "lookup_record", "arguments": "{not json"}
        call = {"id": "call_1", "type": "function", "function": function}
        return httpx2.Response(
            200,
            json={
                "model": "chat-model",
                "choices": [{"message": {"content": None, "tool_calls": [call]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with pytest.raises(LLMProviderError):
        asyncio.run(generate_and_close(build_provider(handler)))


def test_retry_wrapper_passes_tools_through():
    """Check that retries do not lose the tools.

    Why: Without the tools, the agent would silently get answers that never call one.

    Used by: pytest.
    """
    seen = {}

    class Recording(MockProvider):
        """Mock provider that records the tools it receives."""

        async def generate(self, messages, tools=None):
            """Record the tools, then answer like the mock."""
            seen["tools"] = tools
            return await super().generate(messages, tools)

    asyncio.run(RetryingProvider(Recording(), max_retries=1).generate(MESSAGES, [LOOKUP_TOOL]))

    assert seen["tools"] == [LOOKUP_TOOL]


def tool_schema(name: str) -> ToolSchema:
    """Build a tool schema with only a name.

    Why: The mock only looks at which tool names are offered.

    Used by: the mock tool routing tests.

    Args:
        name: The tool name.

    Returns:
        The schema.
    """
    return ToolSchema(name=name, description="", parameters={})


ALL_TOOLS = [tool_schema(n) for n in ("search_knowledge", "lookup_record", "update_record")]


@pytest.mark.parametrize(
    ("message", "expected_name", "expected_arguments"),
    [
        ("What is the status of ORD-1001?", "lookup_record", {"record_id": "ORD-1001"}),
        (
            "Please update ORD-1002 to shipped",
            "update_record",
            {"record_id": "ORD-1002", "status": "shipped"},
        ),
        ("How long do refunds take?", "search_knowledge", {"query": "How long do refunds take?"}),
    ],
)
def test_mock_chooses_a_tool_from_the_message(message, expected_name, expected_arguments):
    """Check the mock's simple keyword routing.

    Why: The offline demo needs the mock to act like a model that calls tools.

    Used by: pytest.

    Args:
        message: The user message.
        expected_name: The tool the mock should call.
        expected_arguments: The arguments it should pass.
    """
    messages = [ChatMessage(role="user", content=message)]

    result = asyncio.run(MockProvider().generate(messages, tools=ALL_TOOLS))

    assert [(c.name, c.arguments) for c in result.tool_calls] == [
        (expected_name, expected_arguments)
    ]


def test_mock_only_calls_tools_it_was_offered():
    """Check a message that matches a tool the caller did not offer.

    Why: A real API would reject an unknown tool, and the mock should not hide such a mistake.

    Used by: pytest.
    """
    messages = [ChatMessage(role="user", content="Status of ORD-1001?")]

    assert asyncio.run(MockProvider().generate(messages)).tool_calls == ()


def test_mock_answers_with_the_tool_result_after_a_tool_call():
    """Check the step after a tool has run.

    Why: Without this, the agent would call the same tool until it reached the step limit.

    Used by: pytest.
    """
    messages = [
        ChatMessage(role="user", content="Status of ORD-1001?"),
        ChatMessage(role="tool", content="Status: shipped", tool_call_id="mock-1"),
    ]

    result = asyncio.run(MockProvider().generate(messages, tools=ALL_TOOLS))

    assert result.tool_calls == ()
    assert result.text == "[mock] Status: shipped"


class FailingProvider:
    """Fake provider that fails a fixed number of times and then succeeds.

    Why: The retry wrapper can be tested without any HTTP code, which proves it works with
    any provider.

    Used by: the retry tests in this module.
    """

    def __init__(self, failures: int, error: Exception) -> None:
        """Store how often to fail and with which error.

        Why: Each test needs a different failure pattern.

        Used by: the retry tests.

        Args:
            failures: Number of calls that fail before one succeeds.
            error: The exception to raise.
        """
        self.failures = failures
        self.error = error
        self.calls = 0

    async def generate(self, messages: list[ChatMessage], tools=None) -> GenerationResult:
        """Fail or succeed according to the call count.

        Why: This is the behaviour under test.

        Used by: `RetryingProvider`.

        Args:
            messages: Ignored.
            tools: Ignored.

        Returns:
            A fixed result once the failures are used up.
        """
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return GenerationResult(text="ok", model="fake", usage=Usage(0, 0))


def run_with_retries(inner: FailingProvider, max_retries: int) -> tuple[GenerationResult, list]:
    """Call `generate` through the retry wrapper and record the waits.

    Why: Passing a fake `sleep` keeps the tests instant and lets them check the backoff.

    Used by: `test_retries_temporary_failures_with_growing_delay`.

    Args:
        inner: The provider to wrap.
        max_retries: Retries allowed after the first attempt.

    Returns:
        The result and the list of delays.
    """
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        """Record the delay instead of waiting."""
        delays.append(seconds)

    provider = RetryingProvider(inner, max_retries=max_retries, sleep=fake_sleep)
    return asyncio.run(provider.generate(MESSAGES)), delays


def test_retries_temporary_failures_with_growing_delay():
    """Check a call that succeeds on the third attempt.

    Why: Growing delays give an overloaded provider time to recover.

    Used by: pytest.
    """
    inner = FailingProvider(2, LLMTimeoutError("timed out"))

    result, delays = run_with_retries(inner, max_retries=2)

    assert result.text == "ok"
    assert inner.calls == 3
    assert delays == [0.5, 1.0]


def test_stops_after_max_retries_and_raises_the_last_error():
    """Check that retries are bounded.

    Why: Without a limit, an outage would keep every request waiting forever.

    Used by: pytest.
    """
    inner = FailingProvider(100, LLMTimeoutError("timed out"))

    with pytest.raises(LLMTimeoutError):
        run_with_retries(inner, max_retries=2)

    assert inner.calls == 3


def test_does_not_retry_permanent_failures():
    """Check that a non-retryable error fails at once.

    Why: Retrying a rejected API key could also get the key rate limited or locked.

    Used by: pytest.
    """
    inner = FailingProvider(100, LLMProviderError("HTTP 401", retryable=False))

    with pytest.raises(LLMProviderError):
        run_with_retries(inner, max_retries=2)

    assert inner.calls == 1

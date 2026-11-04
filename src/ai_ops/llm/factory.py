"""Build the LLM provider that the settings ask for.

Why: This is the only place that names concrete providers, so adding one changes this file
and the settings, and no service code.

Used by: `ai_ops.api.app.create_app`.
"""

from collections.abc import Callable

import httpx2

from ai_ops.config import Settings
from ai_ops.llm.local_embeddings import LocalEmbeddingProvider
from ai_ops.llm.mock import MockProvider
from ai_ops.llm.openai_compatible import OpenAICompatibleProvider
from ai_ops.llm.provider import LLMProvider
from ai_ops.llm.retry import RetryingProvider

# DeepSeek enables thinking mode by default, and then its documentation says that the reasoning of
# all previous turns must be sent back when using tools. The agent does not keep that text, so this
# switches the mode off, as DeepSeek's own non-thinking tool call example does.
DEEPSEEK_CHAT_PARAMETERS = {"thinking": {"type": "disabled"}}


def build_provider(
    settings: Settings,
    transport: httpx2.AsyncBaseTransport | None = None,
    chat_model: Callable[[], str] | None = None,
) -> LLMProvider:
    """Create the configured provider.

    Why: The real provider is wrapped with retries here, so every caller gets them. The mock
    never fails, so it needs no wrapper.

    Used by: `ai_ops.api.app.create_app` and tests.

    Args:
        settings: Validated settings. `Settings` guarantees that the real provider's values
            are present, which is why they can be used without further checks.
        transport: Replaces the network. Tests pass an in-memory transport.
        chat_model: Gives the current model. Defaults to the configured one.

    Returns:
        A provider ready to use. The caller must call `aclose` on it at shutdown.
    """
    if settings.llm_provider == "mock":
        return MockProvider()
    remote = OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        chat_model=chat_model or settings.llm_chat_model,
        embedding_model=settings.llm_embedding_model,
        timeout_seconds=settings.llm_timeout_seconds,
        transport=transport,
        extra_chat_parameters=(
            DEEPSEEK_CHAT_PARAMETERS if settings.llm_provider == "deepseek" else None
        ),
    )
    provider = LocalEmbeddingProvider(remote) if settings.llm_provider == "deepseek" else remote
    return RetryingProvider(provider, max_retries=settings.llm_max_retries)

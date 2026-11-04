"""Build the LLM provider that the settings ask for.

Why: This is the only place that names concrete providers, so adding one changes this file
and the settings, and no service code.

Used by: `ai_ops.api.app.create_app`.
"""

from ai_ops.config import Settings
from ai_ops.llm.mock import MockProvider
from ai_ops.llm.openai_compatible import OpenAICompatibleProvider
from ai_ops.llm.provider import LLMProvider
from ai_ops.llm.retry import RetryingProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Create the configured provider.

    Why: The real provider is wrapped with retries here, so every caller gets them. The mock
    never fails, so it needs no wrapper.

    Used by: `ai_ops.api.app.create_app` and `tests/test_config.py`.

    Args:
        settings: Validated settings. `Settings` guarantees that the real provider's values
            are present, which is why they can be used without further checks.

    Returns:
        A provider ready to use. The caller must call `aclose` on it at shutdown.
    """
    if settings.llm_provider == "mock":
        return MockProvider()
    return RetryingProvider(
        OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            chat_model=settings.llm_chat_model,
            embedding_model=settings.llm_embedding_model,
            timeout_seconds=settings.llm_timeout_seconds,
        ),
        max_retries=settings.llm_max_retries,
    )

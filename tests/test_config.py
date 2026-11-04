"""Behaviour tests for configuration loading and startup failure.

Why: A wrong configuration must stop the app at startup with a readable message, and the
safe defaults must not change by accident.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import asyncio
import os
import subprocess
import sys

import pytest

from ai_ops.config import ConfigError, Settings
from ai_ops.integrations.outbound import HttpOutboundAdapter, StubOutboundService, build_outbound
from ai_ops.llm.factory import build_provider
from ai_ops.llm.mock import MockProvider
from ai_ops.llm.retry import RetryingProvider


def test_defaults_bind_to_localhost_only():
    """Check the default bind address.

    Why: The API has no authentication, so it must not be reachable from the network
    unless someone asks for that explicitly.

    Used by: pytest.
    """
    settings = Settings.from_env({"AI_OPS_LLM_PROVIDER": "mock"})

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_invalid_value_names_the_variable_without_echoing_the_value():
    """Check the error for a bad value.

    Why: The message must point at the variable to fix. It must not repeat the value,
    because a mistyped setting could be a secret and this message reaches the logs.

    Used by: pytest.
    """
    with pytest.raises(ConfigError) as error:
        Settings.from_env({"AI_OPS_LLM_PROVIDER": "mock", "AI_OPS_PORT": "secret-value"})

    assert "AI_OPS_PORT" in str(error.value)
    assert "secret-value" not in str(error.value)


REAL_PROVIDER_ENV = {
    "AI_OPS_LLM_PROVIDER": "openai_compatible",
    "AI_OPS_LLM_BASE_URL": "https://llm.example.test/v1",
    "AI_OPS_LLM_API_KEY": "sk-test-secret-key",
    "AI_OPS_LLM_CHAT_MODEL": "chat-model",
    "AI_OPS_LLM_EMBEDDING_MODEL": "embed-model",
}


def test_real_provider_reports_every_missing_setting():
    """Check the error when real provider settings are incomplete.

    Why: Naming all the missing variables at once saves fixing them one restart at a time.
    The key that is present must not appear in the message.

    Used by: pytest.
    """
    env = {
        "AI_OPS_LLM_PROVIDER": "openai_compatible",
        "AI_OPS_LLM_API_KEY": "sk-test-secret-key",
    }

    with pytest.raises(ConfigError) as error:
        Settings.from_env(env)

    message = str(error.value)
    assert "AI_OPS_LLM_BASE_URL" in message
    assert "AI_OPS_LLM_CHAT_MODEL" in message
    assert "AI_OPS_LLM_EMBEDDING_MODEL" in message
    assert "AI_OPS_LLM_API_KEY" not in message
    assert "sk-test-secret-key" not in message


def test_overlap_must_be_smaller_than_chunk_size():
    """Check the settings that would hang the chunker.

    Why: This should fail at startup with the variable named, not raise later inside
    ingestion.

    Used by: pytest.
    """
    env = {"AI_OPS_LLM_PROVIDER": "mock", "AI_OPS_RAG_CHUNK_OVERLAP": "900"}

    with pytest.raises(ConfigError, match="AI_OPS_RAG_CHUNK_OVERLAP"):
        Settings.from_env(env)


def test_http_outbound_reports_missing_settings():
    """Check the error when the real adapter is not fully configured.

    Why: Finding this at startup is better than when a person approves a reply.

    Used by: pytest.
    """
    env = {"AI_OPS_LLM_PROVIDER": "mock", "AI_OPS_OUTBOUND_PROVIDER": "http"}

    with pytest.raises(ConfigError) as error:
        Settings.from_env(env)

    assert "AI_OPS_OUTBOUND_BASE_URL" in str(error.value)
    assert "AI_OPS_OUTBOUND_API_KEY" in str(error.value)


def test_outbound_service_is_selected_by_configuration_only():
    """Check that configuration alone decides the outbound service.

    Why: The demo must never contact a real service unless someone asks for that. So the stub is
    the default.

    Used by: pytest.
    """

    async def build_and_close(settings):
        """Build the service and close its HTTP client in one event loop."""
        service = build_outbound(settings)
        await service.aclose()
        return service

    stub = asyncio.run(build_and_close(Settings(llm_provider="mock")))
    http = asyncio.run(
        build_and_close(
            Settings.from_env(
                {
                    "AI_OPS_LLM_PROVIDER": "mock",
                    "AI_OPS_OUTBOUND_PROVIDER": "http",
                    "AI_OPS_OUTBOUND_BASE_URL": "https://messages.example.test",
                    "AI_OPS_OUTBOUND_API_KEY": "outbound-test-secret",
                }
            )
        )
    )

    assert isinstance(stub, StubOutboundService)
    assert isinstance(http, HttpOutboundAdapter)


def test_api_key_is_hidden_when_settings_are_printed():
    """Check that printing settings does not reveal the key.

    Why: Settings objects end up in logs and debug output.

    Used by: pytest.
    """
    settings = Settings.from_env(REAL_PROVIDER_ENV)

    assert "sk-test-secret-key" not in repr(settings)
    assert "sk-test-secret-key" not in str(settings.model_dump())


def test_provider_is_selected_by_configuration_only():
    """Check that configuration alone decides the provider.

    Why: Switching to a real provider must need no code change. The real provider is wrapped
    with retries, and the mock has nothing to retry.

    Used by: pytest.
    """

    async def build_and_close(settings):
        """Build the provider and close its HTTP client in one event loop."""
        provider = build_provider(settings)
        await provider.aclose()
        return provider

    mock = asyncio.run(build_and_close(Settings(llm_provider="mock")))
    real = asyncio.run(build_and_close(Settings.from_env(REAL_PROVIDER_ENV)))

    assert isinstance(mock, MockProvider)
    assert isinstance(real, RetryingProvider)


def test_process_exits_with_clear_message_when_config_is_missing():
    """Check the real command fails fast when a required variable is missing.

    Why: This runs `python -m ai_ops` as a real process, because a person sees the exit
    code and stderr, not a Python exception.

    Used by: pytest.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("AI_OPS_")}

    result = subprocess.run(
        [sys.executable, "-m", "ai_ops"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "AI_OPS_LLM_PROVIDER" in result.stderr
    assert "Traceback" not in result.stderr

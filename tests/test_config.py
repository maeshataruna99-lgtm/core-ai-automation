"""Behaviour tests for configuration loading and startup failure.

Why: A wrong configuration must stop the app at startup with a readable message, and the
safe defaults must not change by accident.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import os
import subprocess
import sys

import pytest

from ai_ops.config import ConfigError, Settings


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

"""Application settings loaded from environment variables.

Why: Configuration and secrets stay out of the code, and every module reads them from
one validated object. Pydantic is used directly because FastAPI already installs it,
so `pydantic-settings` would be an extra dependency for a few lines of code.

Used by: `ai_ops.__main__` loads the settings at startup. `ai_ops.api.app.create_app`
receives them as an argument.
"""

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ENV_PREFIX = "AI_OPS_"


class ConfigError(Exception):
    """Raised when the environment holds missing or invalid configuration.

    Why: Callers can show one clean message instead of a Pydantic traceback.

    Used by: `Settings.from_env` raises it. `ai_ops.__main__.main` catches it.
    """


class Settings(BaseModel):
    """Validated application settings.

    Why: A frozen model fails at startup when a value is wrong, instead of failing later
    in the middle of a request, and no code can change the settings while running.

    Used by: `create_app` and `ai_ops.__main__.main`.

    Attributes:
        llm_provider: LLM provider to use. It has no default on purpose, so nobody starts
            a paid provider by accident. Milestone 2 adds the real provider and uses it.
        log_level: Minimum severity written to the logs.
        host: Bind address. The default accepts local connections only.
        port: Bind port.
    """

    model_config = ConfigDict(frozen=True)

    llm_provider: Literal["mock"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from `AI_OPS_*` environment variables.

        Why: This is the only place that reads the environment, so the variable names and
        their validation stay together.

        Used by: `ai_ops.__main__.main`.

        Args:
            env: Variables to read. Defaults to `os.environ`; tests pass a plain dict.

        Returns:
            The validated settings.

        Raises:
            ConfigError: A required variable is missing or a value is invalid. The message
                names each variable. It omits the values, because later settings can be secrets.
        """
        source = os.environ if env is None else env
        values = {
            name: source[f"{ENV_PREFIX}{name.upper()}"]
            for name in cls.model_fields
            if f"{ENV_PREFIX}{name.upper()}" in source
        }
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{ENV_PREFIX}{str(error['loc'][0]).upper()}: {error['msg']}"
                for error in exc.errors()
            )
            raise ConfigError(problems) from exc

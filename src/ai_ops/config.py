"""Application settings loaded from environment variables.

Why: Configuration and secrets stay out of the code, and every module reads them from
one validated object. Pydantic is used directly because FastAPI already installs it,
so `pydantic-settings` would be an extra dependency for a few lines of code.

Used by: `ai_ops.__main__` loads the settings at startup. `ai_ops.api.app.create_app`
receives them as an argument.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

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
            a paid provider by accident.
        llm_base_url: API root of the real provider.
        llm_api_key: Key for the real provider. `SecretStr` hides it when printed.
        llm_chat_model: Model name for text generation.
        llm_embedding_model: Model name for embeddings.
        llm_timeout_seconds: Time limit for each network step of one attempt.
        llm_max_retries: Retries after the first attempt. Capped, because each retry adds a
            full timeout to the worst-case wait.
        knowledge_dir: Directory of documents loaded at startup. A relative path starts from
            the current working directory.
        rag_chunk_size: Maximum characters per chunk. Large enough to hold a paragraph.
        rag_chunk_overlap: Characters repeated between chunks.
        rag_top_k: Maximum chunks used per question.
        rag_min_score: Lowest similarity counted as relevant. It depends on the embedding
            model, so check it when the model changes.
        agent_max_steps: Maximum model calls for one chat message. Capped, because each step
            costs tokens.
        agent_timeout_seconds: Time limit for one chat message.
        log_level: Minimum severity written to the logs.
        host: Bind address. The default accepts local connections only.
        port: Bind port.
    """

    model_config = ConfigDict(frozen=True)

    llm_provider: Literal["mock", "openai_compatible"]
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_chat_model: str | None = None
    llm_embedding_model: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    knowledge_dir: Path = Path("data/knowledge")
    rag_chunk_size: int = Field(default=800, gt=0)
    rag_chunk_overlap: int = Field(default=100, ge=0)
    rag_top_k: int = Field(default=4, gt=0)
    rag_min_score: float = Field(default=0.15, ge=-1, le=1)
    agent_max_steps: int = Field(default=5, ge=1, le=20)
    agent_timeout_seconds: float = Field(default=60.0, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @model_validator(mode="after")
    def _require_real_provider_settings(self) -> "Settings":
        """Check the rules that involve more than one setting.

        Why: The real provider's fields are optional for the mock, and overlap depends on the
        chunk size, so per-field rules cannot express these. Checking here makes a bad value
        fail at startup, not on the first request.

        Used by: Pydantic, each time a `Settings` object is created.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: A rule is broken. The message names the variables involved.
        """
        if self.llm_provider == "openai_compatible":
            required = ("llm_base_url", "llm_api_key", "llm_chat_model", "llm_embedding_model")
            missing = [
                f"{ENV_PREFIX}{name.upper()}" for name in required if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"openai_compatible requires {', '.join(missing)}")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"{ENV_PREFIX}RAG_CHUNK_OVERLAP must be smaller than {ENV_PREFIX}RAG_CHUNK_SIZE"
            )
        return self

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
            # A whole-model error has no field location, and its message names the variables.
            problems = "; ".join(
                f"{ENV_PREFIX}{str(error['loc'][0]).upper()}: {error['msg']}"
                if error["loc"]
                else error["msg"]
                for error in exc.errors()
            )
            # `from None` because the Pydantic error text includes the rejected input values.
            raise ConfigError(problems) from None

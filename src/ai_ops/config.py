"""Application settings loaded from environment variables and a local `.env` file.

Why: Configuration and secrets stay out of the code, and every module reads them from
one validated object. Pydantic is used directly because FastAPI already installs it,
so `pydantic-settings` would be an extra dependency for a few lines of code.

Used by: `ai_ops.__main__` loads the settings at startup. `ai_ops.api.app.create_app`
receives them as an argument.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

ENV_PREFIX = "AI_OPS_"
ENV_FILE_VARIABLE = f"{ENV_PREFIX}ENV_FILE"
DEFAULT_ENV_FILE = ".env"
# From the DeepSeek API docs, read on 2026-11-04. They list no embeddings endpoint.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = ("deepseek-flash", "deepseek-v4-pro")
DEEPSEEK_DEFAULT_MODEL = DEEPSEEK_MODELS[0]
LOCAL_EMBEDDING_MODEL = "local-word-hash"
# Plain HTTP would send a key unencrypted, so it is only allowed to this computer.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Values that a person may leave in by accident, so that the app does not send them to a provider.
PLACEHOLDER_KEYS = {"replace-me", "your-api-key", "your_api_key", "changeme", "change-me", "xxx"}


class ConfigError(Exception):
    """Raised when the environment holds missing or invalid configuration.

    Why: Callers can show one clean message instead of a Pydantic traceback. Its message never holds
    a value, because a value may be a key.

    Used by: `Settings.from_env` and `read_env_file` raise it. The command entry points catch it.
    """


def read_env_file(path: Path) -> dict[str, str]:
    """Read `NAME=value` lines from a local settings file.

    Why: A person can keep a key in one ignored file, which avoids typing it into a shell, where it
    would stay in the command history. This is a few lines of the standard library, so it needs no
    dependency. `utf-8-sig` handles the byte order mark that some Windows editors add. Only this
    project's variables are read, so other secrets in the file are never picked up.

    Used by: `Settings.from_env` and tests.

    Args:
        path: The file. A missing file gives an empty result.

    Returns:
        The variables that have a value. An empty value counts as not set.

    Raises:
        ConfigError: The file cannot be read, or a line is not `NAME=value`. The message gives the
            line number, not the content, because the line may be a pasted key.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError):
        raise ConfigError(f"Cannot read the settings file {path.name}") from None

    values: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.removeprefix("export ").partition("=")
        if not separator:
            raise ConfigError(f"Line {number} of {path.name} is not NAME=value (line {number})")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name.startswith(ENV_PREFIX) and value:
            values[name] = value
    return values


def _environment() -> dict[str, str]:
    """Combine the local settings file with the real environment.

    Why: Real environment variables win, which is the usual rule, and lets CI or a single command
    override the file. An empty `AI_OPS_ENV_FILE` switches the file off, which the demo script uses
    so that it can never pick up a real key.

    Used by: `Settings.from_env`.

    Returns:
        The combined variables.

    Raises:
        ConfigError: The named file does not exist, or it cannot be read.
    """
    named_file = os.environ.get(ENV_FILE_VARIABLE)
    if named_file == "":
        return dict(os.environ)
    path = Path(named_file or DEFAULT_ENV_FILE)
    if named_file and not path.is_file():
        raise ConfigError(f"{ENV_FILE_VARIABLE} names a file that does not exist")
    return {**read_env_file(path), **os.environ}


def _check_url(variable: str, url: str) -> None:
    """Check a URL that a key will be sent to.

    Why: A typo must not send a key in plain text or to the wrong place. Comparing the exact host
    means that `http://localhost.evil.test` does not count as local. The URL is left out of the
    error, because one with a password in it is itself a secret.

    Used by: `Settings._check_rules`.

    Args:
        variable: The variable's name, for the message.
        url: The URL.

    Raises:
        ValueError: The URL is not acceptable.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{variable} must be a full http or https URL")
    if parts.username or parts.password:
        raise ValueError(f"{variable} must not contain a user name or password")
    if parts.scheme == "http" and parts.hostname not in LOCAL_HOSTS:
        raise ValueError(f"{variable} must use https, because a key is sent to it")


def _check_key(variable: str, key: SecretStr) -> None:
    """Check the form of a key.

    Why: A placeholder would fail at the provider with a confusing error. Whitespace usually means a
    bad paste, and a line break could add a forged HTTP header.

    Used by: `Settings._check_rules`.

    Args:
        variable: The variable's name, for the message.
        key: The key.

    Raises:
        ValueError: The key is not acceptable.
    """
    value = key.get_secret_value()
    if value.lower() in PLACEHOLDER_KEYS or value.startswith("<"):
        raise ValueError(f"{variable} still holds a placeholder. Set your real key")
    if re.search(r"\s", value):
        raise ValueError(f"{variable} must not contain spaces or line breaks")


class Settings(BaseModel):
    """Validated application settings.

    Why: A frozen model fails at startup when a value is wrong, instead of failing later
    in the middle of a request, and no code can change the settings while running.

    Used by: `create_app` and `ai_ops.__main__.main`.

    Attributes:
        llm_provider: LLM provider to use. It has no default on purpose, so nobody starts
            a paid provider by accident. `deepseek` needs only a key.
        llm_base_url: API root of the real provider.
        llm_api_key: Key for the real provider. `SecretStr` hides it when printed.
        llm_chat_model: Model name for text generation.
        llm_embedding_model: Model name for embeddings.
        llm_timeout_seconds: Time limit for each network step of one attempt.
        llm_max_retries: Retries after the first attempt. Capped, because each retry adds a
            full timeout to the worst-case wait.
        knowledge_dir: Directory of documents loaded at startup. A relative path starts from
            the current working directory.
        uploads_dir: Folder for documents added while the app runs. Git ignores the default.
        rag_chunk_size: Maximum characters per chunk. Large enough to hold a paragraph.
        rag_chunk_overlap: Characters repeated between chunks.
        rag_top_k: Maximum chunks used per question. See `docs/EVALUATION.md` for how the default
            was chosen.
        rag_min_score: Lowest similarity counted as relevant. The default was measured with the
            mock provider. It depends on the embedding model, so measure it again when the model
            changes.
        agent_max_steps: Maximum model calls for one chat message. Capped, because each step
            costs tokens.
        agent_timeout_seconds: Time limit for one chat message.
        database_path: SQLite file. A relative path starts from the current working directory.
        outbound_provider: Messaging service. The default sends nothing.
        outbound_base_url: API root of the real messaging service.
        outbound_api_key: Key for the real messaging service.
        outbound_timeout_seconds: Time limit for each network step of a send.
        log_level: Minimum severity written to the logs.
        host: Bind address. The default accepts local connections only.
        port: Bind port.
    """

    model_config = ConfigDict(frozen=True)

    llm_provider: Literal["mock", "openai_compatible", "deepseek"]
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_chat_model: str | None = None
    llm_embedding_model: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    knowledge_dir: Path = Path("data/knowledge")
    uploads_dir: Path = Path("data/uploads")
    rag_chunk_size: int = Field(default=800, gt=0)
    rag_chunk_overlap: int = Field(default=100, ge=0)
    rag_top_k: int = Field(default=2, gt=0)
    rag_min_score: float = Field(default=0.10, ge=-1, le=1)
    agent_max_steps: int = Field(default=5, ge=1, le=20)
    agent_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    database_path: Path = Path("data/ai_ops.db")
    outbound_provider: Literal["stub", "http"] = "stub"
    outbound_base_url: str | None = None
    outbound_api_key: SecretStr | None = None
    outbound_timeout_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)

    @model_validator(mode="before")
    @classmethod
    def _apply_deepseek_defaults(cls, values: Any) -> Any:
        """Fill in what DeepSeek needs, so that only a key has to be set.

        Why: The URL and model come from its documentation. Embeddings are always local, because the
        documentation lists no embeddings endpoint.

        Used by: Pydantic, before validating a `Settings` object.

        Args:
            values: The raw input.

        Returns:
            The input with the defaults added.
        """
        if isinstance(values, dict) and values.get("llm_provider") == "deepseek":
            return {
                "llm_base_url": DEEPSEEK_BASE_URL,
                "llm_chat_model": DEEPSEEK_DEFAULT_MODEL,
                **values,
                "llm_embedding_model": LOCAL_EMBEDDING_MODEL,
            }
        return values

    @model_validator(mode="after")
    def _check_rules(self) -> "Settings":
        """Check the rules that involve more than one setting.

        Why: Real services need fields that the mock and the stub do not, and overlap depends on the
        chunk size, so per-field rules cannot express these. Checking here makes a bad value
        fail at startup, not on the first request.

        Used by: Pydantic, each time a `Settings` object is created.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: A rule is broken. The message names the variables involved.
        """
        if self.llm_provider != "mock":
            required = ("llm_base_url", "llm_api_key", "llm_chat_model", "llm_embedding_model")
            missing = [
                f"{ENV_PREFIX}{name.upper()}" for name in required if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"{self.llm_provider} requires {', '.join(missing)}")
        if self.outbound_provider == "http":
            required = ("outbound_base_url", "outbound_api_key")
            missing = [
                f"{ENV_PREFIX}{name.upper()}" for name in required if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"The http outbound provider requires {', '.join(missing)}")
        for name in ("llm_base_url", "outbound_base_url"):
            if getattr(self, name):
                _check_url(f"{ENV_PREFIX}{name.upper()}", getattr(self, name))
        for name in ("llm_api_key", "outbound_api_key"):
            if getattr(self, name):
                _check_key(f"{ENV_PREFIX}{name.upper()}", getattr(self, name))
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"{ENV_PREFIX}RAG_CHUNK_OVERLAP must be smaller than {ENV_PREFIX}RAG_CHUNK_SIZE"
            )
        return self

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from `AI_OPS_*` variables.

        Why: This is the only place that reads the environment, so the variable names and
        their validation stay together.

        Used by: `ai_ops.__main__.main` and `ai_ops.evaluation.__main__.main`.

        Args:
            env: Variables to read. Defaults to the local `.env` file and the real environment;
                tests pass a plain dict.

        Returns:
            The validated settings.

        Raises:
            ConfigError: A required variable is missing or a value is invalid. The message
                names each variable. It omits the values, because later settings can be secrets.
        """
        source = _environment() if env is None else env
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

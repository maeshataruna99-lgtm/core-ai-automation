"""See and change the LLM settings while the app runs, and check the connection.

Why: A person needs to know which model answers, and whether the service really works. "A key is
configured" proves nothing, so the check makes a real request. Only the model can be changed here.
The
key stays in the local `.env`, because this API has no login.

Used by: `ai_ops.api.app` creates it. `ai_ops.api.routes` calls it.
"""

from dataclasses import dataclass

from ai_ops.config import DEEPSEEK_MODELS, Settings
from ai_ops.integrations.app_settings import AppSettingsStore
from ai_ops.llm.provider import (
    ChatMessage,
    LLMError,
    LLMProvider,
    ModelSelection,
    Usage,
    describe_http_failure,
)

MODEL_SETTING = "llm_chat_model"
TEST_MESSAGE = "Reply with the single word OK."


class ModelNotSelectableError(Exception):
    """This provider's model cannot be changed here.

    Why: The API maps this to a 409, so the page can explain it.

    Used by: raised by `LlmSettingsService.select_model`, mapped in `ai_ops.api.errors`.
    """


class UnknownModelError(Exception):
    """The requested model is not on the list.

    Why: A wrong name would otherwise only fail on someone's next question.

    Used by: raised by `LlmSettingsService.select_model`, mapped in `ai_ops.api.errors`.
    """


@dataclass(frozen=True)
class ConnectionReport:
    """The result of a connection check.

    Why: A failed check is the answer to the question, not an error, so it is a normal result.

    Used by: returned by `LlmSettingsService.check`, read by `ai_ops.api.routes`.
    """

    connected: bool
    detail: str
    remote_models: tuple[str, ...]
    message_sent: bool
    model_answered: str | None
    usage: Usage | None


class LlmSettingsService:
    """Hold the current model, and report on the provider.

    Why: Keeps the rules for the model choice in one place. It knows that a key exists, but it never
    reads its value.

    Used by: `ai_ops.api.app.create_app` and `ai_ops.api.routes`.
    """

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        selection: ModelSelection,
        store: AppSettingsStore,
    ) -> None:
        """Store the collaborators.

        Why: `selection` is the object the provider reads for each request, so a change here applies
        to the next request.

        Used by: `ai_ops.api.app.create_app`.

        Args:
            settings: The validated settings.
            provider: The configured provider.
            selection: The current model.
            store: Where the choice is saved.
        """
        self._settings = settings
        self._provider = provider
        self._selection = selection
        self._store = store
        self._selectable = settings.llm_provider == "deepseek"
        self._models = list(DEEPSEEK_MODELS) if self._selectable else []

    def load(self) -> None:
        """Restore a saved model choice.

        Why: This runs at startup, after the migrations, since the table may not exist before them.
        A
        saved value is kept even if it is not in the built-in list, because a check may have found
        it, and the list may be out of date.

        Used by: the lifespan function in `ai_ops.api.app`.

        Side effects:
            Reads the database, and may change the current model.
        """
        saved = self._store.get(MODEL_SETTING) if self._selectable else None
        if saved:
            self._selection.current = saved
            if saved not in self._models:
                self._models.append(saved)

    def status(self) -> dict:
        """Describe the provider, with no secrets.

        Why: The API has no login, so this says whether a key exists, never what it is. The base
        URL is
        safe to show, because `Settings` refuses one that holds a password.

        Used by: `GET /settings/llm`.

        Returns:
            The fields of `LlmSettingsResponse`.
        """
        settings = self._settings
        return {
            "provider": settings.llm_provider,
            "model": self._selection.current,
            "models": self._models,
            "selectable": self._selectable,
            "key_configured": settings.llm_api_key is not None,
            "embeddings": settings.llm_embedding_model or "mock-word-hash",
            "base_url": settings.llm_base_url,
        }

    def select_model(self, model: str) -> None:
        """Change the model, for this run and after a restart.

        Why: Only known models are accepted. A check adds what the service itself lists.

        Used by: `PUT /settings/llm`.

        Args:
            model: The model's name.

        Raises:
            ModelNotSelectableError: This provider's model cannot be changed here.
            UnknownModelError: The model is not on the list.

        Side effects:
            Saves the choice. The next model call uses it.
        """
        if not self._selectable:
            raise ModelNotSelectableError(
                f"The model cannot be changed with the {self._settings.llm_provider} provider."
            )
        if model not in self._models:
            raise UnknownModelError(
                "That model is not available. Check the connection to refresh the list."
            )
        self._store.set(MODEL_SETTING, model)
        self._selection.current = model

    async def check(self, send_message: bool) -> ConnectionReport:
        """Check the connection, and optionally send one short message.

        Why: Listing models is free, so the page can do this each time it opens, and it proves that
        the
        address and the key work. It cannot prove that chat works: a key with no balance still lists
        models. So a person can also send one message, which costs a few tokens.

        Used by: `POST /settings/llm/check`.

        Args:
            send_message: Whether to send a test message after a good connection.

        Returns:
            The report. Failures are in the report, and they are not raised.

        Side effects:
            Makes network requests, and can use tokens. New models join the list.
        """
        result = await self._provider.check_connection()
        for model in result.models:
            if self._selectable and model not in self._models:
                self._models.append(model)
        report = {
            "connected": result.ok,
            "detail": result.detail,
            "remote_models": result.models,
            "message_sent": False,
            "model_answered": None,
            "usage": None,
        }
        if result.ok and send_message:
            try:
                generated = await self._provider.generate(
                    [ChatMessage(role="user", content=TEST_MESSAGE)]
                )
            except LLMError as exc:
                report["connected"] = False
                report["detail"] = describe_http_failure(getattr(exc, "status", None))
            else:
                report["message_sent"] = True
                report["model_answered"] = generated.model
                report["usage"] = generated.usage
                report["detail"] = "Connected. The model answered the test message."
        return ConnectionReport(**report)

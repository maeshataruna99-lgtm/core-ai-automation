"""Turn free text into a draft order with the model. It never saves anything.

Why: Typing orders by hand is slow, but a model can misread text, and the text may come from
outside.
So the model only fills in a form, and a person reviews it and saves it with the normal `POST
/records`.
No code path here can write.

Used by: `ai_ops.api.app` creates it. `ai_ops.api.routes.extract_record` calls it.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, get_args

from ai_ops.integrations.records import RecordStatus
from ai_ops.llm.prompt_loader import load_prompt
from ai_ops.llm.provider import ChatMessage, LLMProvider, Usage

PROMPT_VERSION = 1
FIELD_MAX_LENGTH = 100
NOT_FOUND_MESSAGE = (
    "No order could be read from this text. "
    "The mock provider cannot do this, so it needs a real model."
)


@dataclass(frozen=True)
class RecordDraft:
    """What a model read from the text. Any field can be empty.

    Why: This is a suggestion for a form, so it uses its own type, not `Record`, which is a saved
    order.

    Used by: `ExtractionResult`.
    """

    id: str | None = None
    customer: str | None = None
    item: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    """The result of an extraction.

    Why: Not finding an order is a normal result, so it is not an error.

    Used by: returned by `RecordExtractor.extract`, read by `ai_ops.api.routes`.
    """

    found: bool
    draft: RecordDraft
    message: str
    usage: Usage


def _parse_object(text: str) -> dict[str, Any] | None:
    """Find a JSON object in a model's reply.

    Why: Models often wrap JSON in a code fence or a sentence, even when asked not to.

    Used by: `RecordExtractor.extract`.

    Args:
        text: The reply.

    Returns:
        The object, or None if there is none.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_text(value: Any) -> str | None:
    """Keep a text field only if it follows the same rules as a saved order.

    Why: Model output is untrusted. An invalid field becomes empty, for a person to fill in.

    Used by: `_build_draft`.

    Args:
        value: The model's value.

    Returns:
        The trimmed text, or None.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if 0 < len(cleaned) <= FIELD_MAX_LENGTH else None


def _build_draft(data: dict[str, Any]) -> RecordDraft:
    """Check each field a model returned.

    Why: Only the four known fields are read, so anything extra is dropped.

    Used by: `RecordExtractor.extract`.

    Args:
        data: The parsed reply.

    Returns:
        The draft.
    """
    record_id = data.get("id")
    status = data.get("status")
    return RecordDraft(
        id=record_id
        if isinstance(record_id, str) and re.fullmatch(r"ORD-[0-9]+", record_id)
        else None,
        customer=_clean_text(data.get("customer")),
        item=_clean_text(data.get("item")),
        status=status if status in get_args(RecordStatus) else None,
    )


class RecordExtractor:
    """Ask the model to read an order from free text.

    Why: Rules go in the system message and the pasted text only in the user message, so that the
    text
    does not get the authority of instructions.

    Used by: `ai_ops.api.app.create_app` and `ai_ops.api.routes`.
    """

    def __init__(self, provider: LLMProvider) -> None:
        """Store the provider and load the prompts.

        Why: Loading here makes a missing prompt file fail at startup.

        Used by: `ai_ops.api.app.create_app`.

        Args:
            provider: The model provider.
        """
        self._provider = provider
        self._system_prompt = load_prompt("record_extract_system", PROMPT_VERSION)
        self._user_prompt = load_prompt("record_extract_user", PROMPT_VERSION)

    async def extract(self, text: str) -> ExtractionResult:
        """Read an order from text.

        Why: A one-line text keeps the prompt's structure the same, whatever was pasted.

        Used by: `POST /records/extract`.

        Args:
            text: The text, which is untrusted.

        Returns:
            The draft. Nothing is saved.

        Raises:
            LLMError: The model call failed.

        Side effects:
            Calls the model, which sends the text to the provider.
        """
        one_line = " ".join(text.split())
        generated = await self._provider.generate(
            [
                ChatMessage(role="system", content=self._system_prompt.text),
                ChatMessage(role="user", content=self._user_prompt.render(text=one_line)),
            ]
        )
        data = _parse_object(generated.text)
        draft = _build_draft(data) if data else RecordDraft()
        found = any((draft.id, draft.customer, draft.item, draft.status))
        return ExtractionResult(
            found=found,
            draft=draft,
            message="Review this draft before saving." if found else NOT_FOUND_MESSAGE,
            usage=generated.usage,
        )

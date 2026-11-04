"""Sending messages to customers, behind an interface, with a stub for the demo.

Why: Sending is the one action that cannot be undone, so it is isolated. The stub records
messages and sends nothing, which keeps the project free of real customer contact.

Used by: `ai_ops.services.workflow` and `ai_ops.api.app`.
"""

from typing import Protocol


class OutboundService(Protocol):
    """What the workflow needs from a messaging service.

    Why: Milestone 5 adds a real adapter behind the same interface.

    Used by: `ai_ops.services.workflow.TicketWorkflow`.
    """

    async def send(self, idempotency_key: str, body: str) -> dict:
        """Send a message once per key.

        Why: After a timeout, the caller cannot know whether the message went out. An idempotency
        key lets it retry safely, because a repeated key does not send again.

        Used by: the workflow's approved action.

        Args:
            idempotency_key: Identifies this message across retries.
            body: The message text.

        Returns:
            The delivery details.
        """
        ...


class StubOutboundService:
    """Implementation of `OutboundService` that only records messages.

    Why: The `sent` dict and `send_count` let tests and the demo check what would have gone out.

    Used by: `ai_ops.api.app.create_app` and tests.
    """

    def __init__(self) -> None:
        """Start with nothing sent.

        Why: Each app and test starts clean.

        Used by: `ai_ops.api.app.create_app` and tests.
        """
        self.sent: dict[str, str] = {}
        self.send_count = 0

    async def send(self, idempotency_key: str, body: str) -> dict:
        """Record a message, unless this key was already sent.

        Why: See `OutboundService.send`.

        Used by: the workflow's approved action.

        Args:
            idempotency_key: Identifies this message across retries.
            body: The message text.

        Returns:
            The delivery details. These are the same for a repeated key.

        Side effects:
            Adds to `sent` and `send_count` on the first call for a key.
        """
        if idempotency_key not in self.sent:
            self.sent[idempotency_key] = body
            self.send_count += 1
        return {"delivery_id": f"stub-{idempotency_key}", "delivered": True}

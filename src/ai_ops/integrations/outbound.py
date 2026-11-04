"""Sending messages to customers: the interface, an HTTP adapter, and a stub.

Why: Sending is the one action that cannot be undone, so it is isolated. The adapter pattern keeps
the third party's format and failures in one class. The stub records messages and sends nothing, and
it is the default, so the demo cannot contact anyone by accident.

Used by: `ai_ops.services.workflow` and `ai_ops.api.app`.
"""

from typing import Protocol

import httpx2

from ai_ops.config import Settings
from ai_ops.integrations.errors import IntegrationError


class OutboundError(IntegrationError):
    """A message could not be confirmed as delivered.

    Why: One type hides the HTTP library from callers. The message never holds the key, the customer
    message, or the service's response, so it is safe to show.

    Used by: raised by `HttpOutboundAdapter`, shown by `ai_ops.services.approval`.
    """


class OutboundService(Protocol):
    """What the workflow needs from a messaging service.

    Why: The workflow does not need to know which service is in use.

    Used by: `ai_ops.services.workflow.TicketWorkflow` and `ai_ops.api.app`.
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

        Raises:
            OutboundError: Delivery could not be confirmed.
        """
        ...

    async def aclose(self) -> None:
        """Release any open connections.

        Why: The app closes this at shutdown, whichever implementation it has.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
        ...


class StubOutboundService:
    """Implementation of `OutboundService` that only records messages.

    Why: The `sent` dict and `send_count` let tests and the demo check what would have gone out.

    Used by: `build_outbound` by default, and tests.
    """

    def __init__(self) -> None:
        """Start with nothing sent.

        Why: Each app and test starts clean.

        Used by: `build_outbound` and tests.
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

    async def aclose(self) -> None:
        """Do nothing, because the stub holds no resources.

        Why: `OutboundService` requires the method, so callers need no special case.

        Used by: the lifespan function in `ai_ops.api.app`.
        """


class HttpOutboundAdapter:
    """Implementation of `OutboundService` for a third-party HTTP messaging API.

    Why: This represents a typical vendor API: bearer key, JSON body, and an `Idempotency-Key`
    header. It has only been tested against simulated responses, because no real vendor is in use.

    Used by: `build_outbound` when `AI_OPS_OUTBOUND_PROVIDER=http`, and tests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the HTTP client.

        Why: The key goes in a default header, so no other code handles it after this point.

        Used by: `build_outbound` and tests.

        Args:
            base_url: API root, such as `https://api.example.com/v1`.
            api_key: Secret key, sent as a bearer token.
            timeout_seconds: Limit for each network step.
            transport: Replaces the network. Tests pass an in-memory transport.
        """
        self._client = httpx2.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def send(self, idempotency_key: str, body: str) -> dict:
        """Post one message and confirm that the service accepted it.

        Why: Every failure becomes `OutboundError`, including a 200 with no message id. Reporting
        that as sent would hide a failure from the person who approved the reply. There is no
        automatic retry here: approving again retries it, with a person in control.

        Used by: the workflow's approved action.

        Args:
            idempotency_key: Identifies this message across retries.
            body: The message text.

        Returns:
            The delivery id from the service.

        Raises:
            OutboundError: Delivery could not be confirmed.

        Side effects:
            Sends a real message through the third-party service.
        """
        try:
            response = await self._client.post(
                "/messages",
                json={"reference": idempotency_key, "body": body},
                headers={"Idempotency-Key": idempotency_key},
            )
        except httpx2.TimeoutException:
            raise OutboundError("Timeout calling the outbound service") from None
        except httpx2.TransportError as exc:
            message = f"Connection failure calling the outbound service: {type(exc).__name__}"
            raise OutboundError(message) from None

        if response.status_code >= 400:
            raise OutboundError(f"HTTP {response.status_code} from the outbound service")
        try:
            return {"delivery_id": response.json()["id"], "delivered": True}
        except (ValueError, KeyError, TypeError):
            raise OutboundError("Unexpected response from the outbound service") from None

    async def aclose(self) -> None:
        """Close the HTTP client.

        Why: Open connections would otherwise leak until the process exits.

        Used by: the lifespan function in `ai_ops.api.app`.
        """
        await self._client.aclose()


def build_outbound(settings: Settings) -> OutboundService:
    """Create the configured outbound service.

    Why: This is the only place that names concrete services, so configuration alone decides.

    Used by: `ai_ops.api.app.create_app` and `tests/test_config.py`.

    Args:
        settings: Validated settings. `Settings` guarantees that the adapter's values are present.

    Returns:
        The service. The caller must call `aclose` on it at shutdown.
    """
    if settings.outbound_provider == "stub":
        return StubOutboundService()
    return HttpOutboundAdapter(
        base_url=settings.outbound_base_url,
        api_key=settings.outbound_api_key.get_secret_value(),
        timeout_seconds=settings.outbound_timeout_seconds,
    )

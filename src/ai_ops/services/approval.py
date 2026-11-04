"""Human approval for actions that change data or reach the outside world.

Why: A model can be wrong or be manipulated by text it reads, so it can only propose such an
action. This service holds the proposal until a person decides. No tool lets a model make that
decision, which keeps the rule true whatever the model does.

The approval is a small state machine:

    pending -> executing -> executed
    pending -> rejected
    executing -> failed -> executing (approving again retries)

Used by: `ai_ops.agents.registry`, `ai_ops.services.workflow`, and `ai_ops.api.routes`.
"""

import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ai_ops.integrations.errors import IntegrationError
from ai_ops.logging import LOGGER_NAME

logger = logging.getLogger(f"{LOGGER_NAME}.approval")

ApprovalStatus = Literal["pending", "executing", "executed", "rejected", "failed"]
Executor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ApprovalNotFoundError(Exception):
    """No approval has this id.

    Why: The API maps this to a 404.

    Used by: raised by `ApprovalService.get`, mapped in `ai_ops.api.errors`.
    """


class ApprovalConflictError(Exception):
    """The decision does not fit the approval's current state.

    Why: The API maps this to a 409. It covers reversing a decision, and approving while the
    action is already running.

    Used by: raised by `ApprovalService.decide`, mapped in `ai_ops.api.errors`.
    """


class ActionFailedError(Exception):
    """An approved action failed. Approving again retries it.

    Why: The person who approved the action needs to know it failed. Its message only includes the
    reason for an `IntegrationError`, because other exceptions can hold secrets or internal details.
    Those go to the log only.

    Used by: raised by `ApprovalService.decide`, mapped in `ai_ops.api.errors`.
    """


@dataclass
class Approval:
    """A proposed action and its decision.

    Why: This is mutable because it tracks a process over time. Only `ApprovalService` should
    change it.

    Used by: `ApprovalService`, `ai_ops.services.workflow`, and `ai_ops.api.routes`.
    """

    id: str
    kind: str
    payload: dict[str, Any]
    status: ApprovalStatus = "pending"
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class ApprovalService:
    """Store approvals and run their actions once a person approves.

    Why: Actions are registered by kind and stored as plain data, so a proposal holds no code
    and could later be saved in a database.

    shortcut: approvals are lost on restart while records persist, so a pending change must be
    proposed again after a restart, save them in SQL for a real deployment.

    Used by: `ai_ops.api.app.create_app`, and the modules listed in the module docstring.
    """

    def __init__(self) -> None:
        """Start with no approvals and no actions.

        Why: The app registers actions while it is being assembled.

        Used by: `ai_ops.api.app.create_app` and tests.
        """
        self._approvals: dict[str, Approval] = {}
        self._executors: dict[str, Executor] = {}

    def register(self, kind: str, executor: Executor) -> None:
        """Register the code that runs when an action of this kind is approved.

        Why: This is the only route to the write code, so it is easy to audit.

        Used by: `ai_ops.agents.registry.ToolRegistry` and `ai_ops.services.workflow`.

        Args:
            kind: The action name.
            executor: Async function that takes the payload and returns a result.
        """
        self._executors[kind] = executor

    def request(self, kind: str, payload: dict[str, Any]) -> Approval:
        """Create a pending approval. Nothing runs yet.

        Why: This is all a model or a workflow can do on its own.

        Used by: `ToolRegistry.call` and `ai_ops.services.workflow`.

        Args:
            kind: A registered action name.
            payload: The action's validated input.

        Returns:
            The new approval.

        Raises:
            KeyError: The kind is not registered, which is a programming error.

        Side effects:
            Stores the approval and writes a log line.
        """
        self._executors[kind]  # Fail now, not when a person approves it later.
        approval = Approval(id=uuid.uuid4().hex, kind=kind, payload=payload)
        self._approvals[approval.id] = approval
        logger.info("approval requested", extra={"approval_id": approval.id, "tool": kind})
        return approval

    def get(self, approval_id: str) -> Approval:
        """Read an approval.

        Why: Callers report its current state.

        Used by: `decide`, `ai_ops.services.workflow`, and tests.

        Args:
            approval_id: The approval id.

        Returns:
            The approval.

        Raises:
            ApprovalNotFoundError: No approval has this id.
        """
        try:
            return self._approvals[approval_id]
        except KeyError:
            raise ApprovalNotFoundError(f"Approval {approval_id} does not exist") from None

    def list_all(self) -> list[Approval]:
        """Return every approval, newest first.

        Why: A person needs one place to see what is waiting. Dicts keep insertion order, so
        reversing gives newest first with no sorting.

        Used by: `ai_ops.api.routes.list_approvals`.

        Returns:
            The approvals.
        """
        return list(reversed(self._approvals.values()))

    async def decide(self, approval_id: str, approve: bool) -> Approval:
        """Apply a person's decision.

        Why: Repeating the same decision is safe: it returns the stored state and does not run
        the action again, so a double click or a client retry has no second side effect. The
        status changes before the first `await`, so a concurrent approval cannot also start.

        Used by: `ai_ops.api.routes.decide_approval`.

        Args:
            approval_id: The approval id.
            approve: True to approve and run the action, False to reject.

        Returns:
            The approval in its new state.

        Raises:
            ApprovalNotFoundError: No approval has this id.
            ApprovalConflictError: The decision does not fit the current state.
            ActionFailedError: The action failed. Approving again retries it.

        Side effects:
            On approval, runs the action. Writes a log line for the outcome.
        """
        approval = self.get(approval_id)
        if not approve:
            if approval.status not in ("pending", "rejected"):
                raise ApprovalConflictError(f"Approval is already {approval.status}")
            approval.status = "rejected"
            self._log_outcome(approval)
            return approval

        if approval.status == "executed":
            return approval
        if approval.status not in ("pending", "failed"):
            raise ApprovalConflictError(f"Approval is already {approval.status}")

        approval.status = "executing"
        try:
            approval.result = await self._executors[approval.kind](approval.payload)
        except Exception as exc:
            approval.status = "failed"
            logger.exception(
                "approved action failed",
                extra={"approval_id": approval.id, "tool": approval.kind},
            )
            # Only errors written to be shown get through. Any other text could hold a secret.
            reason = str(exc) if isinstance(exc, IntegrationError) else "An internal error occurred"
            raise ActionFailedError(f"The approved action failed: {reason}.") from exc
        approval.status = "executed"
        self._log_outcome(approval)
        return approval

    def _log_outcome(self, approval: Approval) -> None:
        """Write the audit line for a decision.

        Why: The log must show what was decided about each action.

        Used by: `decide`.

        Args:
            approval: The decided approval.
        """
        logger.info(
            "approval decided",
            extra={"approval_id": approval.id, "tool": approval.kind, "outcome": approval.status},
        )

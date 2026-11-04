"""The ticket workflow: classify, look up, draft, request approval, and send.

Why: This automates the routine part of answering a ticket, while a person stays in control of what
goes out. Unlike the agent, the code decides the steps and the model only fills in text, so
nothing in a ticket can change what happens.

Used by: `ai_ops.api.app` creates it. `ai_ops.api.routes.run_ticket_workflow` calls it.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from ai_ops.integrations.outbound import OutboundService
from ai_ops.integrations.records import RecordRepository
from ai_ops.llm.prompt_loader import load_prompt
from ai_ops.llm.provider import ChatMessage, LLMProvider
from ai_ops.logging import LOGGER_NAME
from ai_ops.services.approval import ApprovalService
from ai_ops.services.rag import RagService, format_context

logger = logging.getLogger(f"{LOGGER_NAME}.workflow")

PROMPT_VERSION = 1
CATEGORIES = ("refund", "delivery", "complaint")
SEND_ACTION = "send_ticket_reply"

TicketStatus = Literal["pending_approval", "sending", "sent", "rejected", "send_failed"]
# The approval is the source of truth, so the workflow never holds a status that could disagree.
_STATUS_BY_APPROVAL: dict[str, TicketStatus] = {
    "pending": "pending_approval",
    "executing": "sending",
    "executed": "sent",
    "rejected": "rejected",
    "failed": "send_failed",
}


class WorkflowConflictError(Exception):
    """A ticket id was reused with different text.

    Why: The id is an idempotency key. Returning the old result would silently ignore the new
    ticket.

    Used by: raised by `TicketWorkflow.run`, mapped to a 409 in `ai_ops.api.errors`.
    """


@dataclass(frozen=True)
class TicketState:
    """The current state of a ticket, as reported to a client.

    Why: A plain snapshot, so a caller cannot change the workflow's internal state.

    Used by: returned by `TicketWorkflow.run`.
    """

    ticket_id: str
    status: TicketStatus
    category: str
    record_id: str | None
    draft: str
    approval_id: str


@dataclass
class _Run:
    """The saved progress of one ticket.

    Why: Each finished step saves its result here, which is what makes a retry skip it. A field that
    is None has not run yet. `looked_up` is separate because None is a valid lookup result.

    shortcut: progress is lost on restart, save it in SQL for a real deployment.

    Used by: `TicketWorkflow.run`.
    """

    message: str
    category: str | None = None
    looked_up: bool = False
    record_id: str | None = None
    draft: str | None = None
    approval_id: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _parse_category(text: str) -> str:
    """Turn a model's answer into a known category.

    Why: Model output is untrusted, so only known labels can get through. Anything else becomes
    `other`, which is safe because the category only informs the person who reviews the draft.

    Used by: `TicketWorkflow.run`.

    Args:
        text: The model's answer.

    Returns:
        The first known category in the text, or `other`.
    """
    match = re.search(rf"\b({'|'.join(CATEGORIES)})\b", text, re.IGNORECASE)
    return match.group().lower() if match else "other"


class TicketWorkflow:
    """Run tickets through the steps, saving progress after each one.

    Why: Saving each step means a retry only repeats what failed, so it does not pay for finished
    model calls or create a second approval.

    Used by: `ai_ops.api.app.create_app` and `tests/test_workflow.py`.
    """

    def __init__(
        self,
        provider: LLMProvider,
        rag: RagService,
        records: RecordRepository,
        approvals: ApprovalService,
        outbound: OutboundService,
    ) -> None:
        """Store the collaborators, load the prompts, and register the send action.

        Why: Sending is registered as an approval action, so `ApprovalService` is the only way to
        reach it. Nothing in `run` can send.

        Used by: `ai_ops.api.app.create_app` and tests.

        Args:
            provider: The model provider.
            rag: The knowledge base.
            records: The record storage.
            approvals: The approval service.
            outbound: The messaging service.
        """
        self._provider = provider
        self._rag = rag
        self._records = records
        self._approvals = approvals
        self._outbound = outbound
        self._runs: dict[str, _Run] = {}
        self._classify_prompt = load_prompt("ticket_classify", PROMPT_VERSION)
        self._draft_system_prompt = load_prompt("ticket_draft_system", PROMPT_VERSION)
        self._draft_user_prompt = load_prompt("ticket_draft_user", PROMPT_VERSION)
        approvals.register(SEND_ACTION, self._send_reply)

    async def _send_reply(self, payload: dict) -> dict:
        """Send an approved reply.

        Why: The ticket id is the idempotency key. Retrying a failed send cannot send twice, even if
        the earlier attempt reached the customer and only the response was lost.

        Used by: `ApprovalService.decide`, after a person approves.

        Args:
            payload: The ticket id and the reply.

        Returns:
            The delivery details.

        Side effects:
            Sends a message through the outbound service.
        """
        return await self._outbound.send(payload["ticket_id"], payload["reply"])

    async def run(self, ticket_id: str, message: str) -> TicketState:
        """Run any unfinished steps and return the ticket's state.

        Why: Calling this again is always safe. It continues after a failure, and otherwise just
        reports the current state. The lock stops two identical requests from both running a
        step, since each model call is an `await` where the other request could start.

        Used by: `ai_ops.api.routes.run_ticket_workflow`.

        Args:
            ticket_id: The ticket id, which is the idempotency key.
            message: The ticket text, which is untrusted.

        Returns:
            The current state.

        Raises:
            WorkflowConflictError: The id was already used with different text.
            LLMError: A model call failed. Calling again retries that step.

        Side effects:
            Calls the model, creates one approval per ticket, and logs each step.
        """
        run = self._runs.setdefault(ticket_id, _Run(message=message))
        if run.message != message:
            raise WorkflowConflictError(f"Ticket {ticket_id} was already used with different text")

        # A one-line ticket keeps each prompt's structure the same, whatever the customer typed.
        ticket = " ".join(message.split())
        async with run.lock:
            if run.category is None:
                run.category = await self._classify(ticket)
                self._log_step(ticket_id, "classify")
            if not run.looked_up:
                run.record_id = self._find_record_id(ticket)
                run.looked_up = True
                self._log_step(ticket_id, "lookup")
            if run.draft is None:
                run.draft = await self._draft(ticket, run.record_id)
                self._log_step(ticket_id, "draft")
            if run.approval_id is None:
                payload = {"ticket_id": ticket_id, "reply": run.draft}
                run.approval_id = self._approvals.request(SEND_ACTION, payload).id
                self._log_step(ticket_id, "request_approval")

        approval = self._approvals.get(run.approval_id)
        return TicketState(
            ticket_id=ticket_id,
            status=_STATUS_BY_APPROVAL[approval.status],
            category=run.category,
            record_id=run.record_id,
            draft=run.draft,
            approval_id=run.approval_id,
        )

    async def _classify(self, ticket: str) -> str:
        """Ask the model for the ticket's category.

        Why: Lets a person sort and prioritise tickets. It affects no later step.

        Used by: `run`.

        Args:
            ticket: The one-line ticket text.

        Returns:
            A known category, or `other`.

        Raises:
            LLMError: The model call failed.
        """
        prompt = self._classify_prompt.render(ticket=ticket)
        result = await self._provider.generate([ChatMessage(role="user", content=prompt)])
        return _parse_category(result.text)

    def _find_record_id(self, ticket: str) -> str | None:
        """Find the order that a ticket mentions.

        Why: A regular expression is exact and free, so this does not need a model. Checking the
        repository means a ticket cannot claim an order that does not exist.

        Used by: `run`.

        Args:
            ticket: The ticket text.

        Returns:
            The id of an existing order, or None.
        """
        match = re.search(r"ORD-[0-9]+", ticket, re.IGNORECASE)
        if match is None:
            return None
        record = self._records.get(match.group().upper())
        return record.id if record else None

    async def _draft(self, ticket: str, record_id: str | None) -> str:
        """Ask the model to draft a reply from the order and the knowledge base.

        Why: Rules go in the system message, and the ticket and documents only in the user message,
        so that they do not get the authority of instructions.

        Used by: `run`.

        Args:
            ticket: The one-line ticket text.
            record_id: The order the ticket is about, if any.

        Returns:
            The draft reply.

        Raises:
            LLMError: A model call failed.
        """
        record = self._records.get(record_id) if record_id else None
        record_text = (
            json.dumps({"id": record.id, "item": record.item, "status": record.status})
            if record
            else "No matching order."
        )
        results = await self._rag.retrieve(ticket)
        result = await self._provider.generate(
            [
                ChatMessage(role="system", content=self._draft_system_prompt.text),
                ChatMessage(
                    role="user",
                    content=self._draft_user_prompt.render(
                        record=record_text, context=format_context(results), ticket=ticket
                    ),
                ),
            ]
        )
        return result.text

    def _log_step(self, ticket_id: str, step: str) -> None:
        """Write the audit line for a finished step.

        Why: The specification requires each step to be logged. The ticket text is left out,
        because it can hold personal data.

        Used by: `run`.

        Args:
            ticket_id: The ticket id.
            step: The step name.
        """
        logger.info("workflow step", extra={"ticket_id": ticket_id, "step": step})

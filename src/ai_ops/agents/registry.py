"""Tool registry: the list of tools a model may ask for, and the gate in front of them.

Why: A model's request is untrusted input, so all of them pass through one place that validates
the arguments, sends writes to approval, and writes the log. No code path can forget a check.

Used by: `ai_ops.agents.tools` registers tools. `ai_ops.agents.loop.Agent` calls them.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ai_ops.llm.provider import ToolSchema
from ai_ops.logging import LOGGER_NAME
from ai_ops.services.approval import ApprovalService

logger = logging.getLogger(f"{LOGGER_NAME}.agent")


@dataclass(frozen=True)
class Tool:
    """A capability offered to the model.

    Why: A Pydantic input model gives both validation and the JSON schema the model sees, so the
    two cannot disagree. `writes` is set in code by the developer, so nothing a model sends can
    change it.

    Used by: `ToolRegistry` and `ai_ops.agents.tools`.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Awaitable[dict[str, Any]]]
    writes: bool = False


@dataclass(frozen=True)
class ToolOutcome:
    """What happened when a tool was requested.

    Why: The API returns these, so a person can see what the agent did. `content` is the text
    sent back to the model.

    Used by: returned by `ToolRegistry.call`, collected by `ai_ops.agents.loop.Agent`.
    """

    name: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "pending_approval"]
    content: str
    approval_id: str | None = None


class ToolRegistry:
    """Hold the tools and control every call to them.

    Why: A write tool's handler is only given to `ApprovalService`. `call` never runs it, so the
    only way to run it is a person's approval.

    Used by: `ai_ops.agents.tools`, `ai_ops.agents.loop.Agent`, and `ai_ops.api.app`.
    """

    def __init__(self, approvals: ApprovalService) -> None:
        """Start with no tools.

        Why: The app decides which tools exist.

        Used by: `ai_ops.api.app.create_app` and tests.

        Args:
            approvals: Where write requests go.
        """
        self._approvals = approvals
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool. A write tool's handler goes to the approval service.

        Why: The input is validated again when the approved action runs, so the handler always
        receives a typed object.

        Used by: `ai_ops.agents.tools.register_default_tools`.

        Args:
            tool: The tool.
        """
        self._tools[tool.name] = tool
        if tool.writes:

            async def run_approved(payload: dict[str, Any]) -> dict[str, Any]:
                """Validate the stored payload and run the write."""
                return await tool.handler(tool.input_model.model_validate(payload))

            self._approvals.register(tool.name, run_approved)

    def schemas(self) -> list[ToolSchema]:
        """Describe the tools for the model.

        Why: The model needs to know what it may ask for.

        Used by: `ai_ops.agents.loop.Agent.run`.

        Returns:
            One schema per tool.
        """
        return [
            ToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_model.model_json_schema(),
            )
            for tool in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Handle one tool request from a model.

        Why: Errors are returned as data, not raised, so the model can correct itself on its
        next step. Only the validated input goes into an approval, so extra arguments that a
        model invents are dropped.

        Used by: `ai_ops.agents.loop.Agent.run`.

        Args:
            name: The requested tool.
            arguments: The model's arguments, which are untrusted.

        Returns:
            The outcome. For a write tool, it only says that an approval is pending.

        Raises:
            LLMError: A read tool needed the model provider and that call failed.

        Side effects:
            Runs read tools, creates approvals for write tools, and logs each call.
        """
        outcome = await self._dispatch(name, arguments)
        logger.info(
            "tool call",
            extra={"tool": name, "tool_input": arguments, "outcome": outcome.status},
        )
        return outcome

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Validate a request and route it.

        Why: Splitting this out gives `call` a single place to log every outcome.

        Used by: `call`.

        Args:
            name: The requested tool.
            arguments: The model's arguments.

        Returns:
            The outcome.

        Raises:
            LLMError: A read tool needed the model provider and that call failed.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error(name, arguments, f"Unknown tool: {name}")
        try:
            validated = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
            return _error(name, arguments, "; ".join(problems))

        if tool.writes:
            approval = self._approvals.request(name, validated.model_dump())
            content = {
                "status": "pending_approval",
                "approval_id": approval.id,
                "message": "Nothing has changed yet. A person must approve this action.",
            }
            return ToolOutcome(
                name=name,
                arguments=arguments,
                status="pending_approval",
                content=json.dumps(content),
                approval_id=approval.id,
            )

        result = await tool.handler(validated)
        return ToolOutcome(name=name, arguments=arguments, status="ok", content=json.dumps(result))


def _error(name: str, arguments: dict[str, Any], message: str) -> ToolOutcome:
    """Build an error outcome.

    Why: Used for each way a request can be invalid.

    Used by: `ToolRegistry._dispatch`.

    Args:
        name: The requested tool.
        arguments: The model's arguments.
        message: What was wrong, for the model to read.

    Returns:
        The outcome.
    """
    return ToolOutcome(
        name=name,
        arguments=arguments,
        status="error",
        content=json.dumps({"error": message}),
    )

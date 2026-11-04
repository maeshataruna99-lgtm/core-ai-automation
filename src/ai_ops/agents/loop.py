"""The agent loop: let a model call tools until it has an answer or a limit is reached.

Why: An agent is a model in a loop: it asks for a tool, reads the result, and decides what to do
next. Each step costs money and time, so the loop has a step limit and a time limit.

Used by: `ai_ops.api.routes.chat`.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from ai_ops.agents.registry import ToolOutcome, ToolRegistry
from ai_ops.llm.prompt_loader import load_prompt
from ai_ops.llm.provider import ChatMessage, LLMProvider, Usage

PROMPT_VERSION = 1


@dataclass(frozen=True)
class AgentResult:
    """The result of one agent run.

    Why: `stop_reason` tells a client why it ended, and `tool_calls` shows what the agent did. A
    client should trust these over the reply text, because a model can claim things it did not do.

    Used by: returned by `Agent.run`, read by `ai_ops.api.routes.chat`.
    """

    reply: str
    stop_reason: Literal["completed", "max_steps", "timeout"]
    tool_calls: list[ToolOutcome]
    usage: Usage
    latency_ms: float


class Agent:
    """Run the loop for one user message.

    Why: Each run starts fresh and holds no conversation history, which keeps it simple.

    Used by: `ai_ops.api.app.create_app` and `tests/test_agent.py`.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_steps: int,
        timeout_seconds: float,
    ) -> None:
        """Store the collaborators and load the system prompt.

        Why: Loading here makes a missing prompt file fail at startup.

        Used by: `ai_ops.api.app.create_app` and tests.

        Args:
            provider: The model provider.
            registry: The tools.
            max_steps: Maximum model calls per run.
            timeout_seconds: Time limit for the whole run.
        """
        self._provider = provider
        self._registry = registry
        self._max_steps = max_steps
        self._timeout_seconds = timeout_seconds
        self._system_prompt = load_prompt("agent_system", PROMPT_VERSION)

    async def run(self, message: str) -> AgentResult:
        """Handle one user message.

        Why: The limits end the run with a normal result, not an error, and the tool calls made
        so far are kept. Stopping is safe at any point, because write tools only create pending
        approvals.

        Used by: `ai_ops.api.routes.chat`.

        Args:
            message: The user's message.

        Returns:
            The reply, why the run stopped, the tool calls, and the usage and latency.

        Raises:
            LLMError: A model call failed. `ai_ops.api.errors` maps it to 502 or 504.

        Side effects:
            Calls the model, runs read tools, and creates approvals for write tools.
        """
        started_at = time.perf_counter()
        tool_calls: list[ToolOutcome] = []
        totals = [0, 0]

        async def loop() -> tuple[str, Literal["completed", "max_steps"]]:
            """Alternate model calls and tool calls until a reply or the step limit."""
            messages = [
                ChatMessage(role="system", content=self._system_prompt.text),
                ChatMessage(role="user", content=message),
            ]
            schemas = self._registry.schemas()
            for _ in range(self._max_steps):
                result = await self._provider.generate(messages, schemas)
                totals[0] += result.usage.input_tokens
                totals[1] += result.usage.output_tokens
                if not result.tool_calls:
                    return result.text, "completed"
                messages.append(
                    ChatMessage(role="assistant", content=result.text, tool_calls=result.tool_calls)
                )
                for call in result.tool_calls:
                    outcome = await self._registry.call(call.name, call.arguments)
                    tool_calls.append(outcome)
                    messages.append(
                        ChatMessage(role="tool", content=outcome.content, tool_call_id=call.id)
                    )
            return f"I stopped after {self._max_steps} steps without finishing.", "max_steps"

        try:
            async with asyncio.timeout(self._timeout_seconds):
                reply, stop_reason = await loop()
        except TimeoutError:
            reply, stop_reason = "I stopped because I reached the time limit.", "timeout"

        return AgentResult(
            reply=reply,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage=Usage(input_tokens=totals[0], output_tokens=totals[1]),
            latency_ms=round((time.perf_counter() - started_at) * 1000, 1),
        )

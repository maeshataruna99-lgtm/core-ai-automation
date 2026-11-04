"""The tools offered to the agent.

Why: Each tool is a small function with a typed input. The descriptions matter: they are what the
model reads when it decides which tool to use.

Used by: `ai_ops.api.app.create_app` and `tests/test_agent.py`.
"""

from typing import Any

from pydantic import BaseModel, Field

from ai_ops.agents.registry import Tool, ToolRegistry
from ai_ops.integrations.records import RecordRepository, RecordStatus
from ai_ops.services.rag import RagService


class SearchKnowledgeInput(BaseModel):
    """Input of `search_knowledge`.

    Why: The length limit keeps a model from sending huge text to the embedding API.

    Used by: `register_default_tools`.
    """

    query: str = Field(min_length=1, max_length=1000)


class LookupRecordInput(BaseModel):
    """Input of `lookup_record`.

    Why: The pattern rejects anything that is not an order id. `[0-9]` is used because the digit
    shorthand also matches digits from other scripts.

    Used by: `register_default_tools`.
    """

    record_id: str = Field(pattern=r"^ORD-[0-9]+$")


class UpdateRecordInput(BaseModel):
    """Input of `update_record`.

    Why: `RecordStatus` limits the change to known statuses, so a model cannot write free text.

    Used by: `register_default_tools`.
    """

    record_id: str = Field(pattern=r"^ORD-[0-9]+$")
    status: RecordStatus


def register_default_tools(
    registry: ToolRegistry, rag: RagService, records: RecordRepository
) -> None:
    """Create the three tools and add them to the registry.

    Why: Handlers are closures, so each one holds the dependency it needs and the registry stays
    independent of them.

    Used by: `ai_ops.api.app.create_app` and `tests/test_agent.py`.

    Args:
        registry: The registry to fill.
        rag: The knowledge base.
        records: The record storage.
    """

    async def search_knowledge(body: SearchKnowledgeInput) -> dict[str, Any]:
        """Return the relevant knowledge base chunks."""
        results = await rag.retrieve(body.query)
        return {
            "results": [
                {"source_id": result.chunk.source_id, "text": result.chunk.text}
                for result in results
            ]
        }

    async def lookup_record(body: LookupRecordInput) -> dict[str, Any]:
        """Return one record, or an error the model can read."""
        record = records.get(body.record_id)
        if record is None:
            return {"error": f"No record with id {body.record_id}"}
        return {"id": record.id, "item": record.item, "status": record.status}

    async def update_record(body: UpdateRecordInput) -> dict[str, Any]:
        """Change a record's status. This only runs after a person approves."""
        record = records.update_status(body.record_id, body.status)
        return {"id": record.id, "status": record.status}

    registry.register(
        Tool(
            name="search_knowledge",
            description="Search company policies and product notes. Returns text with source ids.",
            input_model=SearchKnowledgeInput,
            handler=search_knowledge,
        )
    )
    registry.register(
        Tool(
            name="lookup_record",
            description="Read one order by id, such as ORD-1001.",
            input_model=LookupRecordInput,
            handler=lookup_record,
        )
    )
    registry.register(
        Tool(
            name="update_record",
            description=(
                "Propose changing an order's status. A person must approve it before anything "
                "changes."
            ),
            input_model=UpdateRecordInput,
            handler=update_record,
            writes=True,
        )
    )

"""Business records (orders) and the interface used to read and change them.

Why: The agent and the workflow need records now, and Milestone 5 adds SQL. The `Protocol` means
the SQL version can replace the in-memory one without changing those callers.

Used by: `ai_ops.agents.tools`, `ai_ops.services.workflow`, and `ai_ops.api.app`.
"""

from dataclasses import dataclass, replace
from typing import Literal, Protocol

RecordStatus = Literal["processing", "shipped", "delivered", "cancelled"]


class RecordNotFoundError(Exception):
    """The requested record does not exist.

    Why: Callers can tell this apart from other failures.

    Used by: raised by `update_status`. An approval that hits it ends as `failed`.
    """


@dataclass(frozen=True)
class Record:
    """An order.

    Why: Frozen, so a change only happens through the repository and cannot happen by accident
    on a copy that a tool is holding.

    Used by: `RecordRepository` and its callers.
    """

    id: str
    customer: str
    item: str
    status: RecordStatus


class RecordRepository(Protocol):
    """What the application needs from any record storage.

    Why: The repository pattern keeps data access out of tools and services.

    Used by: `ai_ops.agents.tools` and `ai_ops.services.workflow`.
    """

    def get(self, record_id: str) -> Record | None:
        """Read one record.

        Why: Missing records are a normal result when looking something up.

        Used by: the `lookup_record` tool and the ticket workflow.

        Args:
            record_id: The record id.

        Returns:
            The record, or None if it does not exist.
        """
        ...

    def update_status(self, record_id: str, status: RecordStatus) -> Record:
        """Change a record's status.

        Why: This is the only write in the application. Only an approved action reaches it.

        Used by: the handler of the `update_record` tool.

        Args:
            record_id: The record id.
            status: The new status.

        Returns:
            The updated record.

        Raises:
            RecordNotFoundError: The record does not exist.
        """
        ...


class InMemoryRecordRepository:
    """Implementation of `RecordRepository` backed by a dict.

    Why: Enough to build and test the agent before the SQL work in Milestone 5.

    shortcut: data is lost on restart, Milestone 5 replaces this with SQLite.

    Used by: `ai_ops.api.app.create_app` and tests.
    """

    def __init__(self, records: list[Record]) -> None:
        """Store the starting records.

        Why: Tests and the demo each supply their own data.

        Used by: `ai_ops.api.app.create_app` and tests.

        Args:
            records: The starting records.
        """
        self._records = {record.id: record for record in records}

    def get(self, record_id: str) -> Record | None:
        """Read one record.

        Why: See `RecordRepository.get`.

        Used by: the `lookup_record` tool and the ticket workflow.

        Args:
            record_id: The record id.

        Returns:
            The record, or None if it does not exist.
        """
        return self._records.get(record_id)

    def update_status(self, record_id: str, status: RecordStatus) -> Record:
        """Change a record's status.

        Why: See `RecordRepository.update_status`.

        Used by: the handler of the `update_record` tool.

        Args:
            record_id: The record id.
            status: The new status.

        Returns:
            The updated record.

        Raises:
            RecordNotFoundError: The record does not exist.
        """
        record = self._records.get(record_id)
        if record is None:
            raise RecordNotFoundError(record_id)
        self._records[record_id] = replace(record, status=status)
        return self._records[record_id]


def sample_records() -> list[Record]:
    """Return the synthetic orders used by the demo.

    Why: The project must not use real customer data, so these are invented.

    Used by: `ai_ops.api.app.create_app`.

    Returns:
        The sample orders.
    """
    return [
        Record(id="ORD-1001", customer="Sample Customer A", item="Bolts", status="shipped"),
        Record(id="ORD-1002", customer="Sample Customer B", item="Bearings", status="processing"),
        Record(id="ORD-1003", customer="Sample Customer C", item="Gaskets", status="delivered"),
    ]

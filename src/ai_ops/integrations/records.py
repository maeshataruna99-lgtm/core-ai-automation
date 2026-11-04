"""Business records (orders) and the SQL repository that stores them.

Why: The repository pattern keeps SQL in one file. Tools, services, and routes only see `Record`
objects, so the storage could change without changing any of them.

Every query passes its values as `?` parameters, never by building the SQL string. The database then
treats a value as data only, which is what prevents SQL injection.

Used by: `ai_ops.agents.tools`, `ai_ops.services.workflow`, `ai_ops.api.routes`, and
`ai_ops.api.app`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import IntegrityError
from typing import Literal, Protocol

from ai_ops.integrations.database import apply_migrations, connect
from ai_ops.integrations.errors import IntegrationError

RecordStatus = Literal["processing", "shipped", "delivered", "cancelled"]


class RecordNotFoundError(IntegrationError):
    """The requested record does not exist.

    Why: Callers can tell this apart from other failures. Its message only names the id.

    Used by: raised by `update_status` and `ai_ops.api.routes.get_record`, mapped to a 404 in
    `ai_ops.api.errors`. An approval that hits it ends as `failed`.
    """


class RecordExistsError(IntegrationError):
    """An order with this id already exists.

    Why: Inserting must not overwrite an order. Changes go through an approval.

    Used by: raised by `add`, mapped to a 409 in `ai_ops.api.errors`.
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

    Why: Callers depend on this, not on SQLite.

    Used by: `ai_ops.agents.tools`, `ai_ops.services.workflow`, and `ai_ops.api.routes`.
    """

    def get(self, record_id: str) -> Record | None:
        """Read one record.

        Why: Missing records are a normal result when looking something up.

        Used by: the `lookup_record` tool, the ticket workflow, and `GET /records/{id}`.

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


class RecordInserter(Protocol):
    """Storage that can also add and list records.

    Why: This is separate from `RecordRepository`, the interface that the agent's tools use, so the
    tools have no way to reach these methods. Only a person, through the API, adds records.

    Used by: `ai_ops.api.routes`.
    """

    def add(self, record: Record) -> None:
        """Insert a new record.

        Why: See `SqliteRecordRepository.add`.

        Used by: `POST /records`.

        Args:
            record: The record.

        Raises:
            RecordExistsError: The id is already used.
        """
        ...

    def list_all(self) -> list[Record]:
        """Read all records.

        Why: See `SqliteRecordRepository.list_all`.

        Used by: `GET /records`.

        Returns:
            The records, sorted by id.
        """
        ...


class SqliteRecordRepository:
    """Implementation of `RecordRepository` on a SQLite file.

    Why: Each call opens its own connection, so this object holds no open resource and needs no
    closing.

    shortcut: calls block the event loop while they run, which is fine for small local queries, move
    them to a thread or an async driver if queries become slow.

    Used by: `ai_ops.api.app.create_app` and tests.
    """

    def __init__(self, path: Path) -> None:
        """Store the database path. This does not touch the file.

        Why: The app is assembled before it starts. Doing file work in `initialise` keeps building
        the app free of side effects.

        Used by: `ai_ops.api.app.create_app` and tests.

        Args:
            path: The database file.
        """
        self._path = path

    def initialise(self, seed: Iterable[Record]) -> None:
        """Bring the schema up to date and add any missing seed records.

        Why: This runs at every startup, so it must be safe to repeat. `INSERT OR IGNORE` skips a
        record that already exists, so a restart does not undo a real change.

        Used by: the lifespan function in `ai_ops.api.app`, and tests.

        Args:
            seed: The records to add when missing.

        Raises:
            sqlite3.Error: A migration or an insert failed.

        Side effects:
            Creates and changes the database file.
        """
        apply_migrations(self._path)
        with connect(self._path) as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO records (id, customer, item, status) VALUES (?, ?, ?, ?)",
                [(record.id, record.customer, record.item, record.status) for record in seed],
            )
            connection.commit()

    def get(self, record_id: str) -> Record | None:
        """Read one record.

        Why: See `RecordRepository.get`.

        Used by: the `lookup_record` tool, the ticket workflow, and `GET /records/{id}`.

        Args:
            record_id: The record id. It is passed as a parameter, so any text is safe here.

        Returns:
            The record, or None if it does not exist.
        """
        with connect(self._path) as connection:
            row = connection.execute(
                "SELECT id, customer, item, status FROM records WHERE id = ?", (record_id,)
            ).fetchone()
        return Record(*row) if row else None

    def add(self, record: Record) -> None:
        """Insert a new record.

        Why: The primary key rejects a repeated id, which holds even if two requests arrive
        together.

        Used by: `POST /records`.

        Args:
            record: The record.

        Raises:
            RecordExistsError: The id is already used.

        Side effects:
            Adds one row.
        """
        with connect(self._path) as connection:
            try:
                connection.execute(
                    "INSERT INTO records (id, customer, item, status) VALUES (?, ?, ?, ?)",
                    (record.id, record.customer, record.item, record.status),
                )
            except IntegrityError:
                raise RecordExistsError(f"Record {record.id} already exists") from None
            connection.commit()

    def list_all(self) -> list[Record]:
        """Read all records.

        Why: A person needs to see what was added. Sorting by id keeps the order stable.

        Used by: `GET /records`.

        Returns:
            The records, sorted by id.
        """
        with connect(self._path) as connection:
            rows = connection.execute(
                "SELECT id, customer, item, status FROM records ORDER BY id"
            ).fetchall()
        return [Record(*row) for row in rows]

    def update_status(self, record_id: str, status: RecordStatus) -> Record:
        """Change a record's status.

        Why: SQL reports success when an `UPDATE` matches no row, so `rowcount` must be checked.
        Otherwise an approved action could look successful while changing nothing.

        Used by: the handler of the `update_record` tool.

        Args:
            record_id: The record id.
            status: The new status.

        Returns:
            The updated record.

        Raises:
            RecordNotFoundError: The record does not exist.
            sqlite3.IntegrityError: The status breaks the schema's constraint.

        Side effects:
            Changes one row.
        """
        with connect(self._path) as connection:
            cursor = connection.execute(
                "UPDATE records SET status = ? WHERE id = ?", (status, record_id)
            )
            if cursor.rowcount == 0:
                raise RecordNotFoundError(f"Record {record_id} does not exist")
            connection.commit()
            row = connection.execute(
                "SELECT id, customer, item, status FROM records WHERE id = ?", (record_id,)
            ).fetchone()
        return Record(*row)


def sample_records() -> list[Record]:
    """Return the synthetic orders used by the demo.

    Why: The project must not use real customer data, so these are invented.

    Used by: `ai_ops.api.app`, which seeds them at startup.

    Returns:
        The sample orders.
    """
    return [
        Record(id="ORD-1001", customer="Sample Customer A", item="Bolts", status="shipped"),
        Record(id="ORD-1002", customer="Sample Customer B", item="Bearings", status="processing"),
        Record(id="ORD-1003", customer="Sample Customer C", item="Gaskets", status="delivered"),
    ]

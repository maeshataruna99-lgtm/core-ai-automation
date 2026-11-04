"""SQLite connections and schema migrations.

Why: `sqlite3` is in the standard library and needs no server, which fits a single-user learning
project. Migrations are numbered SQL files, so each schema change is reviewed in Git and any
database can be brought up to date automatically. A framework such as Alembic is not needed yet.

Used by: `ai_ops.integrations.records` and `tests/test_records.py`.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection and always close it.

    Why: One connection per operation is simple, and it avoids sharing a connection across threads.
    Python's `with sqlite3.connect(...)` commits but does not close, so this does both jobs clearly.
    Without a `commit`, changes are rolled back when the connection closes.

    Used by: `apply_migrations`, `ai_ops.integrations.records`, and tests.

    Args:
        path: The database file.

    Yields:
        The open connection.

    Raises:
        sqlite3.Error: The file cannot be opened.
    """
    connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def apply_migrations(path: Path, migrations: Traversable | None = None) -> None:
    """Bring a database up to the latest schema version.

    Why: `PRAGMA user_version` is an integer SQLite stores inside the file, so the database itself
    records which migrations ran, with no extra table. Each migration runs in one transaction, which
    SQLite also applies to schema changes, so a failure leaves the database as it was.

    The version is the count of applied files, so files must only ever be added at the end of the
    sorted list, and an applied file must never be edited.

    Used by: `ai_ops.integrations.records.SqliteRecordRepository.initialise`, and tests.

    Args:
        path: The database file. It and its parent directories are created if missing.
        migrations: Directory of `.sql` files, sorted by name. Defaults to the packaged ones.

    Raises:
        sqlite3.Error: A migration failed. Earlier migrations stay applied.

    Side effects:
        Creates and changes the database file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = migrations or files("ai_ops.integrations") / "migrations"
    scripts = sorted(
        (item for item in directory.iterdir() if item.name.endswith(".sql")),
        key=lambda item: item.name,
    )
    with connect(path) as connection:
        current = connection.execute("PRAGMA user_version").fetchone()[0]
        for version, script in enumerate(scripts, start=1):
            if version > current:
                sql = script.read_text(encoding="utf-8")
                # `version` is an integer from enumerate, so this string is not built from input.
                connection.executescript(f"BEGIN; {sql}; PRAGMA user_version = {version}; COMMIT;")

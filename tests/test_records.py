"""Behaviour tests for the SQLite repository and the migrations.

Why: These run against a real temporary SQLite file, because only real SQL can prove that queries,
constraints, transactions, and persistence work. A fake database would prove none of that.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import sqlite3
from pathlib import Path

import pytest

from ai_ops.integrations.database import apply_migrations, connect
from ai_ops.integrations.records import Record, RecordNotFoundError, SqliteRecordRepository

ORDER = Record(id="ORD-1", customer="Test Customer", item="Bolts", status="processing")
DATABASE = Path("test.db")


def build_repository(*records: Record) -> SqliteRecordRepository:
    """Create a repository on a new database file.

    Why: Each test needs a migrated database with a little data.

    Used by: the repository tests in this module.

    Args:
        *records: The starting records.

    Returns:
        The repository.
    """
    repository = SqliteRecordRepository(DATABASE)
    repository.initialise(records)
    return repository


def schema_version(path: Path) -> int:
    """Read the version stored in the database.

    Why: This is how the migration tests see what actually happened.

    Used by: the migration tests in this module.

    Args:
        path: The database file.

    Returns:
        The version.
    """
    with connect(path) as connection:
        return connection.execute("PRAGMA user_version").fetchone()[0]


def table_names(path: Path) -> set[str]:
    """List the tables in the database.

    Why: A failed migration must not leave a table behind.

    Used by: the migration tests in this module.

    Args:
        path: The database file.

    Returns:
        The table names.
    """
    with connect(path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {row[0] for row in rows}


def test_get_returns_a_stored_record():
    """Check reading a record.

    Why: This is the basic round trip through real SQL.

    Used by: pytest.
    """
    assert build_repository(ORDER).get("ORD-1") == ORDER


def test_get_returns_none_for_a_missing_record():
    """Check a record that does not exist.

    Why: Callers use None to report "not found" without an exception.

    Used by: pytest.
    """
    assert build_repository().get("ORD-404") is None


def test_update_status_changes_only_that_record():
    """Check that an update touches one row.

    Why: An update with a wrong `WHERE` clause would change every order.

    Used by: pytest.
    """
    other = Record(id="ORD-2", customer="Other", item="Nuts", status="processing")
    repository = build_repository(ORDER, other)

    updated = repository.update_status("ORD-1", "shipped")

    assert updated.status == "shipped"
    assert repository.get("ORD-1").status == "shipped"
    assert repository.get("ORD-2").status == "processing"


def test_update_status_of_a_missing_record_raises():
    """Check an update that matches no row.

    Why: SQL reports no error for this, so the code has to check. Otherwise an approved action
    would look successful while changing nothing.

    Used by: pytest.
    """
    with pytest.raises(RecordNotFoundError, match="ORD-404"):
        build_repository().update_status("ORD-404", "shipped")


def test_data_survives_a_new_repository_on_the_same_file():
    """Check persistence.

    Why: This is the reason for moving from memory to SQL. Every call opens its own connection, so a
    new object sees only what was really committed.

    Used by: pytest.
    """
    build_repository(ORDER).update_status("ORD-1", "delivered")

    assert SqliteRecordRepository(DATABASE).get("ORD-1").status == "delivered"


def test_initialising_again_does_not_overwrite_existing_data():
    """Check a restart of the application.

    Why: Sample data is seeded at each startup, and that must not undo a real change.

    Used by: pytest.
    """
    build_repository(ORDER).update_status("ORD-1", "delivered")

    build_repository(ORDER)

    assert SqliteRecordRepository(DATABASE).get("ORD-1").status == "delivered"


@pytest.mark.parametrize(
    "attack",
    ["' OR '1'='1", "ORD-1'; DROP TABLE records; --", "ORD-1' --"],
)
def test_sql_injection_in_an_id_matches_nothing_and_changes_nothing(attack):
    """Check classic SQL injection text as an id, for both a read and a write.

    Why: With parameters the database treats the whole string as one value. If a query were built
    from strings, the first attack would return a row, and the second would drop the table.

    Used by: pytest.

    Args:
        attack: The malicious id.
    """
    repository = build_repository(ORDER)

    assert repository.get(attack) is None
    with pytest.raises(RecordNotFoundError):
        repository.update_status(attack, "cancelled")

    assert repository.get("ORD-1") == ORDER


def test_database_rejects_an_unknown_status():
    """Check the constraint in the schema.

    Why: The database is the last line of defence, in case a bad value gets past the code.

    Used by: pytest.
    """
    repository = build_repository(ORDER)

    with pytest.raises(sqlite3.IntegrityError):
        repository.update_status("ORD-1", "exploded")


def test_connection_can_create_missing_parent_directories():
    """Check a database path in a directory that does not exist yet.

    Why: On a fresh clone there is nothing at the default path.

    Used by: pytest.
    """
    path = Path("new") / "nested" / "app.db"

    SqliteRecordRepository(path).initialise([ORDER])

    assert path.exists()


def write_migrations(directory: Path, **files: str) -> Path:
    """Write migration files for a test.

    Why: The migration runner must be tested with failures that the real files do not have.

    Used by: the migration tests in this module.

    Args:
        directory: Where to write them.
        **files: File name without its `.sql` suffix, mapped to the SQL.

    Returns:
        The directory.
    """
    directory.mkdir()
    for name, sql in files.items():
        (directory / f"{name}.sql").write_text(sql, encoding="utf-8")
    return directory


def test_migrations_apply_in_number_order_and_set_the_version():
    """Check the order of migrations.

    Why: The second file depends on the first. `010` sorts after `002` only because of the zero
    padding, so the names are part of the contract.

    Used by: pytest.
    """
    migrations = write_migrations(
        Path("migrations"),
        m010_third="ALTER TABLE a ADD COLUMN y TEXT;",
        m001_first="CREATE TABLE a (id INTEGER);",
        m002_second="ALTER TABLE a ADD COLUMN x TEXT;",
    )

    apply_migrations(DATABASE, migrations)

    assert schema_version(DATABASE) == 3


def test_migrations_only_run_once():
    """Check running the migrations again.

    Why: They run at every startup. Without tracking, `CREATE TABLE` would fail the second time.

    Used by: pytest.
    """
    migrations = write_migrations(Path("migrations"), m001_first="CREATE TABLE a (id INTEGER);")

    apply_migrations(DATABASE, migrations)
    apply_migrations(DATABASE, migrations)

    assert schema_version(DATABASE) == 1


def test_only_new_migrations_run_on_an_existing_database():
    """Check an upgrade.

    Why: This is the purpose of migrations: changing a database that already holds data.

    Used by: pytest.
    """
    migrations = write_migrations(Path("migrations"), m001_first="CREATE TABLE a (id INTEGER);")
    apply_migrations(DATABASE, migrations)
    with connect(DATABASE) as connection:
        connection.execute("INSERT INTO a VALUES (7)")
        connection.commit()

    (migrations / "m002_second.sql").write_text("ALTER TABLE a ADD COLUMN x TEXT;")
    apply_migrations(DATABASE, migrations)

    assert schema_version(DATABASE) == 2
    with connect(DATABASE) as connection:
        assert connection.execute("SELECT id, x FROM a").fetchall()[0][0] == 7


def test_failed_migration_changes_nothing():
    """Check a migration that fails halfway.

    Why: A half-applied migration leaves a database that no later run can fix automatically. The
    table from the first statement must be gone, and the version must not move.

    Used by: pytest.
    """
    migrations = write_migrations(
        Path("migrations"),
        m001_first="CREATE TABLE a (id INTEGER);",
        m002_broken="CREATE TABLE b (id INTEGER); THIS IS NOT SQL;",
    )

    with pytest.raises(sqlite3.Error):
        apply_migrations(DATABASE, migrations)

    assert schema_version(DATABASE) == 1
    assert table_names(DATABASE) == {"a"}

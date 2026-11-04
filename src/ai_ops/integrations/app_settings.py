"""Store settings that a person changes while the app runs.

Why: The model chosen on the Settings page must survive a restart. Environment variables are
read-only
for a running app, and editing `.env` from the API would put the file that holds the key within
reach
of a service with no login. So these values go in SQLite. Secrets never go here.

Used by: `ai_ops.services.llm_settings`.
"""

from pathlib import Path

from ai_ops.integrations.database import connect


class AppSettingsStore:
    """Read and write named values in the `settings` table.

    Why: Each call opens its own connection, like `SqliteRecordRepository`, so this holds no
    resource.
    The table is created by the migrations that run at startup.

    Used by: `ai_ops.services.llm_settings.LlmSettingsService`.
    """

    def __init__(self, path: Path) -> None:
        """Store the database path. This does not touch the file.

        Why: The app is assembled before the migrations run at startup.

        Used by: `ai_ops.api.app.create_app`.

        Args:
            path: The database file.
        """
        self._path = path

    def get(self, name: str) -> str | None:
        """Read a value.

        Why: A value that was never set is a normal case, so this does not raise.

        Used by: `LlmSettingsService.load`.

        Args:
            name: The setting's name.

        Returns:
            The value, or None if it was never set.
        """
        with connect(self._path) as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE name = ?", (name,)
            ).fetchone()
        return row[0] if row else None

    def set(self, name: str, value: str) -> None:
        """Save a value, replacing any earlier one.

        Why: A setting has one current value, so an upsert fits.

        Used by: `LlmSettingsService.select_model`.

        Args:
            name: The setting's name.
            value: The value.

        Side effects:
            Changes one row.
        """
        with connect(self._path) as connection:
            connection.execute(
                "INSERT INTO settings (name, value) VALUES (?, ?) "
                "ON CONFLICT (name) DO UPDATE SET value = excluded.value",
                (name, value),
            )
            connection.commit()

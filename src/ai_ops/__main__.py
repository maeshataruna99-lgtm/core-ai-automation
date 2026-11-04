"""Command-line entry point: `python -m ai_ops`.

Why: Running through this module validates settings first, so a bad configuration ends
with a short message and exit code 1 instead of a traceback from inside uvicorn.

Used by: people and scripts starting the API, and `tests/test_config.py`.
"""

import sys

import uvicorn

from ai_ops.api.app import create_app
from ai_ops.config import ConfigError, Settings


def main() -> int:
    """Load settings and run the API until it is stopped.

    Why: Returning an exit code keeps the failure path testable and lets `sys.exit` handle
    the process exit in one place.

    Used by: the `__main__` block below.

    Returns:
        0 after a normal shutdown, 1 when the configuration is invalid.

    Side effects:
        Reads environment variables, binds a network port, and blocks while serving.
    """
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    # access_log is off because RequestContextMiddleware writes one structured line per request.
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

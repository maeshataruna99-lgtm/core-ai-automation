"""FastAPI application factory.

Why: A factory builds the app from settings passed as an argument. Importing this module
has no side effects, and each test can build its own app without environment variables.

Used by: `ai_ops.__main__.main` and `tests/test_api.py`.
"""

from fastapi import FastAPI

from ai_ops.api.errors import register_error_handlers
from ai_ops.api.middleware import RequestContextMiddleware
from ai_ops.api.routes import router
from ai_ops.config import Settings
from ai_ops.logging import configure_logging


def create_app(settings: Settings) -> FastAPI:
    """Build the FastAPI app.

    Why: This is the one place where the API is assembled, so the full setup can be read
    from top to bottom.

    Used by: `ai_ops.__main__.main` and the `client` fixtures in `tests/test_api.py`.

    Args:
        settings: Validated application settings.

    Returns:
        The configured app. It is not running yet.

    Side effects:
        Configures the `ai_ops` logger.
    """
    configure_logging(settings.log_level)
    app = FastAPI(title="AI Operations Assistant", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(router)
    return app

"""FastAPI application factory.

Why: A factory builds the app from settings passed as an argument. Importing this module
has no side effects, and each test can build its own app without environment variables.

Used by: `ai_ops.__main__.main` and `tests/test_api.py`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ai_ops.agents.loop import Agent
from ai_ops.agents.registry import ToolRegistry
from ai_ops.agents.tools import register_default_tools
from ai_ops.api.errors import register_error_handlers
from ai_ops.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from ai_ops.api.routes import router
from ai_ops.config import Settings
from ai_ops.integrations.outbound import OutboundService, build_outbound
from ai_ops.integrations.records import SqliteRecordRepository, sample_records
from ai_ops.llm.factory import build_provider
from ai_ops.llm.provider import LLMProvider
from ai_ops.logging import configure_logging
from ai_ops.retrieval.documents import load_documents
from ai_ops.retrieval.vector_store import InMemoryVectorStore
from ai_ops.services.approval import ApprovalService
from ai_ops.services.rag import RagService
from ai_ops.services.workflow import TicketWorkflow

# Far more than any valid request needs today. A question is limited to 1000 characters.
MAX_BODY_BYTES = 64 * 1024
# The web interface: plain files, with no build step. They only call the public API.
UI_DIRECTORY = Path(__file__).parent.parent / "ui"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare the database and knowledge base at startup, and close clients at shutdown.

    Why: FastAPI runs the code before `yield` once before serving, and the code after it at
    shutdown. Doing this work here means the first request does not have to wait for it, and a
    failure stops the startup with a visible error.

    Used by: FastAPI, through the `lifespan` argument in `create_app`.

    Args:
        app: The app being started.

    Yields:
        Control back to FastAPI while the app serves requests.

    Raises:
        sqlite3.Error: A migration failed. The app does not start.
        LLMError: Embedding the documents failed. The app does not start.
    """
    app.state.records.initialise(sample_records())
    documents = load_documents(app.state.knowledge_dir)
    await app.state.rag_service.ingest(documents)
    yield
    await app.state.outbound.aclose()
    await app.state.llm_provider.aclose()


def create_app(
    settings: Settings,
    llm_provider: LLMProvider | None = None,
    outbound: OutboundService | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Why: This is the one place where the API is assembled, so the full setup can be read
    from top to bottom. The provider is stored in `app.state`, FastAPI's place for objects
    shared across requests.

    Used by: `ai_ops.__main__.main` and the fixtures in `tests/test_api.py`.

    Args:
        settings: Validated application settings.
        llm_provider: Replaces the configured provider. Tests pass a failing fake.
        outbound: Replaces the configured outbound service. Tests pass a failing fake.

    Returns:
        The configured app. It is not running yet.

    Side effects:
        Configures the `ai_ops` logger.
    """
    configure_logging(settings.log_level)
    app = FastAPI(title="AI Operations Assistant", version="0.1.0", lifespan=_lifespan)
    app.state.llm_provider = llm_provider or build_provider(settings)
    app.state.provider_name = settings.llm_provider
    app.state.knowledge_dir = settings.knowledge_dir
    app.state.rag_service = RagService(
        provider=app.state.llm_provider,
        store=InMemoryVectorStore(),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    app.state.records = SqliteRecordRepository(settings.database_path)
    app.state.outbound = outbound or build_outbound(settings)
    app.state.approvals = ApprovalService()
    registry = ToolRegistry(app.state.approvals)
    register_default_tools(registry, app.state.rag_service, app.state.records)
    app.state.agent = Agent(
        provider=app.state.llm_provider,
        registry=registry,
        max_steps=settings.agent_max_steps,
        timeout_seconds=settings.agent_timeout_seconds,
    )
    app.state.ticket_workflow = TicketWorkflow(
        provider=app.state.llm_provider,
        rag=app.state.rag_service,
        records=app.state.records,
        approvals=app.state.approvals,
        outbound=app.state.outbound,
    )
    # The middleware added last runs first, so the request id exists before the body is read.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(router)
    app.mount("/ui", StaticFiles(directory=UI_DIRECTORY, html=True), name="ui")
    return app

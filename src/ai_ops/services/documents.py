"""Add and remove knowledge base documents while the app runs.

Why: A person should be able to add a document and ask about it at once, with no restart. Files go
in a
separate uploads folder, which Git ignores, because inserted text may be private, and the tracked
knowledge folder is published with the repository.

Used by: `ai_ops.api.app` creates it. `ai_ops.api.routes` calls it.
"""

from pathlib import Path

from ai_ops.retrieval.documents import Document, load_documents
from ai_ops.services.rag import RagService

UPLOAD_PREFIX = "uploads/"


class DocumentExistsError(Exception):
    """A document with this name already exists.

    Why: Replacing it silently could lose text, and old chunks would stay searchable.

    Used by: raised by `DocumentService.add`, mapped to a 409 in `ai_ops.api.errors`.
    """


class DocumentNotFoundError(Exception):
    """No uploaded document has this name.

    Why: The API maps this to a 404. Sample documents count as not found here, on purpose.

    Used by: raised by `DocumentService.remove`, mapped in `ai_ops.api.errors`.
    """


class DocumentService:
    """Keep the uploads folder and the search index in step.

    Why: A document must be in both or neither, or it would either disappear after a restart, or be
    on
    disk and impossible to find.

    Used by: `ai_ops.api.app.create_app` and `ai_ops.api.routes`.
    """

    def __init__(self, rag: RagService, uploads_dir: Path) -> None:
        """Store the collaborators.

        Why: The folder is created on first use, so a read-only setup still starts.

        Used by: `ai_ops.api.app.create_app`.

        Args:
            rag: The search index.
            uploads_dir: The folder for inserted documents.
        """
        self._rag = rag
        self._uploads_dir = uploads_dir

    def load_uploads(self) -> list[Document]:
        """Read the documents added on earlier runs.

        Why: The vector store is in memory, so these must be indexed again at each startup.

        Used by: the lifespan function in `ai_ops.api.app`.

        Returns:
            The documents, with ids that mark them as uploads.
        """
        if not self._uploads_dir.is_dir():
            return []
        return [
            Document(id=f"{UPLOAD_PREFIX}{document.id}", text=document.text)
            for document in load_documents(self._uploads_dir)
        ]

    async def add(self, name: str, text: str) -> tuple[str, int]:
        """Save a document and index it.

        Why: Opening with mode `x` fails if the file exists, even if two requests arrive together.
        The
        file is written first and removed if indexing fails, so a failure leaves nothing behind.

        Used by: `POST /documents`.

        Args:
            name: The file name, which the API has already validated.
            text: The document's text.

        Returns:
            The document's id and its number of chunks.

        Raises:
            DocumentExistsError: The name is already used.
            LLMError: Embedding failed. Nothing is kept.

        Side effects:
            Creates a file, and may call the provider to embed the text.
        """
        document_id = f"{UPLOAD_PREFIX}{name}"
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        path = self._uploads_dir / name
        try:
            # newline="" keeps the text exactly as it was sent.
            with path.open("x", encoding="utf-8", newline="") as file:
                file.write(text)
        except FileExistsError:
            raise DocumentExistsError(f"A document named {name} already exists.") from None
        try:
            await self._rag.ingest([Document(id=document_id, text=text)])
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return document_id, self._rag.documents[document_id]

    def remove(self, name: str) -> None:
        """Delete an uploaded document from the disk and the index.

        Why: Only looking in the uploads folder means that a repository document cannot be reached,
        whatever name is given.

        Used by: `DELETE /documents/{name}`.

        Args:
            name: The file name, which the API has already validated.

        Raises:
            DocumentNotFoundError: No uploaded document has this name.

        Side effects:
            Deletes a file, and changes the index.
        """
        path = self._uploads_dir / name
        if not path.is_file():
            raise DocumentNotFoundError(f"No uploaded document is named {name}.")
        path.unlink()
        self._rag.remove_document(f"{UPLOAD_PREFIX}{name}")

    def list_all(self) -> list[dict]:
        """List all documents in the index.

        Why: A person needs to see what the assistant can use, and which ones can be removed.

        Used by: `GET /documents`.

        Returns:
            The id, chunk count, and whether it can be removed, sorted by id.
        """
        return [
            {
                "id": document_id,
                "chunks": chunks,
                "removable": document_id.startswith(UPLOAD_PREFIX),
            }
            for document_id, chunks in sorted(self._rag.documents.items())
        ]

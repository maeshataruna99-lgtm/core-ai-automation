"""Behaviour tests for chunking, the vector store, and document loading.

Why: Retrieval quality depends on these parts, and they need no model, so their edge cases
can be pinned exactly with literal expected values.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import pytest

from ai_ops.retrieval.chunking import chunk_text
from ai_ops.retrieval.documents import load_documents
from ai_ops.retrieval.vector_store import Chunk, InMemoryVectorStore


def make_chunk(document_id: str, position: int = 0) -> Chunk:
    """Build a chunk with placeholder text.

    Why: The store tests care about ids and vectors, not text.

    Used by: the vector store tests in this module.

    Args:
        document_id: The document id.
        position: The chunk position.

    Returns:
        The chunk.
    """
    return Chunk(document_id=document_id, position=position, text="text")


def test_chunks_overlap_when_there_is_no_space_to_break_at():
    """Check the exact windows for text without spaces.

    Why: Overlap keeps a fact that crosses a boundary whole in at least one chunk.

    Used by: pytest.
    """
    assert chunk_text("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij"]


def test_chunks_break_at_whitespace_instead_of_inside_a_word():
    """Check that a word is not cut when a space is available.

    Why: Half words embed badly and read badly in a citation.

    Used by: pytest.
    """
    assert chunk_text("one two three four", size=9, overlap=0) == ["one two", "three", "four"]


def test_text_shorter_than_size_stays_whole():
    """Check the boundary where no split is needed.

    Why: Short documents must not gain empty or repeated chunks.

    Used by: pytest.
    """
    assert chunk_text("short text", size=100, overlap=10) == ["short text"]


def test_text_of_exactly_size_gives_one_chunk():
    """Check the exact boundary.

    Why: Off-by-one mistakes in a sliding window usually appear exactly here.

    Used by: pytest.
    """
    assert chunk_text("abcd", size=4, overlap=1) == ["abcd"]


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_blank_text_gives_no_chunks(text):
    """Check empty and whitespace-only input.

    Why: A blank chunk would waste an embedding call and could be retrieved as a source.

    Used by: pytest.

    Args:
        text: The blank input.
    """
    assert chunk_text(text, size=10, overlap=2) == []


@pytest.mark.parametrize(("size", "overlap"), [(4, 4), (4, 5), (0, 0), (4, -1)])
def test_invalid_sizes_raise_instead_of_looping_forever(size, overlap):
    """Check the settings that would stop the window from moving forward.

    Why: An overlap as large as the size never advances, which would hang the startup.

    Used by: pytest.

    Args:
        size: The chunk size.
        overlap: The overlap.
    """
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)


def test_search_on_empty_store_returns_nothing():
    """Check the empty store.

    Why: The API must answer "not found" with no documents loaded, not crash.

    Used by: pytest.
    """
    assert InMemoryVectorStore().search([1.0, 0.0], top_k=3) == []


def test_search_ranks_by_similarity_and_applies_top_k():
    """Check ranking order and the result limit, with hand-computed scores.

    Why: The vectors are chosen so the expected cosine values are known: same direction 1.0,
    45 degrees about 0.707, opposite -1.0.

    Used by: pytest.
    """
    store = InMemoryVectorStore()
    store.add(
        [make_chunk("opposite"), make_chunk("exact"), make_chunk("close")],
        [[-1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk.document_id for result in results] == ["exact", "close"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.70710678)


def test_equal_scores_are_ordered_by_source_id():
    """Check the tie-break.

    Why: Retrieval must be deterministic, and equal scores would otherwise depend on
    insertion order.

    Used by: pytest.
    """
    store = InMemoryVectorStore()
    store.add([make_chunk("b"), make_chunk("a", 1), make_chunk("a", 0)], [[1.0], [1.0], [1.0]])

    results = store.search([1.0], top_k=3)

    assert [result.chunk.source_id for result in results] == ["a#0", "a#1", "b#0"]


def test_zero_vector_scores_zero_instead_of_dividing_by_zero():
    """Check a vector with no direction.

    Why: The mock gives a zero vector for text with no useful words, such as "the".

    Used by: pytest.
    """
    store = InMemoryVectorStore()
    store.add([make_chunk("a")], [[1.0, 0.0]])

    assert store.search([0.0, 0.0], top_k=1)[0].score == 0.0


def test_add_rejects_mismatched_chunks_and_vectors():
    """Check that each chunk gets exactly one vector.

    Why: A count mismatch would attach vectors to the wrong chunks without any error.

    Used by: pytest.
    """
    with pytest.raises(ValueError):
        InMemoryVectorStore().add([make_chunk("a")], [[1.0], [2.0]])


def test_load_documents_reads_supported_files_in_a_stable_order(tmp_path):
    """Check file filtering, ids, and order.

    Why: Ids must not depend on the operating system or the directory listing order, because
    they appear in citations and in the evaluation set.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "refunds.md").write_text("# Refunds", encoding="utf-8")
    (tmp_path / "about.txt").write_text("About us", encoding="utf-8")
    (tmp_path / "image.png").write_text("not text", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [(document.id, document.text) for document in documents] == [
        ("about.txt", "About us"),
        ("policies/refunds.md", "# Refunds"),
    ]


def test_missing_directory_gives_no_documents(tmp_path):
    """Check a knowledge directory that does not exist.

    Why: The API stays usable and answers "not found". `GET /health` shows zero documents.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    assert load_documents(tmp_path / "missing") == []


def test_undecodable_file_is_skipped_and_others_still_load(tmp_path):
    """Check one broken file among good ones.

    Why: A single bad file should not take the whole knowledge base down.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe\xff")
    (tmp_path / "good.txt").write_text("Good", encoding="utf-8")

    assert [document.id for document in load_documents(tmp_path)] == ["good.txt"]

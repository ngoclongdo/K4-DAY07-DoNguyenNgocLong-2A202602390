from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Build a normalized stored record for one document.
        Copies metadata and ensures metadata['doc_id'] is present for deletion.
        """
        meta = dict(doc.metadata) if doc.metadata else {}
        if "doc_id" not in meta:
            meta["doc_id"] = doc.id.split("#")[0] if "#" in doc.id else doc.id

        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": meta,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """
        Run in-memory similarity search over provided records.
        Uses dot product since embeddings are normalized.
        Omits 'embedding' vector from the returned results to keep output clean.
        """
        if not records or top_k <= 0:
            return []

        query_vec = self._embedding_fn(query)
        scored: list[tuple[float, dict[str, Any]]] = []

        for rec in records:
            score = _dot(query_vec, rec["embedding"])
            scored.append((score, rec))

        # Sắp xếp giảm dần theo điểm tương đồng score
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, rec in scored[:top_k]:
            results.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score,
            })

        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        Pre-filtering ensures k candidate slots are not wasted on unwanted documents.
        """
        if metadata_filter:
            filtered_records = [
                rec for rec in self._store
                if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            filtered_records = self._store

        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_count = len(self._store)
        self._store = [
            rec for rec in self._store
            if rec["metadata"].get("doc_id") != doc_id and rec["id"] != doc_id
        ]
        return len(self._store) < initial_count

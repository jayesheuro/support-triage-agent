"""
app/services/kb/retriever.py

Responsibility: Query retrieval against the ShopLite Knowledge Base.
Embeds incoming queries, retrieves nearest chunks from LanceDB, computes relevance
scores, and applies the relevance threshold from configs/app.yaml. Exposes clean
kb_gap detection for the Phase 3 Resolver agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.kb.vector_store import KBVectorStore
from app.services.providers.base import BaseEmbedder
from app.services.providers.factory import get_providers

logger = get_logger(__name__)


class RetrievedChunk(BaseModel):
    """A scored Knowledge Base chunk returned from a retrieval query."""

    id: str = Field(..., description="Chunk ID ({article_id}_{chunk_index})")
    article_id: str = Field(..., description="Parent article ID (e.g. KB-001)")
    title: str = Field(..., description="Article title")
    category: str = Field(..., description="Article category")
    chunk_index: int = Field(..., description="Chunk sequence index")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Cosine similarity relevance score in [0.0, 1.0]")
    content_hash: str = Field(..., description="Chunk content hash")
    tags: list[str] = Field(default_factory=list, description="Article tags")


class KBRetriever:
    """
    Retrieves and ranks relevant KB chunks for user support queries.
    """

    def __init__(
        self,
        vector_store: KBVectorStore,
        embedder: BaseEmbedder,
        relevance_threshold: float = 0.35,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.relevance_threshold = relevance_threshold

    @classmethod
    def create_default(cls) -> KBRetriever:
        """Factory creating retriever from active settings and providers."""
        settings = get_settings()
        _name, _llm, embedder = get_providers(settings)
        
        backend_root = Path(__file__).resolve().parent.parent.parent.parent
        db_path = backend_root / "data" / "lancedb"
        store = KBVectorStore(db_path=db_path)
        
        return cls(
            vector_store=store,
            embedder=embedder,
            relevance_threshold=settings.app.relevance_threshold,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Embed the user query and retrieve top_k most similar chunks.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_vector = self.embedder.embed_query(cleaned_query)
        hits = self.vector_store.search(query_vector=query_vector, top_k=top_k)

        chunks: list[RetrievedChunk] = []
        for hit in hits:
            chunks.append(
                RetrievedChunk(
                    id=hit["id"],
                    article_id=hit["article_id"],
                    title=hit["title"],
                    category=hit["category"],
                    chunk_index=hit["chunk_index"],
                    text=hit["text"],
                    score=hit["score"],
                    content_hash=hit["content_hash"],
                    tags=hit.get("tags", []),
                )
            )

        return chunks

    def has_kb_gap(self, chunks: list[RetrievedChunk]) -> bool:
        """
        Return True if no chunks retrieved or the best similarity score is strictly
        below relevance_threshold. This is the primary signal for Resolver kb_gap=True.
        """
        if not chunks:
            return True
        best_score = max(c.score for c in chunks)
        return best_score < self.relevance_threshold

    def best_score(self, chunks: list[RetrievedChunk]) -> float:
        """Return the highest score among retrieved chunks (or 0.0 if empty)."""
        if not chunks:
            return 0.0
        return max(c.score for c in chunks)

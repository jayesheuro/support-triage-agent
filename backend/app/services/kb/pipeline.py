"""
app/services/kb/pipeline.py

Responsibility: End-to-end Knowledge Base ingestion coordinator.
Orchestrates markdown loading, frontmatter validation, article-aware chunking,
embedding via BaseEmbedder, and idempotent LanceDB storage. Tracks execution metrics
and returns detailed IngestionSummary reports.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.kb.chunkers import ArticleAwareChunker, KBChunk
from app.services.kb.loaders import KBArticle, load_article_file, load_kb_directory
from app.services.kb.vector_store import KBVectorStore
from app.services.providers.base import BaseEmbedder
from app.services.providers.factory import get_providers

logger = get_logger(__name__)


class IngestionSummary(BaseModel):
    """Execution report from a knowledge base ingestion or re-index run."""

    total_articles: int = Field(..., description="Total articles evaluated")
    indexed_articles: int = Field(..., description="Number of new or updated articles embedded and stored")
    skipped_articles: int = Field(..., description="Unchanged articles skipped due to content hash match")
    total_chunks: int = Field(..., description="Total chunks currently stored in vector store")
    duration_sec: float = Field(..., description="Wall-clock duration in seconds")


class KBPipeline:
    """
    Orchestrates the ingestion lifecycle for the ShopLite Knowledge Base.
    """

    def __init__(
        self,
        vector_store: KBVectorStore,
        embedder: BaseEmbedder,
        chunker: ArticleAwareChunker | None = None,
        model_name: str = "nomic-embed-text",
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker or ArticleAwareChunker()
        self.model_name = model_name

    @classmethod
    def create_default(cls, force_recreate: bool = False) -> KBPipeline:
        """
        Factory method instantiating KBPipeline with active settings and providers.
        """
        settings = get_settings()
        _name, _llm, embedder = get_providers(settings)

        backend_root = Path(__file__).resolve().parent.parent.parent.parent
        db_path = backend_root / "data" / "lancedb"

        embed_cfg = settings.provider.embedder
        store = KBVectorStore(db_path=db_path)
        store.ensure_table(
            model_name=embed_cfg.model,
            dimension=embedder.dimension,
            force_recreate=force_recreate,
        )
        return cls(
            vector_store=store,
            embedder=embedder,
            model_name=embed_cfg.model,
        )

    def ingest_articles(
        self,
        articles: list[KBArticle],
        force: bool = False,
    ) -> IngestionSummary:
        """
        Idempotently ingest a list of KBArticle objects.
        Articles with identical content hashes are skipped unless force=True.
        """
        start_time = time.perf_counter()

        # Ensure table is ready without wiping existing data
        self.vector_store.ensure_table(
            model_name=self.model_name,
            dimension=self.embedder.dimension,
            force_recreate=False,
        )

        existing_hashes = {} if force else self.vector_store.get_indexed_article_hashes()

        indexed_count = 0
        skipped_count = 0

        for article in articles:
            # Check idempotency
            if not force and existing_hashes.get(article.id) == article.content_hash:
                logger.debug("Skipping unchanged article '%s' (hash: %s...)", article.id, article.content_hash[:8])
                skipped_count += 1
                continue

            logger.info("Indexing article '%s': '%s'", article.id, article.title)
            chunks: list[KBChunk] = self.chunker.chunk_article(article)

            if chunks:
                texts = [c.text for c in chunks]
                vectors = self.embedder.embed_documents(texts)
                self.vector_store.upsert_article(
                    article_id=article.id,
                    title=article.title,
                    category=article.category,
                    tags=article.tags,
                    article_content_hash=article.content_hash,
                    chunks=chunks,
                    vectors=vectors,
                )
            indexed_count += 1

        duration = round(time.perf_counter() - start_time, 2)
        total_chunks = self.vector_store.count_chunks()

        summary = IngestionSummary(
            total_articles=len(articles),
            indexed_articles=indexed_count,
            skipped_articles=skipped_count,
            total_chunks=total_chunks,
            duration_sec=duration,
        )
        logger.info(
            "Ingestion complete: %d indexed, %d skipped, %d total chunks in %.2f s",
            indexed_count,
            skipped_count,
            total_chunks,
            duration,
        )
        return summary

    def ingest_directory(
        self,
        dir_path: Path,
        force: bool = False,
    ) -> IngestionSummary:
        """Load and ingest all markdown articles from a directory."""
        articles = load_kb_directory(dir_path)
        return self.ingest_articles(articles, force=force)

    def ingest_file(
        self,
        file_path: Path,
        force: bool = False,
    ) -> IngestionSummary:
        """Load and ingest a single article file."""
        article = load_article_file(file_path)
        return self.ingest_articles([article], force=force)

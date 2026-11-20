"""
app/api/routes/kb.py

Responsibility: Admin and search API for the ShopLite Knowledge Base.
Exposes endpoints to list indexed articles, upload new markdown articles, delete
articles, trigger idempotent re-indexing, and test retrieval with relevance scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile
from pydantic import BaseModel, Field

from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.services.kb.loaders import KBArticle, parse_article_text
from app.services.kb.pipeline import IngestionSummary, KBPipeline
from app.services.kb.retriever import KBRetriever, RetrievedChunk

logger = get_logger(__name__)
router = APIRouter(prefix="/kb", tags=["knowledge-base"])

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_KB_DIR = _BACKEND_ROOT / "data" / "kb"


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class KBArticleResponse(BaseModel):
    """Metadata summary of an indexed Knowledge Base article."""

    id: str
    title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    chunk_count: int
    content_hash: str
    last_indexed: str


class KBUploadResponse(BaseModel):
    """Response returned upon successfully uploading and indexing an article."""

    status: str = "indexed"
    article_id: str
    title: str
    category: str
    chunks_created: int
    content_hash: str


class KBDeleteResponse(BaseModel):
    """Response returned upon deleting an article and its vector embeddings."""

    status: str = "deleted"
    article_id: str
    message: str


class KBReindexResponse(BaseModel):
    """Summary metrics of a knowledge base re-indexing operation."""

    total_articles: int
    indexed_articles: int
    skipped_articles: int
    total_chunks: int
    duration_sec: float


class KBSearchResultItem(BaseModel):
    """Single search result chunk with relevance score."""

    id: str
    article_id: str
    title: str
    category: str
    chunk_index: int
    text: str
    score: float
    content_hash: str
    tags: list[str] = Field(default_factory=list)


class KBSearchResponse(BaseModel):
    """Search test response including relevance threshold and gap indicators."""

    query: str
    relevance_threshold: float
    has_kb_gap: bool
    best_score: float
    results: list[KBSearchResultItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/articles", response_model=list[KBArticleResponse])
def list_articles() -> list[KBArticleResponse]:
    """
    List all currently indexed articles in the Knowledge Base with chunk counts.
    """
    pipeline = KBPipeline.create_default()
    summaries = pipeline.vector_store.list_articles()
    return [
        KBArticleResponse(
            id=s.id,
            title=s.title,
            category=s.category,
            tags=s.tags,
            chunk_count=s.chunk_count,
            content_hash=s.content_hash,
            last_indexed=s.last_indexed,
        )
        for s in summaries
    ]


@router.post("/articles", response_model=KBUploadResponse)
async def upload_article(
    file: Annotated[UploadFile, File(description="Markdown (.md) file with YAML frontmatter")],
) -> KBUploadResponse:
    """
    Upload a new .md help article with YAML frontmatter.
    Validates frontmatter, persists to data/kb/, and idempotently embeds into LanceDB.
    """
    if not file.filename or not file.filename.endswith(".md"):
        raise ValidationError("Only markdown files (.md) with YAML frontmatter are supported.")

    content_bytes = await file.read()
    try:
        raw_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"File '{file.filename}' could not be decoded as UTF-8 text.") from exc

    # Parse and validate frontmatter
    article: KBArticle = parse_article_text(raw_text, source_name=file.filename)

    pipeline = KBPipeline.create_default()

    # Save the file to data/kb/{article.id}.md to keep filesystem synchronized
    _KB_DIR.mkdir(parents=True, exist_ok=True)
    target_path = _KB_DIR / f"{article.id}.md"
    target_path.write_text(raw_text, encoding="utf-8")
    article.filepath = target_path

    # Ingest the parsed article
    _ = pipeline.ingest_articles([article], force=True)

    # Retrieve updated summary to get chunk count
    articles = pipeline.vector_store.list_articles()
    match = next((a for a in articles if a.id == article.id), None)
    chunks_created = match.chunk_count if match else 0

    return KBUploadResponse(
        status="indexed",
        article_id=article.id,
        title=article.title,
        category=article.category,
        chunks_created=chunks_created,
        content_hash=article.content_hash,
    )


@router.delete("/articles/{article_id}", response_model=KBDeleteResponse)
def delete_article(article_id: str) -> KBDeleteResponse:
    """
    Remove an article and all its vector chunks from LanceDB and disk.
    """
    pipeline = KBPipeline.create_default()
    deleted = pipeline.vector_store.delete_article(article_id)

    # Also delete corresponding file if on disk
    file_path = _KB_DIR / f"{article_id}.md"
    file_removed = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_removed = True
        except OSError as exc:
            logger.warning("Could not delete file %s: %s", file_path, exc)

    if not deleted and not file_removed:
        raise NotFoundError(f"Article '{article_id}' was not found in the Knowledge Base.")

    return KBDeleteResponse(
        status="deleted",
        article_id=article_id,
        message=f"Article '{article_id}' and all associated chunks were successfully deleted.",
    )


@router.post("/reindex", response_model=KBReindexResponse)
def reindex_kb(
    force: Annotated[bool, Query(description="Force re-embedding of unchanged articles")] = False,
) -> KBReindexResponse:
    """
    Trigger full idempotent re-indexing of all articles in data/kb/.
    Unchanged articles are skipped unless force=true.
    """
    pipeline = KBPipeline.create_default(force_recreate=force)
    summary: IngestionSummary = pipeline.ingest_directory(_KB_DIR, force=force)
    return KBReindexResponse(
        total_articles=summary.total_articles,
        indexed_articles=summary.indexed_articles,
        skipped_articles=summary.skipped_articles,
        total_chunks=summary.total_chunks,
        duration_sec=summary.duration_sec,
    )


@router.get("/search", response_model=KBSearchResponse)
def search_kb(
    q: Annotated[str, Query(description="Search query or customer question")],
    top_k: Annotated[int, Query(description="Maximum chunks to retrieve", ge=1, le=10)] = 5,
) -> KBSearchResponse:
    """
    Debug / search-test endpoint for knowledge base retrieval.
    Computes query embedding, searches nearest chunks in LanceDB, and evaluates relevance.
    """
    retriever = KBRetriever.create_default()
    chunks: list[RetrievedChunk] = retriever.retrieve(query=q, top_k=top_k)

    has_gap = retriever.has_kb_gap(chunks)
    best_score = retriever.best_score(chunks)

    results = [
        KBSearchResultItem(
            id=c.id,
            article_id=c.article_id,
            title=c.title,
            category=c.category,
            chunk_index=c.chunk_index,
            text=c.text,
            score=c.score,
            content_hash=c.content_hash,
            tags=c.tags,
        )
        for c in chunks
    ]

    return KBSearchResponse(
        query=q,
        relevance_threshold=retriever.relevance_threshold,
        has_kb_gap=has_gap,
        best_score=best_score,
        results=results,
    )

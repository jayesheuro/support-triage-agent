"""
app/services/kb/__init__.py

Knowledge Base services package: loading, chunking, vector storage, ingestion,
and retrieval for ShopLite support articles.
"""

from app.services.kb.chunkers import ArticleAwareChunker, KBChunk
from app.services.kb.loaders import KBArticle, load_article_file, load_kb_directory, parse_article_text
from app.services.kb.pipeline import IngestionSummary, KBPipeline
from app.services.kb.retriever import KBRetriever, RetrievedChunk
from app.services.kb.vector_store import ArticleSummary, KBVectorStore

__all__ = [
    "ArticleAwareChunker",
    "ArticleSummary",
    "IngestionSummary",
    "KBArticle",
    "KBChunk",
    "KBPipeline",
    "KBRetriever",
    "KBVectorStore",
    "RetrievedChunk",
    "load_article_file",
    "load_kb_directory",
    "parse_article_text",
]

"""
app/services/kb/vector_store.py

Responsibility: LanceDB vector store interface for ShopLite Knowledge Base chunks.
Stores embedding vectors, metadata, and content hashes. Enforces dimension and
embedding-model consistency, refuses to mix dimensions, and provides idempotent
article-level upsert and deletion capabilities.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.core.errors import KBDimensionMismatchError, KBError
from app.core.logging import get_logger
from app.services.kb.chunkers import KBChunk

if TYPE_CHECKING:
    import lancedb
    import pyarrow as pa

logger = get_logger(__name__)

DEFAULT_TABLE_NAME = "kb_chunks"
METADATA_FILENAME = "kb_metadata.json"


class ArticleSummary(BaseModel):
    """Article summary returned by vector store list operations."""

    id: str
    title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    chunk_count: int
    content_hash: str
    last_indexed: str


def _get_schema(dimension: int) -> pa.Schema:
    """Build the canonical PyArrow schema for KB chunks in LanceDB."""
    import pyarrow as pa

    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("article_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("category", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("content_hash", pa.string()),
        pa.field("article_content_hash", pa.string()),
        pa.field("text", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("vector", pa.list_(pa.float32(), dimension)),
    ])


class KBVectorStore:
    """
    LanceDB-backed vector database for ShopLite help articles.
    """

    def __init__(
        self,
        db_path: Path,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self._db_path_str = str(db_path)
        self.metadata_file = self.db_path / METADATA_FILENAME
        self._table = None

    def _get_db(self) -> lancedb.DBConnection:
        import lancedb

        self.db_path.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(self._db_path_str)

    def _read_metadata(self) -> dict[str, Any] | None:
        if not self.metadata_file.exists():
            return None
        try:
            return json.loads(self.metadata_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse vector store metadata file: %s", exc)
            return None

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def ensure_table(self, model_name: str, dimension: int, force_recreate: bool = False) -> Any:
        """
        Verify existing table or create a new one.
        Refuses to mix dimensions or model names, raising KBDimensionMismatchError.
        """
        db = self._get_db()
        meta = self._read_metadata()

        if force_recreate:
            if self.table_name in db.list_tables():
                db.drop_table(self.table_name)
            meta = None

        if meta is not None:
            stored_dim = meta.get("dimension")
            stored_model = meta.get("embedding_model")
            if stored_dim != dimension or stored_model != model_name:
                raise KBDimensionMismatchError(
                    f"Vector store dimension/model mismatch: index was built with "
                    f"model='{stored_model}' (dim={stored_dim}), but active embedder requires "
                    f"model='{model_name}' (dim={dimension}). "
                    f"Please re-index the knowledge base via /api/kb/reindex or seed_kb.py."
                )

        if self.table_name in db.list_tables():
            self._table = db.open_table(self.table_name)
            return self._table

        # Table does not exist, create it with schema
        schema = _get_schema(dimension)
        self._table = db.create_table(self.table_name, schema=schema)

        # Persist metadata
        meta = {
            "embedding_model": model_name,
            "dimension": dimension,
            "table_name": self.table_name,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "articles": {},
        }
        self._write_metadata(meta)
        return self._table

    def get_table(self) -> Any:
        """Return the opened table or open it if it exists."""
        if self._table is not None:
            return self._table
        db = self._get_db()
        if self.table_name in db.list_tables():
            self._table = db.open_table(self.table_name)
            return self._table
        raise KBError(f"Knowledge base table '{self.table_name}' does not exist. Run seed_kb.py first.")

    def get_indexed_article_hashes(self) -> dict[str, str]:
        """
        Return mapping of article_id -> article_content_hash from metadata.
        """
        meta = self._read_metadata()
        if not meta or "articles" not in meta:
            return {}
        return {k: v.get("content_hash", "") for k, v in meta["articles"].items()}

    def upsert_article(
        self,
        article_id: str,
        title: str,
        category: str,
        tags: list[str],
        article_content_hash: str,
        chunks: list[KBChunk],
        vectors: list[list[float]],
    ) -> int:
        """
        Replace all chunks for article_id with new chunks. Idempotent replacement.
        """
        table = self.get_table()

        # 1. Remove existing chunks for this article if present
        try:
            # Escape single quotes in article_id
            safe_id = article_id.replace("'", "''")
            table.delete(f"article_id = '{safe_id}'")
        except Exception as exc:
            logger.warning("Could not delete prior chunks for article %s: %s", article_id, exc)

        # 2. Build rows to insert
        records: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            records.append({
                "id": chunk.id,
                "article_id": chunk.article_id,
                "title": chunk.title,
                "category": chunk.category,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "article_content_hash": article_content_hash,
                "text": chunk.text,
                "tags": chunk.tags,
                "vector": vector,
            })

        if records:
            table.add(records)

        # 3. Update metadata
        meta = self._read_metadata() or {}
        articles = meta.setdefault("articles", {})
        now_iso = datetime.now(UTC).isoformat()
        articles[article_id] = {
            "title": title,
            "category": category,
            "tags": tags,
            "content_hash": article_content_hash,
            "chunk_count": len(records),
            "last_indexed": now_iso,
        }
        meta["updated_at"] = now_iso
        self._write_metadata(meta)

        return len(records)

    def delete_article(self, article_id: str) -> bool:
        """
        Delete article and all its chunks from LanceDB and metadata.
        """
        table = self.get_table()
        safe_id = article_id.replace("'", "''")

        try:
            table.delete(f"article_id = '{safe_id}'")
        except Exception as exc:
            raise KBError(f"Failed to delete chunks for '{article_id}' from LanceDB: {exc}") from exc

        meta = self._read_metadata()
        if meta and "articles" in meta and article_id in meta["articles"]:
            del meta["articles"][article_id]
            meta["updated_at"] = datetime.now(UTC).isoformat()
            self._write_metadata(meta)
            return True

        return False

    def list_articles(self) -> list[ArticleSummary]:
        """
        List all indexed articles with chunk count and last indexed timestamp.
        """
        meta = self._read_metadata()
        if not meta or "articles" not in meta:
            return []

        summaries: list[ArticleSummary] = []
        for art_id, info in sorted(meta["articles"].items()):
            summaries.append(
                ArticleSummary(
                    id=art_id,
                    title=info.get("title", ""),
                    category=info.get("category", ""),
                    tags=info.get("tags", []),
                    chunk_count=info.get("chunk_count", 0),
                    content_hash=info.get("content_hash", ""),
                    last_indexed=info.get("last_indexed", ""),
                )
            )
        return summaries

    def count_chunks(self) -> int:
        """Total chunks in the vector store."""
        try:
            return self.get_table().count_rows()
        except Exception:
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Cosine vector search returning hits with calculated similarity scores.
        """
        table = self.get_table()
        if table.count_rows() == 0:
            return []

        hits = table.search(query_vector).metric("cosine").limit(top_k).to_list()

        results: list[dict[str, Any]] = []
        for hit in hits:
            # LanceDB cosine distance: distance in [0, 2], similarity = 1 - distance
            dist = float(hit.get("_distance", 1.0))
            score = round(max(0.0, 1.0 - dist), 4)
            results.append({
                "id": hit["id"],
                "article_id": hit["article_id"],
                "title": hit["title"],
                "category": hit["category"],
                "chunk_index": hit["chunk_index"],
                "content_hash": hit["content_hash"],
                "text": hit["text"],
                "tags": hit.get("tags", []),
                "score": score,
            })

        return results

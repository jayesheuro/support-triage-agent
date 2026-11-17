"""
app/services/kb/loaders.py

Responsibility: Load and parse ShopLite Knowledge Base markdown articles with
YAML frontmatter ({id, title, category, tags}). Validates article schemas,
computes stable SHA-256 content hashes, and handles directory traversal.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.errors import ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Matches YAML frontmatter: --- ... --- at the start of a document
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class KBArticle(BaseModel):
    """Parsed Knowledge Base article with validated frontmatter metadata."""

    id: str = Field(..., description="Unique article identifier (e.g., KB-001)")
    title: str = Field(..., description="Human-readable article title")
    category: str = Field(..., description="Taxonomy category for the article")
    tags: list[str] = Field(default_factory=list, description="Descriptive search tags")
    content: str = Field(..., description="Markdown body text excluding frontmatter")
    content_hash: str = Field(..., description="SHA-256 hash of the complete raw document")
    filepath: Path | None = Field(default=None, description="Original source file path if loaded from disk")


def parse_article_text(raw_text: str, source_name: str = "<upload>") -> KBArticle:
    """
    Parse a raw markdown string with YAML frontmatter into a validated KBArticle.

    Raises:
        ValidationError: If frontmatter is missing, invalid YAML, or missing required fields.
    """
    if not raw_text or not raw_text.strip():
        raise ValidationError(f"Article '{source_name}' is empty.")

    match = FRONTMATTER_PATTERN.match(raw_text.strip())
    if not match:
        raise ValidationError(
            f"Article '{source_name}' is missing valid YAML frontmatter delimiters ('---')."
        )

    fm_raw, body = match.groups()

    try:
        data: dict[str, Any] = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as exc:
        raise ValidationError(f"Invalid YAML frontmatter in '{source_name}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValidationError(f"Frontmatter in '{source_name}' must be a YAML mapping.")

    article_id = str(data.get("id", "")).strip()
    title = str(data.get("title", "")).strip()
    category = str(data.get("category", "")).strip()
    raw_tags = data.get("tags", [])

    if not article_id:
        raise ValidationError(f"Article in '{source_name}' is missing mandatory 'id' field.")
    if not title:
        raise ValidationError(f"Article '{article_id}' is missing mandatory 'title' field.")
    if not category:
        raise ValidationError(f"Article '{article_id}' is missing mandatory 'category' field.")

    if isinstance(raw_tags, list):
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    elif isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    else:
        tags = []

    content_hash = hashlib.sha256(raw_text.strip().encode("utf-8")).hexdigest()

    return KBArticle(
        id=article_id,
        title=title,
        category=category,
        tags=tags,
        content=body.strip(),
        content_hash=content_hash,
    )


def load_article_file(path: Path) -> KBArticle:
    """Load a single markdown file from disk into a KBArticle."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Could not read article file '{path}': {exc}") from exc

    article = parse_article_text(text, source_name=path.name)
    article.filepath = path
    return article


def load_kb_directory(dir_path: Path) -> list[KBArticle]:
    """
    Load and validate all .md files in the specified directory.
    Sorts articles deterministically by article ID.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("KB directory '%s' does not exist.", dir_path)
        return []

    articles: list[KBArticle] = []
    seen_ids: set[str] = set()

    for file in sorted(dir_path.glob("*.md")):
        if file.name.startswith("."):
            continue
        try:
            article = load_article_file(file)
            if article.id in seen_ids:
                raise ValidationError(f"Duplicate article ID '{article.id}' found in '{file.name}'.")
            seen_ids.add(article.id)
            articles.append(article)
        except ValidationError as exc:
            logger.error("Skipping malformed KB file %s: %s", file.name, exc)
            raise

    return sorted(articles, key=lambda a: a.id)

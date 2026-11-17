"""
app/services/kb/chunkers.py

Responsibility: Article-aware chunking for ShopLite help articles.
Splits documents along markdown headings (#, ##, ###), preserves hierarchical
context breadcrumbs, and applies paragraph-aware sub-chunking with small overlap
for long sections. Computes deterministic SHA-256 content hashes per chunk.
"""

from __future__ import annotations

import hashlib
import re
from pydantic import BaseModel, Field

from app.services.kb.loaders import KBArticle


class KBChunk(BaseModel):
    """A semantic chunk derived from a KBArticle, ready for vector indexing."""

    id: str = Field(..., description="Unique chunk id: {article_id}_{chunk_index}")
    article_id: str = Field(..., description="ID of parent article")
    title: str = Field(..., description="Parent article title")
    category: str = Field(..., description="Parent article category")
    chunk_index: int = Field(..., description="0-based sequence index within article")
    content_hash: str = Field(..., description="SHA-256 hash of chunk text")
    text: str = Field(..., description="Chunk content including contextual heading breadcrumb")
    tags: list[str] = Field(default_factory=list, description="Inherited search tags")


# Pattern to identify Markdown headings (h1 through h4)
HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


class ArticleAwareChunker:
    """
    Splits help articles based on Markdown heading structure, keeping sections
    semantically coherent. Sub-splits sections that exceed max_chars (~300-500 words)
    with a configurable sliding window overlap.
    """

    def __init__(
        self,
        max_chars: int = 1500,       # ~300-400 words
        min_chars: int = 150,        # minimum size to avoid tiny orphan chunks
        overlap_chars: int = 200,    # ~40-50 words overlap
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    def chunk_article(self, article: KBArticle) -> list[KBChunk]:
        """
        Split a KBArticle into a sequence of contextual KBChunk objects.
        """
        raw_sections = self._split_by_headings(article.content, default_title=article.title)
        
        # Merge very small sections or sub-split very large sections
        processed_texts: list[str] = []
        current_buffer = ""

        for heading_prefix, section_body in raw_sections:
            section_full = f"{heading_prefix}\n\n{section_body}".strip() if heading_prefix else section_body.strip()
            if not section_full:
                continue

            # If section itself exceeds max_chars, split on paragraph boundaries with overlap
            if len(section_full) > self.max_chars:
                # Flush existing buffer first
                if current_buffer:
                    processed_texts.append(current_buffer.strip())
                    current_buffer = ""

                sub_chunks = self._split_long_section(heading_prefix, section_body)
                processed_texts.extend(sub_chunks)
            elif len(current_buffer) + len(section_full) + 2 <= self.max_chars:
                # Append to current buffer
                if current_buffer:
                    current_buffer += "\n\n" + section_full
                else:
                    current_buffer = section_full
            else:
                if current_buffer:
                    processed_texts.append(current_buffer.strip())
                current_buffer = section_full

        if current_buffer:
            processed_texts.append(current_buffer.strip())

        # If empty (e.g. article had no body), fall back to title
        if not processed_texts:
            fallback = f"# {article.title}\n\n{article.content}".strip()
            processed_texts = [fallback]

        chunks: list[KBChunk] = []
        for idx, text in enumerate(processed_texts):
            chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(
                KBChunk(
                    id=f"{article.id}_{idx}",
                    article_id=article.id,
                    title=article.title,
                    category=article.category,
                    chunk_index=idx,
                    content_hash=chunk_hash,
                    text=text,
                    tags=article.tags,
                )
            )

        return chunks

    def _split_by_headings(
        self, markdown_text: str, default_title: str
    ) -> list[tuple[str, str]]:
        """
        Split markdown text by # and ## headings.
        Returns list of tuples: (breadcrumb_header, section_text)
        """
        matches = list(HEADING_PATTERN.finditer(markdown_text))
        if not matches:
            return [(f"# {default_title}", markdown_text.strip())]

        sections: list[tuple[str, str]] = []
        
        # Preamble before first heading if any
        if matches[0].start() > 0:
            preamble = markdown_text[: matches[0].start()].strip()
            if preamble:
                sections.append((f"# {default_title}", preamble))

        current_h1 = default_title
        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
            body = markdown_text[start_pos:end_pos].strip()

            if level == 1:
                current_h1 = heading_text
                breadcrumb = f"# {current_h1}"
            else:
                breadcrumb = f"# {current_h1} > {match.group(1)} {heading_text}"

            sections.append((breadcrumb, body))

        return sections

    def _split_long_section(self, header: str, body: str) -> list[str]:
        """
        Split a single long section into overlapping paragraph chunks,
        retaining the header breadcrumb on each sub-chunk.
        """
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            return [header.strip()]

        chunks: list[str] = []
        current_paragraphs: list[str] = []
        current_len = len(header) + 2

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len + 2 > self.max_chars and current_paragraphs:
                # Emit current chunk
                combined = f"{header}\n\n" + "\n\n".join(current_paragraphs)
                chunks.append(combined)

                # Overlap: keep the last paragraph if possible
                overlap_p = current_paragraphs[-1] if len(current_paragraphs[-1]) <= self.overlap_chars else ""
                current_paragraphs = [overlap_p, p] if overlap_p else [p]
                current_len = len(header) + 2 + sum(len(x) + 2 for x in current_paragraphs)
            else:
                current_paragraphs.append(p)
                current_len += p_len + 2

        if current_paragraphs:
            combined = f"{header}\n\n" + "\n\n".join(current_paragraphs)
            chunks.append(combined)

        return chunks

"""
app/agents/nodes/resolver.py

Responsibility: Resolver agent node.
Retrieves relevant Knowledge Base articles for the ticket, determines if an abstention
condition (kb_gap) is present, and drafts grounded customer replies with strict [KB-xxx] citations.
Validates citations against retrieved articles and retries if hallucinated article IDs are generated.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.agents.state import AgentState
from app.core.logging import get_logger
from app.core.prompts import load_prompt
from app.db.enums import TicketStatus
from app.services.providers.base import BaseLLM
from app.services.providers.factory import get_providers

if TYPE_CHECKING:
    from app.services.kb.retriever import KBRetriever

logger = get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[(KB-\d{3})\]", re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    """Extract and normalize all [KB-xxx] citations found in text."""
    matches = _CITATION_PATTERN.findall(text)
    return sorted(list({m.upper() for m in matches}))


def resolver_node(
    state: AgentState,
    llm: BaseLLM | None = None,
    retriever: KBRetriever | None = None,
) -> dict[str, Any]:
    """
    Execute KB retrieval and response drafting.
    Abstains when best similarity score falls below relevance threshold (kb_gap=True).
    Validates cited article IDs against retrieved set.
    """
    ticket_id = state.get("ticket_id", "unknown")
    llm_calls_used = state.get("llm_calls_used", 0)
    max_llm_calls = state.get("max_llm_calls", 6)

    # 1. Budget enforcement check
    if llm_calls_used >= max_llm_calls:
        logger.error(
            "LLM call budget exceeded before resolver node",
            extra={"ticket_id": ticket_id, "used": llm_calls_used, "max": max_llm_calls},
        )
        return {
            "status": "error",
            "error": f"LLM call budget exceeded ({llm_calls_used}/{max_llm_calls})",
        }

    # 2. Resolve retriever and retrieve candidate chunks
    if retriever is None:
        from app.services.kb.retriever import KBRetriever

        retriever = KBRetriever.create_default()

    subject = state.get("subject", "")
    body = state.get("body", "")
    query = f"{subject}\n{body}".strip()

    chunks = retriever.retrieve(query=query, top_k=2)
    retrieved_data = [c.model_dump() for c in chunks]

    # 3. Abstention Check (KB Gap)
    if retriever.has_kb_gap(chunks):
        best_score = retriever.best_score(chunks)
        logger.info(
            "KB gap detected; abstaining from draft generation and routing to pending_review",
            extra={
                "ticket_id": ticket_id,
                "best_score": best_score,
                "threshold": retriever.relevance_threshold,
                "chunks_found": len(chunks),
            },
        )
        return {
            "kb_gap": True,
            "retrieved_chunks": retrieved_data,
            "draft_body": None,
            "citations": [],
            "status": TicketStatus.PENDING_REVIEW.value,
        }

    # 4. Resolve active LLM provider
    if llm is None:
        _, llm, _ = get_providers()

    # 5. Format KB excerpts and prompt
    valid_article_ids = {c.article_id.upper() for c in chunks}
    kb_parts: list[str] = []
    for c in chunks:
        kb_parts.append(f"[{c.article_id.upper()}] {c.title}\n{c.text}\n")
    kb_context = "\n---\n".join(kb_parts)

    prompt_template = load_prompt("resolver")
    prompt_text = prompt_template.format(
        kb_context=kb_context,
        customer_name=state.get("customer_name", "Customer"),
        category=state.get("category", "general"),
        priority=state.get("priority", "P2"),
        ticket_subject=subject,
        ticket_body=body,
    )

    # Incorporate Critic revision feedback if this is a revision cycle
    violations = state.get("critic_violations", [])
    if violations:
        violation_lines = [
            f"- [{v.get('policy_id')}] ({v.get('severity', '').upper()}): {v.get('reason')} (Offending span: \"{v.get('quoted_span')}\")"
            for v in violations
        ]
        revision_feedback = (
            "\n\n=== COMPLIANCE CRITIC REVISION FEEDBACK ===\n"
            "Your previous draft was flagged by our compliance critic for violating store policies. "
            "You MUST rewrite the draft to eliminate all of the following violations while retaining "
            "empathetic tone and strict [KB-xxx] citations:\n"
            + "\n".join(violation_lines)
            + "\nEnsure your new draft complies with all policies.\n"
        )
        prompt_text += revision_feedback

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]

    # 6. Generate draft response
    response = llm.generate(messages, max_tokens=220)
    llm_calls_used += 1
    draft_text = response.text.strip()
    cited_ids = extract_citations(draft_text)

    # 7. Validate citations against retrieved article IDs
    hallucinated = set(cited_ids) - valid_article_ids
    if hallucinated and llm_calls_used < max_llm_calls:
        logger.warning(
            "Hallucinated citation detected in draft; retrying with explicit correction",
            extra={
                "ticket_id": ticket_id,
                "hallucinated": list(hallucinated),
                "valid": sorted(list(valid_article_ids)),
                "llm_calls_used": llm_calls_used,
            },
        )

        messages.append({"role": "assistant", "content": draft_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your draft cited {sorted(list(hallucinated))} which is NOT in the provided excerpts. "
                    f"You may ONLY cite article IDs that appear in the excerpts: {sorted(list(valid_article_ids))}. "
                    f"Please rewrite the reply strictly citing only valid article IDs."
                ),
            }
        )

        retry_response = llm.generate(messages, max_tokens=220)
        llm_calls_used += 1
        draft_text = retry_response.text.strip()
        cited_ids = extract_citations(draft_text)

    # Retain only citations that are strictly valid
    final_citations = sorted(list({cid for cid in cited_ids if cid in valid_article_ids}))

    logger.info(
        "Resolver draft produced successfully",
        extra={
            "ticket_id": ticket_id,
            "citations": final_citations,
            "llm_calls_used": llm_calls_used,
        },
    )

    draft_history = list(state.get("draft_history", []))
    new_version = len(draft_history) + 1
    draft_history.append(
        {
            "version": new_version,
            "body": draft_text,
            "citations": final_citations,
            "created_by_node": "resolver" if new_version == 1 else "resolver_revision",
        }
    )

    revisions_used = state.get("revisions_used", 0)
    if violations:
        revisions_used += 1

    return {
        "kb_gap": False,
        "retrieved_chunks": retrieved_data,
        "draft_body": draft_text,
        "citations": final_citations,
        "draft_history": draft_history,
        "revisions_used": revisions_used,
        "llm_calls_used": llm_calls_used,
        "status": TicketStatus.DRAFTED.value,
    }

"""
app/db/enums.py

Responsibility: Lifecycle enums and classification types for the support ticket domain.
"""

from enum import Enum


class TicketStatus(str, Enum):
    """
    Lifecycle status of a support ticket.
    Transitions:
    new -> triaging -> untriaged | pending_review | resolving -> drafted -> pending_review
    (auto_sent, sent, rejected are terminal / human approval states for Phase 4+).
    """

    NEW = "new"
    TRIAGING = "triaging"
    UNTRIAGED = "untriaged"
    RESOLVING = "resolving"
    DRAFTED = "drafted"
    PENDING_REVIEW = "pending_review"
    AUTO_SENT = "auto_sent"
    SENT = "sent"
    REJECTED = "rejected"


class TicketPriority(str, Enum):
    """Priority level mapped from taxonomy rules."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Sentiment(str, Enum):
    """Customer sentiment classification."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class ChurnRisk(str, Enum):
    """Customer churn risk assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketChannel(str, Enum):
    """Ingress channel for the ticket."""

    WEB = "web"
    EMAIL = "email"
    CHAT = "chat"
    API = "api"

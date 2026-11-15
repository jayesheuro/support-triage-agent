"""
tests/fakes.py

Responsibility: Fake provider implementations for unit testing. FakeLLM accepts
a scriptable response queue — including malformed-then-valid sequences to exercise
retry paths. FakeEmbedder returns deterministic fixed-dimension vectors.

Usage in tests:
    llm = FakeLLM(responses=["valid reply", '{"bad json": }', "valid after retry"])
    embedder = FakeEmbedder(dimension=768)
"""

from __future__ import annotations

from app.services.providers.base import BaseEmbedder, BaseLLM, LLMResponse


class FakeLLM(BaseLLM):
    """
    Scriptable fake LLM. Pops responses from a queue; raises on empty queue.
    Supply Exception instances in the queue to simulate failures.

    Example:
        llm = FakeLLM(["first response", ValueError("boom"), "recovered"])
    """

    def __init__(self, responses: list[str | Exception] | None = None) -> None:
        # NOTE: pass an explicit list (even empty []) to control the queue.
        # None means "use the single-item default ['OK']" for convenience.
        self._queue: list[str | Exception] = list(responses) if responses is not None else ["OK"]
        self.calls: list[list[dict[str, str]]] = []  # records all generate() calls

    def _generate_impl(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        self.calls.append(messages)
        if not self._queue:
            raise RuntimeError("FakeLLM response queue is empty")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item,
            model_name="fake-llm",
            prompt_tokens=10,
            completion_tokens=len(item.split()),
            total_tokens=10 + len(item.split()),
        )

    def push(self, response: str | Exception) -> None:
        """Add a response to the back of the queue."""
        self._queue.append(response)


class FakeEmbedder(BaseEmbedder):
    """
    Deterministic fake embedder. Returns a vector where each element equals
    the hash of the text modulo 1, scaled to [0, 1].
    """

    def __init__(self, dimension: int = 768) -> None:
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_documents_impl(self, texts: list[str]) -> list[list[float]]:
        return [self._make_vector(t) for t in texts]

    def _embed_query_impl(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._make_vector(text)

    def _make_vector(self, text: str) -> list[float]:
        """Produce a deterministic pseudo-random vector from text."""
        seed = hash(text)
        import random
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]

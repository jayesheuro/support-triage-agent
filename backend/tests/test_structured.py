"""
tests/test_structured.py

Unit tests for the structured output parsing and retry-with-error-feedback helper.
Uses FakeLLM exclusively — zero network or real LLM calls.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.agents.structured import call_structured, extract_json_str
from app.core.errors import StructuredOutputError
from tests.fakes import FakeLLM


class SampleModel(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class TestExtractJsonStr:
    def test_extract_from_code_fence(self) -> None:
        raw = "Prefix explanation\n```json\n{\"category\": \"order_status\", \"confidence\": 0.9}\n```\nPostfix text"
        extracted = extract_json_str(raw)
        assert extracted == '{"category": "order_status", "confidence": 0.9}'

    def test_extract_from_outermost_braces(self) -> None:
        raw = "Here is the result: {\"category\": \"billing\", \"confidence\": 0.8} - hope this helps!"
        extracted = extract_json_str(raw)
        assert extracted == '{"category": "billing", "confidence": 0.8}'

    def test_raw_json(self) -> None:
        raw = '{"category": "general", "confidence": 0.5}'
        assert extract_json_str(raw) == raw


class TestCallStructured:
    def test_happy_path_first_attempt(self) -> None:
        llm = FakeLLM(['{"category": "order_status", "confidence": 0.95, "tags": ["shipping"]}'])
        result, attempts = call_structured(
            llm=llm,
            prompt="Classify this",
            schema=SampleModel,
            return_metadata=True,
        )
        assert isinstance(result, SampleModel)
        assert result.category == "order_status"
        assert result.confidence == 0.95
        assert result.tags == ["shipping"]
        assert attempts == 1
        assert len(llm.calls) == 1

    def test_recovers_after_two_retries(self) -> None:
        """
        Attempt 1: Invalid JSON syntax (DecodeError)
        Attempt 2: Invalid Pydantic schema (ValidationError: confidence out of bounds)
        Attempt 3: Valid JSON adhering to schema
        """
        llm = FakeLLM(
            [
                "Sorry, here is the classification: {bad json syntax",
                '{"category": "order_status", "confidence": 2.5}',  # confidence > 1.0 fails
                '```json\n{"category": "order_status", "confidence": 0.88, "tags": ["delay"]}\n```',
            ]
        )

        result, attempts = call_structured(
            llm=llm,
            prompt="Classify ticket",
            schema=SampleModel,
            max_retries=2,
            return_metadata=True,
        )

        assert isinstance(result, SampleModel)
        assert result.category == "order_status"
        assert result.confidence == 0.88
        assert attempts == 3
        assert len(llm.calls) == 3

        # Assert conversational feedback was appended to message history
        last_call_messages = llm.calls[-1]
        assert len(last_call_messages) == 5  # original user, assistant bad 1, user feedback 1, assistant bad 2, user feedback 2
        assert "Your previous output failed validation:" in last_call_messages[-1]["content"]

    def test_exceeds_max_retries_raises_structured_output_error(self) -> None:
        """
        When max_retries is 2 (3 attempts total), all 3 fail.
        Assert StructuredOutputError is raised carrying attempt count.
        """
        llm = FakeLLM(
            [
                "broken 1",
                "broken 2",
                "broken 3",
            ]
        )

        with pytest.raises(StructuredOutputError) as exc_info:
            call_structured(
                llm=llm,
                prompt="Classify ticket",
                schema=SampleModel,
                max_retries=2,
            )

        err = exc_info.value
        assert err.attempts == 3
        assert "broken 3" in (err.raw_response or "")
        assert "Failed to produce valid SampleModel after 3 attempt(s)" in str(err)

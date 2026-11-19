"""
app/agents/structured.py

Responsibility: Generic structured-output parsing and retry-with-error-feedback loop.
Calls BaseLLM, extracts and parses JSON, validates against a Pydantic schema, and
re-prompts with validation error feedback on failure up to max_retries.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel, ValidationError

from app.core.errors import StructuredOutputError
from app.core.logging import get_logger
from app.services.providers.base import BaseLLM

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_str(text: str) -> str:
    """
    Extract a JSON string candidate from an LLM response.
    Handles markdown code blocks (```json ... ```) or finds the outermost JSON delimiters.
    """
    # 1. Match code fences: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    # 2. Outermost object { ... }
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1].strip()

    # 3. Outermost array [ ... ]
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return text[start_arr : end_arr + 1].strip()

    return text.strip()


@overload
def call_structured(
    llm: BaseLLM,
    prompt: str | list[dict[str, str]],
    schema: type[T],
    max_retries: int = 2,
    *,
    return_metadata: Literal[True],
    **kwargs: Any,
) -> tuple[T, int]: ...


@overload
def call_structured(
    llm: BaseLLM,
    prompt: str | list[dict[str, str]],
    schema: type[T],
    max_retries: int,
    return_metadata: Literal[True],
    **kwargs: Any,
) -> tuple[T, int]: ...


@overload
def call_structured(
    llm: BaseLLM,
    prompt: str | list[dict[str, str]],
    schema: type[T],
    max_retries: int = 2,
    return_metadata: Literal[False] = False,
    **kwargs: Any,
) -> T: ...


def call_structured(
    llm: BaseLLM,
    prompt: str | list[dict[str, str]],
    schema: type[T],
    max_retries: int = 2,
    return_metadata: bool = False,
    **kwargs: Any,
) -> T | tuple[T, int]:
    """
    Call an LLM and parse/validate its response into a Pydantic model.

    If JSON parsing or Pydantic validation fails, feeds the exact error back to the LLM
    in a subsequent prompt turn, up to max_retries times.

    Raises:
        StructuredOutputError: If output fails validation after max_retries attempts.
    """
    if isinstance(prompt, str):
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    else:
        messages = [dict(m) for m in prompt]

    total_attempts = 0
    last_error: str | None = None
    last_raw: str | None = None

    # max_retries retries means up to 1 + max_retries total calls
    max_attempts = 1 + max(0, max_retries)

    for attempt in range(1, max_attempts + 1):
        total_attempts = attempt
        response = llm.generate(messages, **kwargs)
        last_raw = response.text.strip()

        try:
            json_str = extract_json_str(last_raw)
            parsed_data = json.loads(json_str)
            if not isinstance(parsed_data, dict):
                raise ValueError(f"Expected a JSON object (dict), but got {type(parsed_data).__name__}")
            validated = schema.model_validate(parsed_data)

            if attempt > 1:
                logger.info(
                    "Structured output recovered after retry",
                    extra={"attempt": attempt, "schema": schema.__name__},
                )

            return (validated, total_attempts) if return_metadata else validated

        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Structured output validation failed; retrying with error feedback",
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error": last_error,
                    "schema": schema.__name__,
                },
            )

            if attempt < max_attempts:
                # Append error feedback to message history for conversational retry
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous output failed validation:\n{last_error}\n\n"
                            f"Please return ONLY valid JSON matching the required schema: {schema.model_json_schema()}"
                        ),
                    }
                )

    raise StructuredOutputError(
        message=f"Failed to produce valid {schema.__name__} after {total_attempts} attempt(s): {last_error}",
        attempts=total_attempts,
        last_error=last_error,
        raw_response=last_raw,
    )

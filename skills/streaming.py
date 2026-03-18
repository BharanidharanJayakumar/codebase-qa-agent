"""
Streaming LLM response utility.

Wraps litellm for streaming completions from Groq/OpenAI-compatible providers.
Used for real-time answer delivery in the QA pipeline.
"""
import json
import logging
import os
import time
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile")


async def stream_llm_response(
    system: str,
    user: str,
    model: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> AsyncGenerator[dict, None]:
    """
    Stream LLM response chunks with metadata.

    Yields dicts with:
        - type: "chunk" | "done" | "error"
        - content: the text chunk (for "chunk" type)
        - metadata: timing/token info (for "done" type)
    """
    import litellm

    model = model or DEFAULT_MODEL
    start_time = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_time is None:
                    first_token_time = time.time()

                full_content += delta.content
                chunk_count += 1

                yield {
                    "type": "chunk",
                    "content": delta.content,
                    "chunk_index": chunk_count,
                }

        end_time = time.time()
        yield {
            "type": "done",
            "full_content": full_content,
            "metadata": {
                "model": model,
                "total_time_ms": round((end_time - start_time) * 1000),
                "time_to_first_token_ms": (
                    round((first_token_time - start_time) * 1000)
                    if first_token_time
                    else None
                ),
                "chunk_count": chunk_count,
                "total_chars": len(full_content),
            },
        }

    except Exception as e:
        logger.error(f"Streaming LLM error: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "partial_content": full_content,
        }


async def stream_and_collect(
    system: str,
    user: str,
    model: str = "",
    max_tokens: int = 2048,
    on_chunk: callable = None,
) -> dict:
    """
    Stream LLM response and collect the full result.

    Optionally calls on_chunk(text) for each chunk received.
    Returns the complete response with timing metadata.
    """
    full_content = ""
    metadata = {}

    async for event in stream_llm_response(system, user, model, max_tokens):
        if event["type"] == "chunk":
            full_content += event["content"]
            if on_chunk:
                on_chunk(event["content"])
        elif event["type"] == "done":
            metadata = event["metadata"]
            full_content = event["full_content"]
        elif event["type"] == "error":
            return {
                "content": event.get("partial_content", ""),
                "error": event["error"],
                "metadata": {},
            }

    return {
        "content": full_content,
        "error": None,
        "metadata": metadata,
    }


def parse_json_from_stream(content: str, schema_class=None):
    """
    Parse structured JSON from a streamed LLM response.

    The LLM may wrap JSON in markdown code blocks or include extra text.
    This extracts and validates the JSON against an optional Pydantic schema.
    """
    import re

    # Try direct parse first
    try:
        data = json.loads(content)
        if schema_class:
            return schema_class.model_validate(data)
        return data
    except (json.JSONDecodeError, Exception):
        pass

    # Try extracting from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if schema_class:
                return schema_class.model_validate(data)
            return data
        except (json.JSONDecodeError, Exception):
            pass

    # Try finding JSON object in the content
    brace_start = content.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(content[brace_start : i + 1])
                        if schema_class:
                            return schema_class.model_validate(data)
                        return data
                    except (json.JSONDecodeError, Exception):
                        pass
                    break

    return None

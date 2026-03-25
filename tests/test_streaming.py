"""Tests for the streaming utility module."""
import json
from skills.streaming import parse_json_from_stream


class TestParseJsonFromStream:
    def test_direct_json(self):
        content = '{"answer": "hello", "confidence": "high"}'
        result = parse_json_from_stream(content)
        assert result["answer"] == "hello"
        assert result["confidence"] == "high"

    def test_json_in_code_block(self):
        content = 'Here is the answer:\n```json\n{"answer": "test"}\n```'
        result = parse_json_from_stream(content)
        assert result["answer"] == "test"

    def test_json_with_prefix_text(self):
        content = 'Let me analyze... {"answer": "found it", "files": ["a.py"]}'
        result = parse_json_from_stream(content)
        assert result["answer"] == "found it"

    def test_nested_json(self):
        content = '{"answer": "test", "citations": [{"file": "a.py", "start_line": 1}]}'
        result = parse_json_from_stream(content)
        assert len(result["citations"]) == 1
        assert result["citations"][0]["file"] == "a.py"

    def test_invalid_json(self):
        content = "This is not JSON at all"
        result = parse_json_from_stream(content)
        assert result is None

    def test_empty_content(self):
        result = parse_json_from_stream("")
        assert result is None

    def test_with_pydantic_schema(self):
        from pydantic import BaseModel, Field

        class SimpleAnswer(BaseModel):
            answer: str
            confidence: str = "medium"

        content = '{"answer": "works!", "confidence": "high"}'
        result = parse_json_from_stream(content, SimpleAnswer)
        assert isinstance(result, SimpleAnswer)
        assert result.answer == "works!"
        assert result.confidence == "high"

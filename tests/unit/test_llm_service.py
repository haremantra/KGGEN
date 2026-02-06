"""Tests for src/utils/llm.py — generate, retry, fallback, JSON parsing."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.utils.llm import LLMService, get_llm_service


@pytest.fixture
def llm_service(monkeypatch):
    """LLMService with mocked Anthropic/OpenAI clients."""
    mock_settings = MagicMock()
    mock_settings.anthropic_api_key = "test-key"
    mock_settings.openai_api_key = "test-key"
    mock_settings.primary_llm_provider = "anthropic"
    mock_settings.primary_llm_model = "claude-test"
    mock_settings.fallback_llm_provider = "openai"
    mock_settings.fallback_llm_model = "gpt-test"
    mock_settings.llm_max_tokens = 1024
    mock_settings.extraction_temperature = 0.0
    monkeypatch.setattr("src.utils.llm.get_settings", lambda: mock_settings)

    svc = LLMService.__new__(LLMService)
    svc._settings = mock_settings
    svc.primary_provider = "anthropic"
    svc.primary_model = "claude-test"
    svc.fallback_provider = "openai"
    svc.fallback_model = "gpt-test"

    # Mock Anthropic client
    mock_anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_anthropic.messages.create.return_value = mock_response
    svc._anthropic = mock_anthropic

    # Mock OpenAI client
    mock_openai = MagicMock()
    mock_oi_response = MagicMock()
    mock_oi_response.choices = [MagicMock(message=MagicMock(content="Fallback response"))]
    mock_openai.chat.completions.create.return_value = mock_oi_response
    svc._openai = mock_openai

    return svc


class TestGenerate:

    def test_returns_text(self, llm_service):
        text, model = llm_service.generate("system", "user")
        assert text == "Test response"
        assert model == "claude-test"

    def test_retry_on_api_error(self, llm_service):
        """Simulate 1 failure then success."""
        call_count = {"n": 0}
        original_create = llm_service._anthropic.messages.create

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("API error")
            return original_create(*args, **kwargs)

        llm_service._anthropic.messages.create.side_effect = side_effect
        # The retry decorator wraps _call_anthropic, so we need to test through generate
        # Since retry is on _call_anthropic, calling generate should retry
        text, model = llm_service.generate("system", "user")
        assert call_count["n"] >= 2

    def test_fallback_to_openai(self, llm_service):
        """Primary always fails, fallback succeeds."""
        llm_service._anthropic.messages.create.side_effect = Exception("Always fails")
        text, model = llm_service.generate("system", "user", use_fallback=True)
        assert text == "Fallback response"
        assert model == "gpt-test"

    def test_all_providers_fail(self, llm_service):
        """Both providers fail."""
        llm_service._anthropic.messages.create.side_effect = Exception("Fail")
        llm_service._openai.chat.completions.create.side_effect = Exception("Also fail")
        with pytest.raises(Exception):
            llm_service.generate("system", "user", use_fallback=True)


class TestParseJson:

    def test_parse_json_plain(self, llm_service):
        result = llm_service._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_code_block(self, llm_service):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = llm_service._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_generic_code_block(self, llm_service):
        text = 'Result:\n```\n[1, 2, 3]\n```'
        result = llm_service._parse_json(text)
        assert result == [1, 2, 3]

    def test_parse_json_malformed(self, llm_service):
        result = llm_service._parse_json("This is not JSON at all")
        assert result is None

    def test_parse_json_embedded_object(self, llm_service):
        # Note: _parse_json tries arrays before objects, so embedded arrays may be found first
        text = 'Here is the data: {"name": "ACME"} end'
        result = llm_service._parse_json(text)
        assert result == {"name": "ACME"}


class TestExtractMethods:

    def test_extract_entities_parses(self, llm_service):
        entities_json = json.dumps([
            {"name": "ACME Corp", "type": "Party", "confidence": "HIGH"},
        ])
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=entities_json)]
        llm_service._anthropic.messages.create.return_value = mock_resp

        entities = llm_service.extract_entities("contract text")
        assert len(entities) == 1
        assert entities[0]["name"] == "ACME Corp"

    def test_extract_entities_malformed_json(self, llm_service):
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Not valid JSON")]
        llm_service._anthropic.messages.create.return_value = mock_resp

        entities = llm_service.extract_entities("contract text")
        assert entities == []

    def test_select_canonical_returns_name(self, llm_service):
        resp_json = json.dumps({"canonical": "ACME Corporation", "aliases": ["ACME Corp"]})
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=resp_json)]
        llm_service._anthropic.messages.create.return_value = mock_resp

        canonical, aliases = llm_service.select_canonical(["ACME Corp", "ACME Corporation"])
        assert canonical == "ACME Corporation"

    def test_identify_duplicates_returns_list(self, llm_service):
        resp_json = json.dumps([["ACME Corp", "ACME Corporation"]])
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=resp_json)]
        llm_service._anthropic.messages.create.return_value = mock_resp

        groups = llm_service.identify_duplicates(["ACME Corp", "ACME Corporation", "Beta LLC"])
        assert isinstance(groups, list)
        assert len(groups) == 1


class TestAnswerQuery:

    def test_answer_query_includes_context(self, llm_service):
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ANSWER: Non-exclusive license\nCONFIDENCE: HIGH")]
        llm_service._anthropic.messages.create.return_value = mock_resp

        answer, confidence = llm_service.answer_query("What license?", "Context here")
        assert "Non-exclusive" in answer
        assert confidence == 0.9  # HIGH


class TestSingleton:

    def test_get_llm_service_singleton(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = ""
        mock_settings.openai_api_key = ""
        mock_settings.primary_llm_provider = "anthropic"
        mock_settings.primary_llm_model = "test"
        mock_settings.fallback_llm_provider = "openai"
        mock_settings.fallback_llm_model = "test"
        monkeypatch.setattr("src.utils.llm.get_settings", lambda: mock_settings)

        svc1 = get_llm_service()
        svc2 = get_llm_service()
        assert svc1 is svc2

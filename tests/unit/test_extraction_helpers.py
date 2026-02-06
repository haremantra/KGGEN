"""Tests for extraction helper functions — _normalize_name, _map_confidence, _chunk_text."""

import pytest
from src.extraction.extractor import (
    CONFIDENCE_MAP, DEFAULT_CONFIDENCE,
    _normalize_name, _map_confidence, _chunk_text,
)


class TestConfidenceMap:

    def test_low(self):
        assert CONFIDENCE_MAP["LOW"] == 0.4

    def test_medium(self):
        assert CONFIDENCE_MAP["MEDIUM"] == 0.7

    def test_high(self):
        assert CONFIDENCE_MAP["HIGH"] == 0.9

    def test_default(self):
        assert DEFAULT_CONFIDENCE == 0.7


class TestNormalizeName:

    def test_strip_the(self):
        assert _normalize_name("the Company") == "Company"

    def test_strip_a(self):
        assert _normalize_name("a License") == "License"

    def test_strip_an(self):
        assert _normalize_name("an Agreement") == "Agreement"

    def test_collapse_whitespace(self):
        assert _normalize_name("ACME  Software  Corp") == "ACME Software Corp"

    def test_strip_leading_trailing(self):
        assert _normalize_name("  text  ") == "text"

    def test_preserves_case(self):
        assert _normalize_name("The ACME Corp") == "ACME Corp"

    def test_empty_string(self):
        assert _normalize_name("") == ""

    def test_only_article(self):
        # "the " strips to "the", which doesn't start with "the " (with space)
        assert _normalize_name("the ") == "the"


class TestMapConfidence:

    def test_low_string(self):
        assert _map_confidence("LOW") == 0.4

    def test_medium_string(self):
        assert _map_confidence("MEDIUM") == 0.7

    def test_high_string(self):
        assert _map_confidence("HIGH") == 0.9

    def test_case_insensitive(self):
        assert _map_confidence("low") == 0.4
        assert _map_confidence("High") == 0.9

    def test_unknown_string(self):
        assert _map_confidence("UNKNOWN") == DEFAULT_CONFIDENCE

    def test_numeric_passthrough(self):
        assert _map_confidence(0.85) == 0.85

    def test_int_passthrough(self):
        assert _map_confidence(1) == 1.0

    def test_none_returns_default(self):
        assert _map_confidence(None) == DEFAULT_CONFIDENCE


class TestChunkText:

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = _chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_size_single_chunk(self):
        text = "x" * 100
        chunks = _chunk_text(text, chunk_size=100)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        text = "Word. " * 500  # ~3000 chars
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1

    def test_chunks_cover_all_text(self):
        text = "Hello world. This is a test. More text here. Even more words."
        chunks = _chunk_text(text, chunk_size=30, overlap=5)
        # All characters should appear in at least one chunk
        combined = "".join(chunks)
        for char_pos in range(len(text)):
            assert text[char_pos] in combined

    def test_overlap_present(self):
        text = "A" * 100 + ". " + "B" * 100
        chunks = _chunk_text(text, chunk_size=110, overlap=20)
        if len(chunks) >= 2:
            # Some overlap between consecutive chunks
            end_of_first = chunks[0][-20:]
            start_of_second = chunks[1][:20]
            # At least some overlap
            assert len(set(end_of_first) & set(start_of_second)) > 0

    def test_sentence_boundary_preferred(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = _chunk_text(text, chunk_size=40, overlap=5)
        # Should prefer breaking at ". "
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith(".") or chunk.endswith(". ")

    def test_custom_params(self):
        text = "x" * 500
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 5

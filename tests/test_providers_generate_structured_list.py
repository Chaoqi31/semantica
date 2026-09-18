"""Tests for generate_structured() list-return behaviour at the backend-provider level.

These tests exercise the concrete provider implementations in
``semantica.semantic_extract.providers`` directly, without going through the
public ``semantica.llms.*`` wrappers.  They focus on the two behaviours that
are at the core of issue #1270:

1. ``BaseProvider._parse_json()`` accepts and returns top-level JSON arrays.
2. ``HuggingFaceLLMProvider.generate_structured()`` correctly extracts a
   JSON array from the model's raw text output via the fallback boundary search.
"""

import json

import pytest

from semantica.semantic_extract.providers import BaseProvider, HuggingFaceLLMProvider
from semantica.utils.exceptions import ProcessingError


# ---------------------------------------------------------------------------
# BaseProvider._parse_json  (shared helper used by all API-based providers)
# ---------------------------------------------------------------------------

class _ConcreteProvider(BaseProvider):
    """Minimal concrete subclass so we can call _parse_json without mocking."""

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError


def test_parse_json_returns_dict_for_object():
    p = _ConcreteProvider()
    result = p._parse_json('{"key": "value"}')
    assert result == {"key": "value"}
    assert isinstance(result, dict)


def test_parse_json_returns_list_for_top_level_array():
    p = _ConcreteProvider()
    result = p._parse_json('[{"id": 1}, {"id": 2}]')
    assert result == [{"id": 1}, {"id": 2}]
    assert isinstance(result, list)


def test_parse_json_extracts_list_from_prose():
    """The fallback boundary search must find [...] even when surrounded by prose."""
    p = _ConcreteProvider()
    result = p._parse_json('Here is the result: [{"id": 1}] end.')
    assert result == [{"id": 1}]
    assert isinstance(result, list)


def test_parse_json_extracts_object_from_prose():
    p = _ConcreteProvider()
    result = p._parse_json('Sure! Here: {"key": "value"} done.')
    assert result == {"key": "value"}
    assert isinstance(result, dict)


def test_parse_json_strips_markdown_code_fence():
    p = _ConcreteProvider()
    result = p._parse_json('```json\n[{"id": 1}]\n```')
    assert result == [{"id": 1}]
    assert isinstance(result, list)


def test_parse_json_raises_on_no_json():
    p = _ConcreteProvider()
    with pytest.raises(ProcessingError, match="No valid JSON"):
        p._parse_json("no json here at all")


# ---------------------------------------------------------------------------
# HuggingFaceLLMProvider.generate_structured  (has its own fallback logic)
# ---------------------------------------------------------------------------

def _make_hf_provider(response_text: str) -> HuggingFaceLLMProvider:
    """Return a HuggingFaceLLMProvider whose generate() returns ``response_text``
    without loading any real model."""
    try:
        import torch  # noqa: F401 — needed by HuggingFaceLLMProvider.__init__
    except ImportError:
        pytest.skip("torch not installed; skipping HuggingFaceLLMProvider tests")

    provider = object.__new__(HuggingFaceLLMProvider)
    # Initialise only the attributes that generate_structured() needs.
    provider.model = None
    provider.tokenizer = None
    provider.device = "cpu"
    provider.config = {}

    import logging
    provider.logger = logging.getLogger("test_hf_provider")

    # Patch generate() to return fixed text without touching any model.
    provider.generate = lambda prompt, **kw: response_text
    return provider


def test_hf_generate_structured_returns_dict():
    provider = _make_hf_provider('{"key": "value"}')
    result = provider.generate_structured("prompt")
    assert result == {"key": "value"}
    assert isinstance(result, dict)


def test_hf_generate_structured_returns_list_via_primary_path():
    """When generate() returns a bare JSON array, json.loads() succeeds directly."""
    provider = _make_hf_provider('[{"id": 1}, {"id": 2}]')
    result = provider.generate_structured("prompt")
    assert result == [{"id": 1}, {"id": 2}]
    assert isinstance(result, list)


def test_hf_generate_structured_returns_list_via_fallback():
    """When json.loads() fails (trailing prose), the boundary search must still
    extract a top-level JSON array."""
    provider = _make_hf_provider('Here is the list: [{"id": 1}] done.')
    result = provider.generate_structured("prompt")
    assert result == [{"id": 1}]
    assert isinstance(result, list)


def test_hf_generate_structured_raises_when_no_json():
    provider = _make_hf_provider("no json here")
    with pytest.raises(ProcessingError, match="Failed to parse JSON from HuggingFace response"):
        provider.generate_structured("prompt")

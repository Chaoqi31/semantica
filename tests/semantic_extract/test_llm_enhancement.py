"""Regression tests for LLMExtraction enhancement response parsing.

These tests would ALL FAIL against the old stub implementation because
_parse_entity_response / _parse_relation_response used to ignore the
LLM response entirely. They pass after the fix.

Coverage:
  Entity enhancement
    1. LLM updates an existing entity's label
    2. LLM updates an existing entity's confidence
    3. LLM adds a brand-new entity not in the original list
    4. Empty LLM response preserves originals without data loss
    5. Malformed/unusable LLM response falls back gracefully
    6. enhanced_by / model metadata is always present
    7. Entities not mentioned by the LLM are preserved unchanged

  Relation enhancement
    8.  LLM updates an existing relation's predicate
    9.  LLM updates an existing relation's confidence
    10. LLM adds a brand-new relation not in the original list
    11. Empty LLM response preserves originals without data loss
    12. Malformed/unusable LLM response falls back gracefully
    13. enhanced_by / model metadata is always present on relations
    14. Relations not mentioned by the LLM are preserved unchanged

  Integration helpers (_parse_entity_response / _parse_relation_response)
    15. No duplicate entities when LLM repeats an existing entity
    16. No duplicate relations when LLM repeats an existing pair
"""

import pytest
from unittest.mock import MagicMock, patch

from semantica.semantic_extract.llm_extraction import LLMExtraction
from semantica.semantic_extract.schemas import EntitiesResponse, EntityOut, RelationsResponse, RelationOut
from semantica.semantic_extract.types import Entity, Relation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(text: str, label: str, confidence: float = 0.8) -> Entity:
    return Entity(
        text=text,
        label=label,
        start_char=0,
        end_char=len(text),
        confidence=confidence,
        metadata={},
    )


def _make_relation(
    subj_text: str, pred: str, obj_text: str, confidence: float = 0.8
) -> Relation:
    return Relation(
        subject=_make_entity(subj_text, "ORG"),
        predicate=pred,
        object=_make_entity(obj_text, "PERSON"),
        confidence=confidence,
        context="",
        metadata={},
    )


def _make_extractor(llm_response) -> LLMExtraction:
    """Return an LLMExtraction instance whose provider.generate_typed returns
    *llm_response* without hitting any real API."""
    extractor = LLMExtraction.__new__(LLMExtraction)
    extractor.provider_name = "openai"
    extractor.model = "gpt-4"
    extractor.temperature = None
    extractor.config = {}

    from semantica.utils.logging import get_logger
    extractor.logger = get_logger("test_llm_enhancement")

    from semantica.utils.progress_tracker import get_progress_tracker
    extractor.progress_tracker = get_progress_tracker()
    extractor.progress_tracker.enabled = False

    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True
    mock_provider.generate_typed.return_value = llm_response
    extractor.provider = mock_provider

    return extractor


# ---------------------------------------------------------------------------
# Entity enhancement tests
# ---------------------------------------------------------------------------

class TestEnhanceEntitiesLabelUpdate:
    """Test 1 – LLM corrects an existing entity's label."""

    def test_label_is_updated(self):
        original = [_make_entity("Apple Inc.", "PRODUCT")]  # wrong label

        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_entities("Apple Inc. is a company.", original)

        assert len(result) == 1
        assert result[0].text == "Apple Inc."
        assert result[0].label == "ORG", "LLM-corrected label must be applied"


class TestEnhanceEntitiesConfidenceUpdate:
    """Test 2 – LLM updates an existing entity's confidence score."""

    def test_confidence_is_updated(self):
        original = [_make_entity("Steve Jobs", "PERSON", confidence=0.5)]

        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Steve Jobs", label="PERSON", confidence=0.99),
        ])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_entities("Steve Jobs founded Apple.", original)

        assert result[0].confidence == pytest.approx(0.99)


class TestEnhanceEntitiesNewEntity:
    """Test 3 – LLM adds a new entity absent from the original list."""

    def test_new_entity_appended(self):
        original = [_make_entity("Apple Inc.", "ORG")]

        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
            EntityOut(text="Steve Jobs", label="PERSON", confidence=0.95),  # new
        ])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_entities(
            "Apple Inc. was founded by Steve Jobs.", original
        )

        texts = [e.text for e in result]
        assert "Steve Jobs" in texts, "New entity from LLM must be appended"
        assert len(result) == 2


class TestEnhanceEntitiesEmptyResponse:
    """Test 4 – Empty LLM response preserves the original list."""

    def test_empty_response_preserves_originals(self):
        original = [
            _make_entity("Apple Inc.", "ORG"),
            _make_entity("Tim Cook", "PERSON"),
        ]

        llm_resp = EntitiesResponse(entities=[])  # LLM returned nothing
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_entities("Apple Inc.", original)

        assert len(result) == 2
        texts = [e.text for e in result]
        assert "Apple Inc." in texts
        assert "Tim Cook" in texts


class TestEnhanceEntitiesMalformedResponse:
    """Test 5 – Provider raises; method falls back to original entities."""

    def test_fallback_on_provider_error(self):
        original = [_make_entity("Apple Inc.", "ORG")]

        extractor = LLMExtraction.__new__(LLMExtraction)
        extractor.provider_name = "openai"
        extractor.model = "gpt-4"
        extractor.temperature = None
        extractor.config = {}

        from semantica.utils.logging import get_logger
        extractor.logger = get_logger("test_llm_enhancement")
        from semantica.utils.progress_tracker import get_progress_tracker
        extractor.progress_tracker = get_progress_tracker()
        extractor.progress_tracker.enabled = False

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate_typed.side_effect = Exception("network timeout")
        extractor.provider = mock_provider

        result = extractor.enhance_entities("Apple Inc. is a company.", original)

        assert len(result) == 1
        assert result[0].text == "Apple Inc."


class TestEnhanceEntitiesMetadata:
    """Test 6 – enhanced_by and model metadata always present."""

    def test_metadata_present_on_updated_entity(self):
        original = [_make_entity("Apple Inc.", "PRODUCT")]

        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        assert result[0].metadata.get("enhanced_by") == "openai"
        assert result[0].metadata.get("model") == "gpt-4"

    def test_metadata_present_on_new_entity(self):
        original = [_make_entity("Apple Inc.", "ORG")]

        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
            EntityOut(text="Steve Jobs", label="PERSON", confidence=0.95),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        new_ent = next(e for e in result if e.text == "Steve Jobs")
        assert new_ent.metadata.get("enhanced_by") == "openai"
        assert new_ent.metadata.get("model") == "gpt-4"

    def test_metadata_present_on_untouched_entity(self):
        """Entities NOT returned by the LLM must still get the metadata stamp."""
        original = [
            _make_entity("Apple Inc.", "ORG"),
            _make_entity("Tim Cook", "PERSON"),
        ]
        # LLM only mentions Apple, not Tim Cook
        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.98),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        tim = next(e for e in result if e.text == "Tim Cook")
        assert tim.metadata.get("enhanced_by") == "openai"


class TestEnhanceEntitiesPreservesUntouched:
    """Test 7 – Entities not referenced by the LLM are preserved unchanged."""

    def test_untouched_entity_preserved(self):
        original = [
            _make_entity("Apple Inc.", "ORG", confidence=0.9),
            _make_entity("Cupertino", "GPE", confidence=0.85),
        ]
        # LLM only mentions Apple
        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        cupertino = next(e for e in result if e.text == "Cupertino")
        assert cupertino.label == "GPE"
        assert cupertino.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Relation enhancement tests
# ---------------------------------------------------------------------------

class TestEnhanceRelationsPredicateUpdate:
    """Test 8 – LLM corrects an existing relation's predicate.

    Because the relation identity includes the predicate, a correction appears
    as a *new* relation with the corrected predicate (the old one is preserved).
    If the caller's intent is to replace a generic "related_to" with a specific
    type, both the original and the corrected relation will be present in the
    result.  This is the correct graph-semantics behaviour: the LLM is additive,
    not destructive.
    """

    def test_new_predicate_appended_old_preserved(self):
        original = [_make_relation("Apple Inc.", "related_to", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            # LLM returns a more specific predicate for the same pair
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_relations(
            "Apple Inc. was founded by Steve Jobs.", original
        )

        # The original "related_to" is preserved; "founded_by" is added
        predicates = {r.predicate for r in result}
        assert "founded_by" in predicates, "LLM-suggested predicate must be added"
        assert "related_to" in predicates, "Original predicate must be preserved"

    def test_exact_triple_match_updates_confidence(self):
        """If the LLM returns the exact same triple it matched, only confidence changes."""
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs", confidence=0.5)]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.99),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        assert len(result) == 1
        assert result[0].predicate == "founded_by"
        assert result[0].confidence == pytest.approx(0.99)


class TestEnhanceRelationsConfidenceUpdate:
    """Test 9 – LLM updates an existing relation's confidence."""

    def test_confidence_is_updated(self):
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs", confidence=0.5)]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.99),
        ])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_relations("text", original)

        assert result[0].confidence == pytest.approx(0.99)


class TestEnhanceRelationsNewRelation:
    """Test 10 – LLM adds a new relation absent from the original list."""

    def test_new_relation_appended(self):
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
            # New relation: Apple Inc. located_in Cupertino
            RelationOut(subject="Apple Inc.", predicate="located_in", object="Cupertino", confidence=0.95),
        ])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_relations(
            "Apple Inc. is located in Cupertino.", original
        )

        predicates = [r.predicate for r in result]
        assert "located_in" in predicates, "New relation from LLM must be appended"
        assert len(result) == 2


class TestEnhanceRelationsEmptyResponse:
    """Test 11 – Empty LLM response preserves the original relations."""

    def test_empty_response_preserves_originals(self):
        original = [
            _make_relation("Apple Inc.", "founded_by", "Steve Jobs"),
            _make_relation("Steve Jobs", "ceo_of", "Apple Inc."),
        ]

        llm_resp = RelationsResponse(relations=[])
        extractor = _make_extractor(llm_resp)

        result = extractor.enhance_relations("text", original)

        assert len(result) == 2


class TestEnhanceRelationsMalformedResponse:
    """Test 12 – Provider raises; method falls back to original relations."""

    def test_fallback_on_provider_error(self):
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        extractor = LLMExtraction.__new__(LLMExtraction)
        extractor.provider_name = "openai"
        extractor.model = "gpt-4"
        extractor.temperature = None
        extractor.config = {}

        from semantica.utils.logging import get_logger
        extractor.logger = get_logger("test_llm_enhancement")
        from semantica.utils.progress_tracker import get_progress_tracker
        extractor.progress_tracker = get_progress_tracker()
        extractor.progress_tracker.enabled = False

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.generate_typed.side_effect = RuntimeError("api down")
        extractor.provider = mock_provider

        result = extractor.enhance_relations("text", original)

        assert len(result) == 1
        assert result[0].predicate == "founded_by"


class TestEnhanceRelationsMetadata:
    """Test 13 – enhanced_by and model metadata always present on relations."""

    def test_metadata_present_on_updated_relation(self):
        original = [_make_relation("Apple Inc.", "related_to", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        assert result[0].metadata.get("enhanced_by") == "openai"
        assert result[0].metadata.get("model") == "gpt-4"

    def test_metadata_present_on_new_relation(self):
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
            RelationOut(subject="Apple Inc.", predicate="located_in", object="Cupertino", confidence=0.9),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        new_rel = next(r for r in result if r.predicate == "located_in")
        assert new_rel.metadata.get("enhanced_by") == "openai"
        assert new_rel.metadata.get("model") == "gpt-4"

    def test_metadata_present_on_untouched_relation(self):
        """Relations not returned by the LLM must still get the metadata stamp."""
        original = [
            _make_relation("Apple Inc.", "founded_by", "Steve Jobs"),
            _make_relation("Steve Jobs", "ceo_of", "Apple Inc."),
        ]
        # LLM only mentions first relation
        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        untouched = next(r for r in result if r.predicate == "ceo_of")
        assert untouched.metadata.get("enhanced_by") == "openai"


class TestEnhanceRelationsPreservesUntouched:
    """Test 14 – Relations not referenced by the LLM are preserved unchanged."""

    def test_untouched_relation_preserved(self):
        original = [
            _make_relation("Apple Inc.", "founded_by", "Steve Jobs", confidence=0.9),
            _make_relation("Steve Jobs", "ceo_of", "Apple Inc.", confidence=0.85),
        ]
        # LLM only mentions first relation
        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        untouched = next(r for r in result if r.predicate == "ceo_of")
        assert untouched.subject.text == "Steve Jobs"
        assert untouched.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestNoDuplicateEntities:
    """Test 15 – LLM repeating an entity does not duplicate it."""

    def test_no_duplicate_on_repeated_entity(self):
        original = [_make_entity("Apple Inc.", "ORG")]

        # LLM returns the same entity twice
        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.95),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        apple_count = sum(1 for e in result if e.text == "Apple Inc.")
        assert apple_count == 1, "Duplicate entity must not be created"

    def test_no_duplicate_new_entity_mentioned_twice(self):
        original = [_make_entity("Apple Inc.", "ORG")]

        # LLM returns a new entity twice
        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Steve Jobs", label="PERSON", confidence=0.95),
            EntityOut(text="Steve Jobs", label="PERSON", confidence=0.93),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        jobs_count = sum(1 for e in result if e.text == "Steve Jobs")
        assert jobs_count == 1, "New entity mentioned twice must only be appended once"


class TestNoDuplicateRelations:
    """Test 16 – LLM repeating the same (subject, predicate, object) triple
    does not produce duplicate relations."""

    def test_no_duplicate_on_repeated_existing_triple(self):
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.95),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        triple_count = sum(
            1 for r in result
            if r.subject.text == "Apple Inc."
            and r.predicate == "founded_by"
            and r.object.text == "Steve Jobs"
        )
        assert triple_count == 1, "Duplicate triple must not be created"

    def test_no_duplicate_new_relation_triple_mentioned_twice(self):
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="located_in", object="Cupertino", confidence=0.9),
            RelationOut(subject="Apple Inc.", predicate="located_in", object="Cupertino", confidence=0.85),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        new_count = sum(1 for r in result if r.predicate == "located_in")
        assert new_count == 1, "New triple mentioned twice must only be appended once"


# ---------------------------------------------------------------------------
# Case-insensitive matching
# ---------------------------------------------------------------------------

class TestCaseInsensitiveMatching:
    """LLM may return entity text with different casing; matching must be
    case-insensitive so the existing entity is updated, not duplicated."""

    def test_entity_match_is_case_insensitive(self):
        original = [_make_entity("apple inc.", "PRODUCT")]

        llm_resp = EntitiesResponse(entities=[
            EntityOut(text="Apple Inc.", label="ORG", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_entities("text", original)

        # Should update in-place, not append a second entry
        assert len(result) == 1
        # The working copy retains the original casing for the text field
        assert result[0].label == "ORG"

    def test_relation_match_is_case_insensitive(self):
        """A relation whose endpoints differ only in casing from an LLM-returned
        triple with the same predicate must be matched and its confidence updated,
        not treated as a new relation."""
        original = [_make_relation("apple inc.", "founded_by", "steve jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs", confidence=0.97),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        # Exact triple match (case-insensitive) → update confidence only, no append
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# Correctness review additions (from final review pass)
# ---------------------------------------------------------------------------

class TestMultiplePredicatesSameEndpoints:
    """Two relations with the same subject/object but different predicates are
    both legitimate and must be preserved independently.

    This is the key regression for the (subj, obj) → (subj, pred, obj) identity
    fix: a (subj, obj)-keyed lookup would merge the second relation into the
    first instead of keeping them distinct.
    """

    def test_two_predicates_same_pair_both_preserved(self):
        """Original list has two relations between the same pair; LLM only
        mentions one of them. Both must survive in the result."""
        original = [
            _make_relation("Apple Inc.", "founded_by", "Steve Jobs"),
            _make_relation("Apple Inc.", "employs", "Steve Jobs"),
        ]
        # LLM only returns one of the two relations (updating its confidence)
        llm_resp = RelationsResponse(relations=[
            RelationOut(
                subject="Apple Inc.", predicate="founded_by",
                object="Steve Jobs", confidence=0.99,
            ),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        predicates = {r.predicate for r in result}
        assert "founded_by" in predicates
        assert "employs" in predicates, (
            "Second relation with different predicate must not be overwritten"
        )
        assert len(result) == 2

    def test_llm_adds_second_predicate_between_same_pair(self):
        """Original has one relation; LLM returns both the original and a second
        predicate between the same pair. Both must appear in the result."""
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(
                subject="Apple Inc.", predicate="founded_by",
                object="Steve Jobs", confidence=0.99,
            ),
            RelationOut(
                subject="Apple Inc.", predicate="employs",
                object="Steve Jobs", confidence=0.85,
            ),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        predicates = {r.predicate for r in result}
        assert "founded_by" in predicates
        assert "employs" in predicates
        assert len(result) == 2

    def test_llm_relation_with_different_predicate_does_not_corrupt_original(self):
        """An LLM relation with the same endpoints but a different predicate must
        not modify the confidence or any field of the original relation."""
        original = [
            _make_relation("Apple Inc.", "founded_by", "Steve Jobs", confidence=0.9),
        ]
        llm_resp = RelationsResponse(relations=[
            # Different predicate — this is a new relation, not an update
            RelationOut(
                subject="Apple Inc.", predicate="employs",
                object="Steve Jobs", confidence=0.5,
            ),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        # Original relation must be unmodified
        original_rel = next(r for r in result if r.predicate == "founded_by")
        assert original_rel.confidence == pytest.approx(0.9), (
            "Original relation confidence must not be changed by a different-predicate LLM entry"
        )


class TestNewRelationEndpointResolution:
    """When the LLM returns a new relation, its endpoints must resolve to the
    canonical Entity objects already present in the entity pool (from the
    original relations), not create inconsistent duplicate objects."""

    def test_new_relation_reuses_canonical_subject_entity(self):
        # Set up an original relation so "Apple Inc." is in the entity pool
        apple_entity = Entity(
            text="Apple Inc.", label="ORG",
            start_char=0, end_char=10,
            confidence=0.95,
            metadata={"canonical": True},
        )
        original = [
            Relation(
                subject=apple_entity,
                predicate="founded_by",
                object=_make_entity("Steve Jobs", "PERSON"),
                confidence=0.9, context="", metadata={},
            )
        ]

        llm_resp = RelationsResponse(relations=[
            # New relation reusing "Apple Inc." as subject
            RelationOut(
                subject="Apple Inc.", predicate="located_in",
                object="Cupertino", confidence=0.88,
            ),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("Apple Inc. is in Cupertino.", original)

        new_rel = next(r for r in result if r.predicate == "located_in")
        # The subject should be the canonical entity, not a fresh synthetic one
        assert new_rel.subject.label == "ORG", (
            "New relation's subject must resolve to the canonical entity, not UNKNOWN"
        )
        assert new_rel.subject.metadata.get("canonical") is True, (
            "New relation must reuse the canonical entity object from the pool"
        )

    def test_unresolvable_new_relation_endpoint_becomes_synthetic(self):
        """An endpoint text that does not match anything in the entity pool
        must produce a synthetic UNKNOWN entity, not raise."""
        original = [_make_relation("Apple Inc.", "founded_by", "Steve Jobs")]

        llm_resp = RelationsResponse(relations=[
            RelationOut(
                subject="Completely Unknown Corp", predicate="partner_of",
                object="Apple Inc.", confidence=0.7,
            ),
        ])
        extractor = _make_extractor(llm_resp)
        result = extractor.enhance_relations("text", original)

        new_rel = next(r for r in result if r.predicate == "partner_of")
        assert new_rel.subject.label == "UNKNOWN"
        assert new_rel.subject.metadata.get("synthetic") is True
        # Known endpoint should still resolve to canonical
        assert new_rel.object.label == "ORG"

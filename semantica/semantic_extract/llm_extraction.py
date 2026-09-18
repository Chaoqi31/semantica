"""
LLM Extraction Module

This module provides LLM-based extraction and enhancement capabilities using multiple
language model providers to improve entity and relation extraction quality through
post-processing and refinement.

Supported Providers:
    - "openai": OpenAI (GPT-3.5, GPT-4, etc.)
    - "gemini": Google Gemini (gemini-pro, etc.)
    - "groq": Groq (llama2, mixtral, etc.)
    - "anthropic": Anthropic Claude (claude-3-sonnet, etc.)
    - "ollama": Ollama (local open-source models)
    - "huggingface_llm": HuggingFace Transformers (custom LLM models)

Algorithms Used:
    - Prompt Engineering: Structured prompt construction for enhancement tasks
    - LLM Generation: Transformer-based language model text generation
    - Response Parsing: JSON parsing and structured output extraction
    - Entity Refinement: Confidence-based entity validation and correction
    - Relation Enhancement: Context-aware relation validation and improvement
    - Temperature Sampling: Stochastic sampling for diverse outputs

Key Features:
    - Entity extraction enhancement using LLMs
    - Relation extraction enhancement
    - Multi-provider support:
        * OpenAI (GPT-3.5, GPT-4, etc.)
        * Google Gemini (gemini-pro, etc.)
        * Groq (llama2, mixtral, etc.)
        * Anthropic Claude (claude-3-sonnet, etc.)
        * Ollama (local open-source models)
        * HuggingFace Transformers (custom LLM models)
    - Unified provider interface via providers module
    - Configurable model selection per provider
    - Automatic API key management from environment variables
    - Graceful fallback when LLM unavailable
    - Structured prompt generation for enhancement tasks

Main Classes:
    - LLMExtraction: Main LLM extraction coordinator
    - LLMResponse: LLM response representation dataclass

Example Usage:
    >>> from semantica.semantic_extract import LLMExtraction
    >>> # Using OpenAI
    >>> extractor = LLMExtraction(provider="openai", model="gpt-4")
    >>> enhanced_entities = extractor.enhance_entities(text, entities)
    >>> enhanced_relations = extractor.enhance_relations(text, relations)
    >>> 
    >>> # Using Gemini
    >>> extractor = LLMExtraction(provider="gemini", model="gemini-pro")
    >>> enhanced_entities = extractor.enhance_entities(text, entities)
    >>> 
    >>> # Using Ollama (local)
    >>> extractor = LLMExtraction(provider="ollama", model="llama2")
    >>> enhanced_entities = extractor.enhance_entities(text, entities)

Author: Semantica Contributors
License: MIT
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .ner_extractor import Entity
from .providers import create_provider
from .relation_extractor import Relation

try:
    from .schemas import EntitiesResponse, RelationsResponse
    _SCHEMAS_AVAILABLE = True
except ImportError:  # pydantic not installed
    _SCHEMAS_AVAILABLE = False
    EntitiesResponse = None  # type: ignore[assignment,misc]
    RelationsResponse = None  # type: ignore[assignment,misc]


@dataclass
class LLMResponse:
    """LLM response representation."""

    content: str
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]


class LLMExtraction:
    """LLM-based extraction and enhancement."""

    def __init__(self, provider: str = "openai", **config):
        """
        Initialize LLM extraction.

        Args:
            provider: LLM provider ("openai", "gemini", "groq", "anthropic", "ollama", "huggingface_llm")
            **config: Configuration options:
                - model: Model name (default depends on provider)
                - api_key: API key (from environment if not provided)
                - temperature: Temperature for generation (None = use model's default)
        """
        self.logger = get_logger("llm_extraction")
        self.config = config
        self.progress_tracker = get_progress_tracker()
        # Ensure progress tracker is enabled
        if not self.progress_tracker.enabled:
            self.progress_tracker.enabled = True

        self.provider_name = provider
        self.model = config.get("model")
        self.temperature = config.get("temperature")  # None = use model default

        # Initialize provider using new system
        try:
            # Sanitize config: remove api_key if it's None/empty to allow fallback
            provider_config = config.copy()
            if "api_key" in provider_config and not provider_config["api_key"]:
                del provider_config["api_key"]
                
            self.provider = create_provider(provider, **provider_config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize {provider} provider: {e}")
            self.provider = None

    def enhance_extractions(
        self, extractions: List[Any], text: str, **options
    ) -> List[Any]:
        """
        Enhance generic extractions (entities, relations, etc.).

        Args:
            extractions: List of extractions
            text: Input text
            **options: Enhancement options

        Returns:
            list: Enhanced extractions
        """
        if not extractions:
            return []

        # Determine type based on first element
        first = extractions[0]

        # Check for Entity
        if hasattr(first, "text") and hasattr(first, "label") and hasattr(first, "start_char"):
            return self.enhance_entities(text, extractions, **options)

        # Check for Relation
        if hasattr(first, "subject") and hasattr(first, "predicate") and hasattr(first, "object"):
            return self.enhance_relations(text, extractions, **options)

        # Default/Event fallback (mock implementation for now)
        self.logger.warning("Extraction type not fully supported for enhancement. Returning original.")
        return extractions

    def enhance_entities(
        self, text: str, entities: List[Entity], **options
    ) -> List[Entity]:
        """
        Enhance entity extraction using LLM.

        Calls the configured LLM provider with a structured prompt, parses the
        response using :class:`~.schemas.EntitiesResponse`, and merges the
        result back into the original entity list:

        - Existing entities whose text matches an LLM-returned entity have
          their ``label`` and ``confidence`` updated.
        - New entities returned by the LLM that have no match in the original
          list are appended.
        - Original entities not mentioned by the LLM are preserved unchanged.
        - All returned entities carry ``enhanced_by`` and ``model`` metadata.

        Falls back to the original entities on any LLM or parsing failure so
        that downstream processing is never blocked by an enhancement error.

        Args:
            text: Input text
            entities: Pre-extracted entities
            **options: Enhancement options

        Returns:
            list: Enhanced entities
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="semantic_extract",
            submodule="LLMExtraction",
            message="Enhancing entities using LLM",
        )

        try:
            if not self.provider or not self.provider.is_available():
                self.logger.warning(
                    "LLM provider not available. Returning original entities."
                )
                self.progress_tracker.stop_tracking(
                    tracking_id,
                    status="completed",
                    message="LLM provider not available",
                )
                return entities

            if not _SCHEMAS_AVAILABLE:
                self.logger.warning(
                    "Pydantic schemas not available; cannot perform typed LLM "
                    "enhancement. Returning original entities."
                )
                self.progress_tracker.stop_tracking(
                    tracking_id,
                    status="completed",
                    message="Schemas unavailable",
                )
                return entities

            self.progress_tracker.update_tracking(
                tracking_id, message="Building prompt..."
            )
            prompt = self._build_entity_prompt(text, entities)

            self.progress_tracker.update_tracking(
                tracking_id, message="Calling LLM API..."
            )
            gen_kwargs: Dict[str, Any] = {}
            if options.get("temperature", self.temperature) is not None:
                gen_kwargs["temperature"] = options.get("temperature", self.temperature)
            response_obj = self.provider.generate_typed(
                prompt, schema=EntitiesResponse, **gen_kwargs
            )

            self.progress_tracker.update_tracking(
                tracking_id, message="Parsing LLM response..."
            )
            enhanced_entities = self._parse_entity_response(response_obj, entities)

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Enhanced {len(enhanced_entities)} entities",
            )
            return enhanced_entities
        except Exception as e:
            self.logger.error(f"Failed to enhance entities with LLM: {e}")
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            return entities

    def enhance_relations(
        self, text: str, relations: List[Relation], **options
    ) -> List[Relation]:
        """
        Enhance relation extraction using LLM.

        Calls the configured LLM provider with a structured prompt, parses the
        response using :class:`~.schemas.RelationsResponse`, and merges the
        result back into the original relation list:

        - Existing relations whose subject+object pair matches an LLM-returned
          relation have their ``predicate`` and ``confidence`` updated.
        - New relations returned by the LLM that have no match in the original
          list are appended.
        - Original relations not affected by the LLM response are preserved.
        - All returned relations carry ``enhanced_by`` and ``model`` metadata.

        Falls back to the original relations on any LLM or parsing failure.

        Args:
            text: Input text
            relations: Pre-extracted relations
            **options: Enhancement options

        Returns:
            list: Enhanced relations
        """
        if not self.provider or not self.provider.is_available():
            self.logger.warning(
                "LLM provider not available. Returning original relations."
            )
            return relations

        if not _SCHEMAS_AVAILABLE:
            self.logger.warning(
                "Pydantic schemas not available; cannot perform typed LLM "
                "enhancement. Returning original relations."
            )
            return relations

        prompt = self._build_relation_prompt(text, relations)

        try:
            gen_kwargs: Dict[str, Any] = {}
            if options.get("temperature", self.temperature) is not None:
                gen_kwargs["temperature"] = options.get("temperature", self.temperature)
            response_obj = self.provider.generate_typed(
                prompt, schema=RelationsResponse, **gen_kwargs
            )
            enhanced_relations = self._parse_relation_response(
                response_obj, relations, text
            )
            return enhanced_relations
        except Exception as e:
            self.logger.error(f"Failed to enhance relations with LLM: {e}")
            return relations

    def _build_entity_prompt(self, text: str, entities: List[Entity]) -> str:
        """Build prompt for entity enhancement.

        User-supplied content is serialised as JSON strings so that special
        characters (newlines, quotes, prompt-injection attempts) cannot escape
        the data section and override the system instructions.
        """
        import json as _json
        safe_text = _json.dumps(text)
        safe_entities = _json.dumps([{"text": e.text, "label": e.label} for e in entities])

        return f"""Analyze the text provided in the JSON fields below and enhance the entity extraction.

INPUT_TEXT: {safe_text}

EXTRACTED_ENTITIES: {safe_entities}

Please:
1. Verify each entity is correctly identified
2. Suggest any missing entities
3. Improve entity type classifications
4. Provide confidence scores

Return a JSON object with an "entities" key containing an array of entity objects.
Each entity object must have: "text" (string), "label" (string), "confidence" (float 0-1).
Example: {{"entities": [{{"text": "Apple Inc.", "label": "ORG", "confidence": 0.97}}]}}"""

    def _build_relation_prompt(self, text: str, relations: List[Relation]) -> str:
        """Build prompt for relation enhancement.

        User-supplied content is serialised as JSON strings to prevent
        prompt-injection via crafted text or relation labels.
        """
        import json as _json
        safe_text = _json.dumps(text)
        safe_relations = _json.dumps(
            [
                {
                    "subject": r.subject.text,
                    "predicate": r.predicate,
                    "object": r.object.text,
                }
                for r in relations
            ]
        )

        return f"""Analyze the text provided in the JSON fields below and enhance the relation extraction.

INPUT_TEXT: {safe_text}

EXTRACTED_RELATIONS: {safe_relations}

Please:
1. Verify each relation is correct
2. Suggest any missing relations
3. Improve relation type classifications
4. Provide confidence scores

Return a JSON object with a "relations" key containing an array of relation objects.
Each relation object must have: "subject" (string), "predicate" (string), "object" (string), "confidence" (float 0-1).
Example: {{"relations": [{{"subject": "Apple", "predicate": "founded_by", "object": "Steve Jobs", "confidence": 0.95}}]}}"""

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_entity_response(
        self,
        response_obj: Any,
        original_entities: List[Entity],
    ) -> List[Entity]:
        """Merge the LLM-returned :class:`~.schemas.EntitiesResponse` into the
        original entity list.

        Merge strategy:

        * For every entity returned by the LLM, look for an existing entity
          with the same text (case-insensitive).  If found, update its
          ``label`` and ``confidence`` in-place (on a copy so the caller's
          original list is not mutated).
        * LLM-returned entities with no match in the original list are
          **appended** as new entities (``start_char``/``end_char`` default to
          0 because the LLM does not reliably return character offsets).
        * Original entities absent from the LLM response are **preserved**.
        * All returned entities carry ``enhanced_by`` / ``model`` metadata.
        * If ``response_obj`` is empty or unusable the original list is
          returned unchanged (no silent data loss).
        """
        # Guard: nothing from the LLM → return originals as-is
        llm_entities = getattr(response_obj, "entities", None)
        if not llm_entities:
            self.logger.debug(
                "_parse_entity_response: empty LLM response; keeping original entities"
            )
            for entity in original_entities:
                if entity.metadata is None:
                    entity.metadata = {}
                entity.metadata.update({"enhanced_by": self.provider_name, "model": self.model})
            return original_entities

        # Build a lookup from lowercased text → index in the working copy
        working: List[Entity] = []
        text_to_idx: Dict[str, int] = {}
        for entity in original_entities:
            idx = len(working)
            # Deep-copy metadata so we don't mutate the caller's objects
            new_meta = dict(entity.metadata) if entity.metadata else {}
            working.append(Entity(
                text=entity.text,
                label=entity.label,
                start_char=entity.start_char,
                end_char=entity.end_char,
                confidence=entity.confidence,
                metadata=new_meta,
            ))
            text_to_idx[entity.text.lower()] = idx

        seen_new: set = set()  # guard against duplicate new entities

        for e_out in llm_entities:
            key = e_out.text.lower()
            if key in text_to_idx:
                # Update existing entity
                existing = working[text_to_idx[key]]
                if e_out.label:
                    existing.label = e_out.label
                if e_out.confidence is not None:
                    existing.confidence = e_out.confidence
                existing.metadata.update({
                    "enhanced_by": self.provider_name,
                    "model": self.model,
                })
            else:
                # New entity from LLM — append only once
                if key not in seen_new:
                    seen_new.add(key)
                    working.append(Entity(
                        text=e_out.text,
                        label=e_out.label or "UNKNOWN",
                        start_char=0,
                        end_char=0,
                        confidence=e_out.confidence,
                        metadata={
                            "enhanced_by": self.provider_name,
                            "model": self.model,
                            "extraction_method": "llm_enhancement",
                        },
                    ))

        # Stamp any original entities not touched by the LLM
        for entity in working:
            if entity.metadata is None:
                entity.metadata = {}
            entity.metadata.setdefault("enhanced_by", self.provider_name)
            entity.metadata.setdefault("model", self.model)

        return working

    def _parse_relation_response(
        self,
        response_obj: Any,
        original_relations: List[Relation],
        text: str = "",
    ) -> List[Relation]:
        """Merge the LLM-returned :class:`~.schemas.RelationsResponse` into
        the original relation list.

        Merge strategy:

        * For every relation returned by the LLM, look for an existing relation
          whose ``(subject.text, predicate, object.text)`` triple all match
          (case-insensitive).  If found, update its ``confidence`` (the predicate
          already matches, so there is nothing else to update on an exact triple
          match).
        * If the LLM returns a relation with a matching ``(subject, object)`` but
          a **different** predicate, that is treated as a **new** relation and
          appended — multiple predicates between the same entity pair are
          legitimate in a property graph.
        * LLM-returned relations with no match in the original list are
          **appended** as new relations. Subject/object entities are resolved
          against the subjects/objects already present in the original list;
          unresolved endpoints become synthetic UNKNOWN entities.
        * Original relations absent from the LLM response are **preserved**.
        * All returned relations carry ``enhanced_by`` / ``model`` metadata.
        * Empty/unusable responses preserve the original list unchanged.
        """
        llm_relations = getattr(response_obj, "relations", None)
        if not llm_relations:
            self.logger.debug(
                "_parse_relation_response: empty LLM response; keeping original relations"
            )
            for rel in original_relations:
                if rel.metadata is None:
                    rel.metadata = {}
                rel.metadata.update({"enhanced_by": self.provider_name, "model": self.model})
            return original_relations

        # Build a lookup: (subj_lower, predicate_lower, obj_lower) → index in
        # working copy.  Including the predicate ensures that two relations with
        # the same endpoints but different predicates are treated as distinct
        # relations and never accidentally merged.
        working: List[Relation] = []
        triple_to_idx: Dict[tuple, int] = {}
        for rel in original_relations:
            idx = len(working)
            new_meta = dict(rel.metadata) if rel.metadata else {}
            working.append(Relation(
                subject=rel.subject,
                predicate=rel.predicate,
                object=rel.object,
                confidence=rel.confidence,
                context=rel.context,
                metadata=new_meta,
            ))
            key = (
                rel.subject.text.lower(),
                rel.predicate.lower(),
                rel.object.text.lower(),
            )
            # Keep first occurrence only (guarding against pre-existing duplicates)
            triple_to_idx.setdefault(key, idx)

        # Build a flat entity pool from original relations for endpoint resolution
        entity_pool: List[Entity] = []
        seen_ent_texts: set = set()
        for rel in original_relations:
            for ent in (rel.subject, rel.object):
                t = ent.text.lower()
                if t not in seen_ent_texts:
                    entity_pool.append(ent)
                    seen_ent_texts.add(t)

        seen_new: set = set()  # guard against duplicate new relations

        for r_out in llm_relations:
            subj_text = (r_out.subject or "").strip()
            obj_text = (r_out.object or "").strip()
            pred_text = (r_out.predicate or "").strip()
            if not subj_text or not obj_text:
                continue

            triple_key = (
                subj_text.lower(),
                pred_text.lower(),
                obj_text.lower(),
            )

            if triple_key in triple_to_idx:
                # Exact triple match — update only the confidence
                existing = working[triple_to_idx[triple_key]]
                if r_out.confidence is not None:
                    existing.confidence = r_out.confidence
                existing.metadata.update({
                    "enhanced_by": self.provider_name,
                    "model": self.model,
                })
            else:
                # No match — append as a new relation (once per unique triple)
                if triple_key not in seen_new:
                    seen_new.add(triple_key)

                    # Resolve endpoints against the existing entity pool so the
                    # new relation reuses canonical Entity objects where available
                    subj_entity = next(
                        (e for e in entity_pool if e.text.lower() == subj_text.lower()),
                        Entity(
                            text=subj_text, label="UNKNOWN",
                            start_char=0, end_char=len(subj_text),
                            confidence=0.8, metadata={"synthetic": True},
                        ),
                    )
                    obj_entity = next(
                        (e for e in entity_pool if e.text.lower() == obj_text.lower()),
                        Entity(
                            text=obj_text, label="UNKNOWN",
                            start_char=0, end_char=len(obj_text),
                            confidence=0.8, metadata={"synthetic": True},
                        ),
                    )

                    working.append(Relation(
                        subject=subj_entity,
                        predicate=pred_text or "related_to",
                        object=obj_entity,
                        confidence=(
                            r_out.confidence if r_out.confidence is not None else 0.9
                        ),
                        context=text,
                        metadata={
                            "enhanced_by": self.provider_name,
                            "model": self.model,
                            "extraction_method": "llm_enhancement",
                        },
                    ))

        # Stamp any original relations not touched by the LLM
        for rel in working:
            if rel.metadata is None:
                rel.metadata = {}
            rel.metadata.setdefault("enhanced_by", self.provider_name)
            rel.metadata.setdefault("model", self.model)

        return working


# Alias for backward compatibility
LLMEnhancer = LLMExtraction

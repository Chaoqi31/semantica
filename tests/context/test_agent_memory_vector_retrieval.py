"""AgentMemory.retrieve returns the memories a VectorStore finds (#1724)."""

import pytest

from semantica.context import AgentMemory
from semantica.vector_store import VectorStore

REACTOR = "The reactor coolant pump tripped on Tuesday"
REVENUE = "Quarterly revenue grew by twelve percent"
QUERY = "nuclear plant outage"


@pytest.fixture
def vector_store():
    return VectorStore(backend="inmemory")


@pytest.fixture
def memory(vector_store):
    memory = AgentMemory(vector_store=vector_store)
    memory.store(REACTOR)
    memory.store(REVENUE)
    return memory


def test_retrieve_returns_vector_hits_without_keyword_overlap(memory):
    results = memory.retrieve(QUERY)

    assert REACTOR in {r["content"] for r in results}
    assert {r["memory_id"] for r in results} <= set(memory.memory_items)


def test_retrieve_skips_vectors_that_are_not_memories(vector_store, memory):
    vector_store.store_vectors(
        vectors=[vector_store.embed("nuclear plant outage report")],
        metadata=[{"type": "decision"}],
    )

    results = memory.retrieve(QUERY)

    assert results
    assert {r["memory_id"] for r in results} <= set(memory.memory_items)

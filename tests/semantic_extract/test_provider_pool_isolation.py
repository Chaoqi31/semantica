"""
Regression tests for ProviderPool credential isolation and concurrency safety.

Issue
-----
``ProviderPool`` computed its cache key from the raw kwargs passed to
``create_provider()``.  When callers omitted ``api_key`` and relied on
environment-variable resolution inside the provider ``__init__``, the
resolved credential was *not* part of the pool key.  Rotating
``OPENAI_API_KEY`` (or any provider env var) between two calls with
otherwise identical arguments caused the second call to receive the
stale cached provider — a credential cross-contamination bug.

There was also an unsynchronised check-then-create-then-store sequence
in ``ProviderPool.get()`` that let concurrent threads construct duplicate
providers for the same key.

Fix
---
* ``ProviderPool.get()`` now resolves the effective API key (via the same
  ``config.get_api_key() → env-var`` fallback chain the provider
  ``__init__`` uses) *before* computing the cache key, and injects it into
  kwargs so the resolved credential is always represented in the key.
* A ``threading.Lock`` with double-checked locking serialises first-time
  construction so that concurrent callers with the same key always receive
  the same instance.

Tests in this module
---------------------
TestCredentialIsolation
  - env-key rotation creates a fresh provider
  - explicit api_key=A and api_key=B are isolated
  - same effective config still reuses the cached provider
  - use_pool=False always bypasses the pool
  - api_key is not leaked into debug log output

TestConcurrencySafety
  - concurrent first-time requests for the same config produce exactly one
    canonical provider (single construction, same instance returned to all)
"""

import os
import threading
import unittest
from unittest.mock import MagicMock, patch

from semantica.semantic_extract.providers import (
    ProviderPool,
    _provider_pool,
    create_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool() -> ProviderPool:
    """Return a fresh, isolated ProviderPool for each test."""
    return ProviderPool()


def _mock_openai_cls(instances: list):
    """Return a side_effect callable that appends each new mock to *instances*."""
    def factory(*args, **kwargs):
        m = MagicMock(name=f"OpenAIProvider-{len(instances)}")
        instances.append(m)
        return m
    return factory


# ---------------------------------------------------------------------------
# Credential isolation
# ---------------------------------------------------------------------------

class TestCredentialIsolation(unittest.TestCase):
    """ProviderPool must isolate providers by effective credential."""

    def setUp(self):
        # Every test gets its own pool so there is no cross-test state.
        self.pool = _make_pool()

    # ------------------------------------------------------------------
    # Core bug: env-key rotation must yield a new provider
    # ------------------------------------------------------------------

    def test_env_key_rotation_creates_fresh_provider(self):
        """
        Rotating the environment API key between two otherwise-identical
        calls must produce two distinct provider instances.

        This is the primary regression test: before the fix the pool key
        did not contain the env-var-resolved credential, so both calls
        received the same (stale) instance.
        """
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key-A"}, clear=False):
                p1 = self.pool.get("openai", model="gpt-4")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key-B"}, clear=False):
                p2 = self.pool.get("openai", model="gpt-4")

        # Two different credentials → two different instances
        self.assertIsNot(
            p1,
            p2,
            "Rotating OPENAI_API_KEY must produce a new provider instance, "
            "not reuse the one initialised with the old key.",
        )
        self.assertEqual(len(instances), 2, "Expected exactly two provider constructions.")

    def test_env_key_rotation_via_create_provider(self):
        """Same assertion exercised through the public ``create_provider`` API."""
        pool = _make_pool()
        instances = []

        with patch(
            "semantica.semantic_extract.providers._provider_pool",
            pool,
        ):
            with patch(
                "semantica.semantic_extract.providers.OpenAIProvider",
                side_effect=_mock_openai_cls(instances),
            ):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-A"}, clear=False):
                    p1 = create_provider("openai", model="gpt-4")

                with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-B"}, clear=False):
                    p2 = create_provider("openai", model="gpt-4")

        self.assertIsNot(p1, p2)
        self.assertEqual(len(instances), 2)

    # ------------------------------------------------------------------
    # Explicit api_key isolation (must still work after the fix)
    # ------------------------------------------------------------------

    def test_explicit_different_keys_are_isolated(self):
        """Two calls with different explicit api_key values must not share an instance."""
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            p1 = self.pool.get("openai", api_key="sk-A", model="gpt-4")
            p2 = self.pool.get("openai", api_key="sk-B", model="gpt-4")

        self.assertIsNot(p1, p2)
        self.assertEqual(len(instances), 2)

    def test_explicit_same_key_reuses_instance(self):
        """Two calls with the same explicit api_key and model must share one instance."""
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            p1 = self.pool.get("openai", api_key="sk-same", model="gpt-4")
            p2 = self.pool.get("openai", api_key="sk-same", model="gpt-4")

        self.assertIs(p1, p2)
        self.assertEqual(len(instances), 1, "Same config must not construct a second instance.")

    # ------------------------------------------------------------------
    # Same effective credential reuses the cached provider
    # ------------------------------------------------------------------

    def test_same_env_key_reuses_provider(self):
        """If the env key does not change, repeat calls must return the cached instance."""
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-stable"}, clear=False):
                p1 = self.pool.get("openai", model="gpt-4")
                p2 = self.pool.get("openai", model="gpt-4")

        self.assertIs(p1, p2, "Same config must reuse the cached provider.")
        self.assertEqual(len(instances), 1)

    def test_explicit_key_matches_env_key_reuses_provider(self):
        """
        If the caller passes api_key=X explicitly and the env var is also X,
        repeated calls must still reuse the same instance (key must be stable).
        """
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-same-X"}, clear=False):
                p1 = self.pool.get("openai", api_key="sk-same-X", model="gpt-4")
                # Second call omits explicit key — env var resolves to same value
                p2 = self.pool.get("openai", model="gpt-4")

        self.assertIs(p1, p2)
        self.assertEqual(len(instances), 1)

    # ------------------------------------------------------------------
    # use_pool=False
    # ------------------------------------------------------------------

    def test_use_pool_false_always_creates_fresh_instance(self):
        """use_pool=False must bypass the pool and always construct a new provider."""
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-any"}, clear=False):
                p1 = create_provider("openai", use_pool=False, model="gpt-4")
                p2 = create_provider("openai", use_pool=False, model="gpt-4")

        self.assertIsNot(p1, p2)
        self.assertEqual(len(instances), 2)

    # ------------------------------------------------------------------
    # Providers that do not use api_key are unaffected
    # ------------------------------------------------------------------

    def test_ollama_provider_unaffected_no_api_key(self):
        """
        OllamaProvider does not use an API key.  The pool must still work
        correctly (same config → same instance) without injecting a spurious
        api_key kwarg.
        """
        instances = []

        with patch(
            "semantica.semantic_extract.providers.OllamaProvider",
            side_effect=_mock_openai_cls(instances),
        ):
            p1 = self.pool.get("ollama", model="llama2", base_url="http://localhost:11434")
            p2 = self.pool.get("ollama", model="llama2", base_url="http://localhost:11434")

        self.assertIs(p1, p2)
        self.assertEqual(len(instances), 1)

        # Verify api_key was NOT injected into the pool key for Ollama.
        pool_key = ProviderPool._make_key(
            "ollama", model="llama2", base_url="http://localhost:11434"
        )
        self.assertNotIn("api_key", pool_key)

    # ------------------------------------------------------------------
    # api_key must not appear in log output
    # ------------------------------------------------------------------

    def test_api_key_not_logged(self):
        """
        The debug log message emitted when a new provider is created must
        not contain the raw API key value.
        """
        import logging

        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(self.format(record))

        handler = CapturingHandler()
        logging.getLogger().addHandler(handler)
        try:
            with patch(
                "semantica.semantic_extract.providers.OpenAIProvider",
                side_effect=_mock_openai_cls([]),
            ):
                with patch.dict(
                    os.environ, {"OPENAI_API_KEY": "sk-secret-should-not-log"}, clear=False
                ):
                    self.pool.get("openai", model="gpt-4")
        finally:
            logging.getLogger().removeHandler(handler)

        combined = "\n".join(log_records)
        self.assertNotIn(
            "sk-secret-should-not-log",
            combined,
            "Raw API key value must not appear in log output.",
        )


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------

class TestConcurrencySafety(unittest.TestCase):
    """ProviderPool.get() must serialise first-time construction under concurrency."""

    def setUp(self):
        self.pool = _make_pool()

    def test_concurrent_first_access_single_construction(self):
        """
        N threads simultaneously requesting the same (provider, config) for
        the first time must result in exactly ONE provider construction and
        all threads receiving the SAME instance.

        This is the regression test for the check-then-create race:
        before the fix, concurrent threads could each pass the ``if key in
        self._providers`` check before any one of them stored its result,
        causing N duplicate constructions and non-deterministic aliasing.
        """
        num_threads = 20
        results: list = [None] * num_threads
        construction_count = 0
        construction_lock = threading.Lock()

        # Introduce a small delay inside the constructor to widen the race
        # window so that — without the fix — multiple threads would almost
        # certainly construct duplicate providers.
        import time

        def slow_factory(*args, **kwargs):
            nonlocal construction_count
            time.sleep(0.02)  # 20 ms — enough to expose the race
            with construction_lock:
                construction_count += 1
            m = MagicMock(name=f"provider-{construction_count}")
            return m

        barrier = threading.Barrier(num_threads)

        def worker(idx):
            barrier.wait()  # all threads start simultaneously
            results[idx] = self.pool.get("openai", api_key="sk-concurrent", model="gpt-4")

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=slow_factory,
        ):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        # Exactly one construction must have occurred.
        self.assertEqual(
            construction_count,
            1,
            f"Expected exactly 1 provider construction under concurrency, "
            f"got {construction_count}.",
        )

        # Every thread must have received the same instance.
        canonical = results[0]
        self.assertIsNotNone(canonical)
        for idx, result in enumerate(results):
            self.assertIs(
                result,
                canonical,
                f"Thread {idx} received a different provider instance than thread 0.",
            )

    def test_clear_is_thread_safe(self):
        """
        ``clear()`` must not corrupt the pool's internal state when called
        concurrently with ``get()`` calls.
        """
        import time

        instances: list = []
        errors: list = []

        def factory(*args, **kwargs):
            time.sleep(0.001)
            m = MagicMock()
            instances.append(m)
            return m

        stop_event = threading.Event()

        def getter():
            while not stop_event.is_set():
                try:
                    self.pool.get("openai", api_key="sk-x", model="gpt-4")
                except Exception as exc:
                    errors.append(exc)

        def clearer():
            for _ in range(20):
                self.pool.clear()
                time.sleep(0.005)

        with patch(
            "semantica.semantic_extract.providers.OpenAIProvider",
            side_effect=factory,
        ):
            getter_threads = [threading.Thread(target=getter) for _ in range(4)]
            clear_thread = threading.Thread(target=clearer)

            for t in getter_threads:
                t.start()
            clear_thread.start()

            clear_thread.join(timeout=5)
            stop_event.set()
            for t in getter_threads:
                t.join(timeout=5)

        self.assertEqual(
            errors,
            [],
            f"No exceptions expected during concurrent get/clear, got: {errors}",
        )


# ---------------------------------------------------------------------------
# _make_key and _resolve_api_key unit tests
# ---------------------------------------------------------------------------

class TestProviderPoolInternals(unittest.TestCase):
    """Unit tests for ProviderPool helper methods."""

    def setUp(self):
        self.pool = _make_pool()

    def test_make_key_is_deterministic(self):
        """_make_key must return the same string for equivalent kwargs regardless of insertion order."""
        k1 = ProviderPool._make_key("openai", api_key="sk-x", model="gpt-4", temperature=0.5)
        k2 = ProviderPool._make_key("openai", model="gpt-4", temperature=0.5, api_key="sk-x")
        self.assertEqual(k1, k2)

    def test_make_key_differs_for_different_api_keys(self):
        k1 = ProviderPool._make_key("openai", api_key="sk-A", model="gpt-4")
        k2 = ProviderPool._make_key("openai", api_key="sk-B", model="gpt-4")
        self.assertNotEqual(k1, k2)

    def test_make_key_nested_dict_is_stable(self):
        k1 = ProviderPool._make_key("openai", opts={"a": 1, "b": 2})
        k2 = ProviderPool._make_key("openai", opts={"b": 2, "a": 1})
        self.assertEqual(k1, k2)

    def test_resolve_api_key_explicit_kwarg(self):
        """Explicit kwarg is returned as-is."""
        result = self.pool._resolve_api_key("openai", {"api_key": "sk-explicit"})
        self.assertEqual(result, "sk-explicit")

    def test_resolve_api_key_env_var_fallback(self):
        """Falls back to env var when no explicit kwarg."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}, clear=False):
            result = self.pool._resolve_api_key("openai", {})
        self.assertEqual(result, "sk-from-env")

    def test_resolve_api_key_none_for_ollama(self):
        """Ollama is not in _API_KEY_PROVIDERS; must return None."""
        result = self.pool._resolve_api_key("ollama", {})
        self.assertIsNone(result)

    def test_resolve_api_key_none_for_huggingface_llm(self):
        """HuggingFace LLM provider must return None."""
        result = self.pool._resolve_api_key("huggingface_llm", {})
        self.assertIsNone(result)

    def test_resolve_api_key_none_when_no_key_anywhere(self):
        """When no key is set anywhere, must return None (not raise)."""
        env_key = "OPENAI_API_KEY"
        original = os.environ.pop(env_key, None)
        try:
            # Also ensure config singleton has no key cached
            with patch(
                "semantica.semantic_extract.providers.config.get_api_key",
                return_value=None,
            ):
                result = self.pool._resolve_api_key("openai", {})
            self.assertIsNone(result)
        finally:
            if original is not None:
                os.environ[env_key] = original

    def test_all_api_key_providers_covered(self):
        """All built-in API-key-using providers must be listed in _API_KEY_PROVIDERS."""
        expected = {"openai", "gemini", "groq", "anthropic", "deepseek", "novita"}
        self.assertEqual(ProviderPool._API_KEY_PROVIDERS, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

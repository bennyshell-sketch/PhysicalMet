"""Every external call passes the guard first, or it is never made."""

from dataclasses import replace

import pytest

from dive.privacy import EgressGuard, PrivacyPolicy
from dive.providers import (
    ClaudeProvider,
    GeminiProvider,
    LocalProvider,
    ProviderRouter,
    _to_gemini_schema,
    available_providers,
)

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "array", "items": {"type": "number"}}},
    "required": ["a"],
    "additionalProperties": False,
}


def test_every_provider_reports_its_availability():
    described = {d["provider"]: d for d in available_providers()}
    assert {"anthropic", "google", "local"} <= set(described)
    for entry in described.values():
        assert isinstance(entry["available"], bool)


def test_a_missing_key_is_reported_not_guessed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    available, reason = ClaudeProvider().available()
    assert not available
    assert "ANTHROPIC_API_KEY" in reason


def test_a_key_is_never_exposed_by_a_provider_description(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-super-secret-key-value")
    described = GeminiProvider().describe()
    assert "super-secret" not in str(described)
    assert len(described["key_fingerprint"]) == 12


def test_the_router_blocks_an_identifier_bearing_prompt_before_any_call():
    router = ProviderRouter(guard=EgressGuard(PrivacyPolicy()))
    response = router.complete("system", "summarise for WL042, NHS 943 476 5919")
    assert not response.ok
    assert response.blocked_by_guard
    assert "privacy guard" in response.reason


def test_a_clean_prompt_reaches_the_availability_check():
    """It should fail on credentials, not on the guard."""
    router = ProviderRouter(guard=EgressGuard(PrivacyPolicy()))
    response = router.complete("system", "grip strength sarcopenia older adults")
    assert not response.blocked_by_guard


def test_offline_only_stops_every_provider():
    router = ProviderRouter(guard=EgressGuard(PrivacyPolicy(offline_only=True)))
    response = router.complete("system", "entirely benign clinical question")
    assert not response.ok
    assert response.blocked_by_guard


def test_each_destination_is_reviewed_separately():
    """A payload cleared for one provider is not thereby cleared for another."""
    guard = EgressGuard(PrivacyPolicy(allowed_destinations=("anthropic",)))
    router = ProviderRouter(order=("google",), guard=guard)
    response = router.complete("system", "benign clinical question")
    assert not response.ok
    assert "not allowed" in response.reason


def test_gemini_schema_translation():
    converted = _to_gemini_schema(SCHEMA)
    assert converted["type"] == "OBJECT"
    assert converted["properties"]["a"]["type"] == "STRING"
    assert converted["properties"]["b"]["items"]["type"] == "NUMBER"
    # Gemini rejects additionalProperties.
    assert "additionalProperties" not in converted


def test_local_provider_needs_no_third_party_key():
    available, _ = LocalProvider(base_url="http://localhost:8000/v1").available()
    assert available

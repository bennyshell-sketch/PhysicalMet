"""External model providers, behind one interface and one privacy boundary.

Dive can use Claude, Gemini or OpenAI. None is called directly from anywhere
else in the codebase: every request passes through :class:`~dive.privacy.EgressGuard`
first, and a blocked decision means the HTTP call is never made.

The point of routing every provider through a single chokepoint is that the
privacy question then has exactly one answer to audit. "What did we send to
Google?" is answered by the audit log, not by reading every call site.

Three things worth knowing before enabling either provider:

**Retention.** Both vendors offer arrangements where prompts are not retained
or used for training, but the terms differ, change, and depend on your account
tier. That is a contract to check, not a flag to set. Dive's position is to
assume the provider sees whatever is sent and to minimise what is sent.

**Failover is not free.** ``ProviderRouter`` can fall back from one provider to
another, which means a payload blocked for one destination must be re-reviewed
for the next - a guard decision is per-destination, and the router re-runs it.

**Local models.** ``provider="local"`` targets an OpenAI-compatible endpoint on
your own infrastructure (llama.cpp, vLLM, Ollama). It is the only configuration
under which no data leaves your network at all, and it is the right answer for
deployments that cannot accept third-party processing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .config import SETTINGS, Settings
from .privacy import EgressGuard, PrivacyPolicy
from .security import SecretRef

__all__ = [
    "ProviderResponse",
    "LLMProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "LocalProvider",
    "ProviderRouter",
    "available_providers",
]

ProviderName = Literal["anthropic", "google", "openai", "local"]


@dataclass
class ProviderResponse:
    text: str = ""
    parsed: dict | None = None
    provider: str = ""
    model: str = ""
    ok: bool = False
    reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    blocked_by_guard: bool = False


class LLMProvider(ABC):
    """One external model, with its credential held by reference."""

    name: ProviderName
    default_model: str
    key_env: str

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model
        self.secret = SecretRef(self.key_env)

    def available(self) -> tuple[bool, str]:
        if not self.secret.present():
            return False, f"{self.key_env} is not set"
        return self._sdk_available()

    @abstractmethod
    def _sdk_available(self) -> tuple[bool, str]: ...

    @abstractmethod
    def complete(
        self, system: str, user: str, schema: dict | None = None, max_tokens: int = 16000
    ) -> ProviderResponse: ...

    def describe(self) -> dict[str, Any]:
        ok, reason = self.available()
        return {
            "provider": self.name,
            "model": self.model,
            "available": ok,
            "reason": reason,
            "key_fingerprint": self.secret.fingerprint(),
        }


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------

class ClaudeProvider(LLMProvider):
    name: ProviderName = "anthropic"
    default_model = "claude-opus-5"
    key_env = "ANTHROPIC_API_KEY"

    def _sdk_available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the `anthropic` package is not installed"
        return True, ""

    def complete(
        self, system: str, user: str, schema: dict | None = None, max_tokens: int = 16000
    ) -> ProviderResponse:
        ok, reason = self.available()
        if not ok:
            return ProviderResponse(provider=self.name, model=self.model, reason=reason)

        import anthropic

        output_config: dict[str, Any] = {"effort": "high"}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config=output_config,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason=f"{type(exc).__name__}: {exc}",
            )

        if getattr(response, "stop_reason", None) == "refusal":
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason="the model declined this request",
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }
        return ProviderResponse(
            text=text, parsed=_safe_json(text) if schema else None,
            provider=self.name, model=self.model, ok=True, usage=usage,
        )


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    name: ProviderName = "google"
    default_model = "gemini-2.5-pro"
    key_env = "GEMINI_API_KEY"

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def _sdk_available(self) -> tuple[bool, str]:
        return True, ""     # REST; no SDK dependency

    def complete(
        self, system: str, user: str, schema: dict | None = None, max_tokens: int = 16000
    ) -> ProviderResponse:
        ok, reason = self.available()
        if not ok:
            return ProviderResponse(provider=self.name, model=self.model, reason=reason)

        generation: dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": 0.2}
        if schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = _to_gemini_schema(schema)

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }

        request = urllib.request.Request(
            self.ENDPOINT.format(model=self.model),
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                # Header, not query string: a key in a URL ends up in logs.
                "x-goog-api-key": self.secret.resolve() or "",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as handle:
                body = json.loads(handle.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300] if hasattr(exc, "read") else ""
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason=f"HTTP {exc.code}: {detail}",
            )
        except Exception as exc:
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason=f"{type(exc).__name__}: {exc}",
            )

        candidates = body.get("candidates", [])
        if not candidates:
            blocked = body.get("promptFeedback", {}).get("blockReason")
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason=f"no candidate returned{f' (blocked: {blocked})' if blocked else ''}",
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage_meta = body.get("usageMetadata", {})
        return ProviderResponse(
            text=text, parsed=_safe_json(text) if schema else None,
            provider=self.name, model=self.model, ok=True,
            usage={
                "input_tokens": usage_meta.get("promptTokenCount"),
                "output_tokens": usage_meta.get("candidatesTokenCount"),
            },
        )


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    name: ProviderName = "openai"
    default_model = "gpt-4o"
    key_env = "OPENAI_API_KEY"

    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def _sdk_available(self) -> tuple[bool, str]:
        return True, ""     # REST; no SDK dependency

    def complete(
        self, system: str, user: str, schema: dict | None = None, max_tokens: int = 16000
    ) -> ProviderResponse:
        ok, reason = self.available()
        if not ok:
            return ProviderResponse(provider=self.name, model=self.model, reason=reason)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "dive_response", "strict": True, "schema": schema},
            }

        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.secret.resolve() or ''}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as handle:
                body = json.loads(handle.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300] if hasattr(exc, "read") else ""
            return ProviderResponse(
                provider=self.name, model=self.model, reason=f"HTTP {exc.code}: {detail}"
            )
        except Exception as exc:
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason=f"{type(exc).__name__}: {exc}",
            )

        choices = body.get("choices", [])
        if not choices:
            return ProviderResponse(
                provider=self.name, model=self.model, reason="no choice returned"
            )
        text = choices[0].get("message", {}).get("content", "") or ""
        usage = body.get("usage", {})
        return ProviderResponse(
            text=text, parsed=_safe_json(text) if schema else None,
            provider=self.name, model=self.model, ok=bool(text),
            reason="" if text else "empty response",
            usage={
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
        )


# --------------------------------------------------------------------------
# Local / self-hosted
# --------------------------------------------------------------------------

class LocalProvider(LLMProvider):
    """An OpenAI-compatible endpoint on your own infrastructure.

    The only configuration under which nothing leaves your network. The egress
    guard still runs, because a "local" endpoint pointed at a public URL by
    misconfiguration would otherwise bypass every control.
    """

    name: ProviderName = "local"
    default_model = "local-model"
    key_env = "DIVE_LOCAL_LLM_KEY"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        super().__init__(model)
        import os

        self.base_url = base_url or os.environ.get("DIVE_LOCAL_LLM_URL", "http://localhost:8000/v1")

    def available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "DIVE_LOCAL_LLM_URL is not set"
        return True, ""

    def _sdk_available(self) -> tuple[bool, str]:
        return True, ""

    def complete(
        self, system: str, user: str, schema: dict | None = None, max_tokens: int = 16000
    ) -> ProviderResponse:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        token = self.secret.resolve()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(), headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as handle:
                body = json.loads(handle.read().decode())
        except Exception as exc:
            return ProviderResponse(
                provider=self.name, model=self.model,
                reason=f"{type(exc).__name__}: {exc}",
            )

        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ProviderResponse(
            text=text, parsed=_safe_json(text) if schema else None,
            provider=self.name, model=self.model, ok=bool(text),
            reason="" if text else "empty response",
            usage=body.get("usage", {}),
        )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": ClaudeProvider,
    "google": GeminiProvider,
    "openai": OpenAIProvider,
    "local": LocalProvider,
}


def available_providers() -> list[dict[str, Any]]:
    """Describe every provider and whether it is usable right now."""
    return [PROVIDERS[name]().describe() for name in PROVIDERS]


class ProviderRouter:
    """Routes a request to the first available provider, through the guard.

    Every provider call in Dive goes through :meth:`complete`. There is no
    other path to an external model, which is what makes the audit log a
    complete record rather than a sample.
    """

    def __init__(
        self,
        order: Sequence[ProviderName] = ("anthropic", "google", "openai"),
        guard: EgressGuard | None = None,
        settings: Settings | None = None,
        models: dict[str, str] | None = None,
    ):
        self.cfg = settings or SETTINGS
        self.order = list(order)
        self.models = models or {}
        self.guard = guard or EgressGuard(
            self.cfg.privacy, audit_path=self.cfg.paths.data_dir / "audit.jsonl"
        )

    def _build(self, name: str) -> LLMProvider:
        return PROVIDERS[name](model=self.models.get(name))

    def complete(
        self,
        system: str,
        user: str,
        schema: dict | None = None,
        max_tokens: int = 16000,
    ) -> ProviderResponse:
        """Send a prompt, after the guard has cleared it for that destination."""
        attempts: list[str] = []

        for name in self.order:
            if name not in PROVIDERS:
                attempts.append(f"{name}: unknown provider")
                continue

            # Re-reviewed per destination: a decision for one provider does not
            # authorise sending the same bytes to another.
            decision = self.guard.review_text(user, destination=name)
            if not decision.allowed:
                attempts.append(f"{name}: blocked by privacy guard - {'; '.join(decision.reasons)}")
                continue

            provider = self._build(name)
            ok, reason = provider.available()
            if not ok:
                attempts.append(f"{name}: {reason}")
                continue

            response = provider.complete(system, user, schema, max_tokens)
            if response.ok:
                return response
            attempts.append(f"{name}: {response.reason}")

        blocked = any("blocked by privacy guard" in a for a in attempts)
        return ProviderResponse(
            provider="none", ok=False, blocked_by_guard=blocked,
            reason="no provider could serve this request. " + " | ".join(attempts),
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _safe_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _to_gemini_schema(schema: dict) -> dict:
    """Translate a JSON Schema into Gemini's responseSchema dialect.

    Gemini uses uppercase type names and rejects ``additionalProperties``.
    """
    type_map = {
        "object": "OBJECT", "array": "ARRAY", "string": "STRING",
        "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN",
    }

    def convert(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(value, str):
                out["type"] = type_map.get(value, value.upper())
            elif key == "properties" and isinstance(value, dict):
                out["properties"] = {k: convert(v) for k, v in value.items()}
            elif key == "items":
                out["items"] = convert(value)
            else:
                out[key] = value
        return out

    return convert(schema)

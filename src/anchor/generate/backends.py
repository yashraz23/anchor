"""Generation backends.

Two paths behind one protocol: the Anthropic API, and an OpenAI-compatible
server for the local vLLM path in phase 3. Making vLLM a backend rather than a
separate script is the point of that phase: it turns vLLM into something used,
not merely profiled.

Every call returns its token usage, so cost and latency are recorded from the
first run rather than estimated later.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from anchor.config import GenerateSettings, Settings, cost_usd

logger = logging.getLogger(__name__)

# Generous on purpose: a long answer at high effort is slow, not hung. The
# Anthropic SDK carries its own default, so this applies to the vLLM path only.
_REQUEST_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class Completion:
    """One generation, with everything needed to price and audit it."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    stop_reason: str | None = None

    @property
    def cost_usd(self) -> float | None:
        """None when the model has no price on file, never 0.0."""
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


class Backend(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...


class AnthropicBackend:
    """Claude via the official SDK."""

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self.cfg: GenerateSettings = settings.generate
        key = settings.anthropic_api_key
        # The SDK resolves credentials from the environment on its own, so an
        # unset key here is not an error: it may still find one.
        self._client = (
            anthropic.Anthropic(api_key=key.get_secret_value())
            if key is not None
            else anthropic.Anthropic()
        )

    def complete(self, system: str, user: str) -> Completion:
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self.cfg.anthropic_model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Adaptive thinking, with depth controlled by effort. Note there is
            # no temperature: Opus 5 rejects sampling parameters outright, so
            # run-to-run stability comes from a fixed prompt and a fixed effort
            # level instead.
            thinking={"type": "adaptive", "display": self.cfg.thinking_display},
            output_config={"effort": self.cfg.effort},
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        # stop_reason has to be checked before reading content: a refusal
        # returns HTTP 200 with no usable text.
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            model=self.cfg.anthropic_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
        )


class VLLMBackend:
    """A local vLLM server over its OpenAI-compatible API.

    Kept deliberately thin. It exists so the same prompts and the same grounding
    checks can run against a self-hosted model, which is what makes the phase 3
    cost and latency comparison like-for-like.
    """

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.generate

    def complete(self, system: str, user: str) -> Completion:
        import httpx

        started = time.perf_counter()
        response = httpx.post(
            f"{self.cfg.vllm_base_url.rstrip('/')}/chat/completions",
            json={
                "model": self.cfg.vllm_model,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.vllm_temperature,
                "seed": self.cfg.vllm_seed,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = payload.get("usage", {})
        return Completion(
            text=payload["choices"][0]["message"]["content"] or "",
            model=self.cfg.vllm_model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            stop_reason=payload["choices"][0].get("finish_reason"),
        )


def make_backend(settings: Settings) -> Backend:
    """The configured backend."""
    from anchor.config import GeneratorBackend

    if settings.generate.backend is GeneratorBackend.VLLM:
        return VLLMBackend(settings)
    return AnthropicBackend(settings)

"""LLM observability with Langfuse.

Every model call in this project goes through one of three places: the
generation backend, the grounding judge, and the triad scorer. Each of the three
already measures its own latency and reads its own token usage, because those
were needed for costing before any tracer existed. This module is where those
measurements get published, so nothing is timed twice and the trace can never
disagree with the cost table.

A generation is recorded after the call returns rather than around it. That
loses the ability to trace an exception, which is a real cost, but it buys
something worth more here: the recorded numbers are the same objects the
database stores, not a second measurement of the same event.

Without `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` this disables itself.
Unlike W&B there is no offline mode to fall back to — Langfuse is a server or
nothing — so it stays quiet rather than pretending to have traced something.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from anchor.config import Settings

logger = logging.getLogger(__name__)

# Built once per process. The client owns a background exporter thread, and one
# per call would make the tracer more expensive than the calls it traces.
_client: Any = None
_resolved = False


def _tracer(settings: Settings) -> Any:
    """The process-wide Langfuse client, or None when it is not configured."""
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True

    public = settings.langfuse_public_key
    secret = settings.langfuse_secret_key
    if not settings.track.use_langfuse or public is None or secret is None:
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public.get_secret_value(),
            secret_key=secret.get_secret_value(),
            host=settings.langfuse_host,
            release=settings.generate.anthropic_model,
        )
    except Exception as exc:
        logger.warning("langfuse init failed (%s); continuing untraced", exc)
        _client = None
    return _client


@contextlib.contextmanager
def trace_session(
    settings: Settings, name: str, metadata: dict[str, Any] | None = None
) -> Iterator[None]:
    """Group the calls made inside this block under one trace.

    Used to put a whole golden-set run, or a whole grounding pass, under a single
    parent rather than scattering several hundred unrelated generations across
    the timeline.
    """
    client = _tracer(settings)
    if client is None:
        yield
        return

    try:
        with client.start_as_current_observation(name=name, as_type="chain", metadata=metadata):
            yield
    except Exception as exc:
        logger.warning("langfuse session %r failed (%s); the work itself is unaffected", name, exc)
        yield
    finally:
        with contextlib.suppress(Exception):
            client.flush()


def record_generation(
    settings: Settings,
    *,
    name: str,
    model: str,
    prompt: str,
    output: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    cost_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Publish one completed model call.

    `cost_usd` is None when the model has no price on file. It is omitted rather
    than sent as zero, so an unpriced model shows as missing in Langfuse exactly
    as it does in the results table.
    """
    client = _tracer(settings)
    if client is None:
        return

    try:
        observation = client.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=prompt,
            output=output,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            cost_details=({"total": cost_usd} if cost_usd is not None else None),
            metadata={"latency_ms": latency_ms, **(metadata or {})},
        )
        observation.end()
    except Exception as exc:
        # Observability is never allowed to break the thing it observes.
        logger.warning("langfuse failed to record %r: %s", name, exc)


def flush_traces(settings: Settings) -> None:
    """Block until queued observations are sent.

    Needed before the process exits: the exporter is asynchronous, and a CLI
    command that returns immediately would drop the tail of its own run.
    """
    client = _tracer(settings)
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.flush()

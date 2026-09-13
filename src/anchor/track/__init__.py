"""Experiment tracking and LLM observability.

Two separate concerns behind two separate modules, deliberately not merged:

`experiment` (Weights & Biases) records *runs* — a configuration, the metrics it
produced, and the git SHA it was produced at. Its unit is one sweep.

`tracing` (Langfuse) records *calls* — one model invocation, its prompt, its
output, its tokens, its latency, its price. Its unit is one request.

Both are strictly observers. Neither may change a result, and neither may fail a
run: every entry point here swallows its own exceptions and logs a warning,
because a telemetry outage that kills a paid evaluation would be a worse bug
than the missing telemetry.

They differ in how they behave without credentials, and the difference is not an
oversight. W&B falls back to offline mode, which is a real run written to a
local directory and syncable later, so the integration is demonstrable without
an account. Langfuse has no offline equivalent — it is a server or nothing — so
it disables itself rather than pretend.
"""

from anchor.track.experiment import experiment, log_metrics, log_table
from anchor.track.tracing import flush_traces, record_generation, trace_session

__all__ = [
    "experiment",
    "flush_traces",
    "log_metrics",
    "log_table",
    "record_generation",
    "trace_session",
]

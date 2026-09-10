"""Gradio demo.

Shows the grounding, not just the answer. A demo that printed prose would look
like every other RAG chatbot and would hide the only thing about this one worth
looking at: which spans the answer used, how many of its claims are supported,
and why it sometimes refuses to answer at all.

Runs on Hugging Face Spaces, which has no Postgres, so the corpus must be
reachable over the network. `ANCHOR_PG_DSN` points at it.
"""

from __future__ import annotations

import logging

import gradio as gr

from anchor.config import get_settings
from anchor.ground.service import GroundedAnswer, grounded_answer
from anchor.index.embed import Embedder

logger = logging.getLogger(__name__)

EXAMPLES = [
    "What do max_num_seqs and max_model_len do?",
    "what is enforce_eager",
    "How do I use tensorizer to load models from S3?",
    "How do I limit the thinking budget for reasoning mode?",
    "How do I disable logging?",
]

_settings = get_settings()
_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    """Load the encoder once. Loading per request would dominate latency."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(
            _settings.index.embedding_model,
            normalize=_settings.index.normalize_embeddings,
            query_instruction=_settings.index.query_instruction,
        )
    return _embedder


def format_grounding(result: GroundedAnswer) -> str:
    """The evidence panel: what the answer rests on."""
    lines: list[str] = []

    if result.withheld:
        lines.append(f"**Answer withheld.** {result.reason}")
        lines.append("")
    elif result.support is not None:
        lines.append(
            f"**{result.supported_claims} of {result.claim_count} claims** "
            f"supported by their cited span ({result.support:.0%})."
        )
        lines.append("")

    completion = result.answer.completion
    if completion is not None:
        cost = "unpriced" if completion.cost_usd is None else f"${completion.cost_usd:.4f}"
        lines.append(f"`{completion.latency_ms} ms` · `{cost}` · ")
        lines.append(f"`{completion.input_tokens} in / {completion.output_tokens} out`")
        lines.append("")

    cited = result.answer.cited
    lines.append("### Retrieved spans")
    for i, hit in enumerate(result.spans, start=1):
        mark = "**cited**" if i in cited else "not cited"
        where = hit.heading_path or hit.source_path
        link = f"[{hit.source_path}]({hit.url})" if hit.url else f"`{hit.source_path}`"
        lines.append(f"**[{i}]** {mark} · {link} · `{hit.vllm_version}`")
        lines.append(f"> {where}")
        body = hit.text if len(hit.text) <= 400 else hit.text[:400] + " …"
        lines.append("")
        lines.append("```")
        lines.append(body)
        lines.append("```")
    return "\n".join(lines)


def answer(question: str) -> tuple[str, str]:
    """Answer a question and render the grounding beside it."""
    if not question.strip():
        return "", "Ask something about vLLM."
    try:
        result = grounded_answer(question, _settings, embedder=_get_embedder())
    except Exception as exc:
        logger.exception("failed to answer")
        # Surfaced rather than swallowed: a demo that silently returns nothing
        # is indistinguishable from a demo that is broken.
        return "", f"**Error.** {exc}"
    return result.text, format_grounding(result)


def build() -> gr.Blocks:
    with gr.Blocks(title="anchor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# anchor\n"
            "Retrieval-augmented generation over vLLM documentation, with a "
            "grounding verification layer. Every claim is traced to the span it "
            "cites; claims naming CLI flags or config keys are checked against "
            "vLLM's actual source. Answers that cannot be grounded are withheld "
            "rather than guessed."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="What do max_num_seqs and max_model_len do?",
                scale=4,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(examples=EXAMPLES, inputs=question)

        with gr.Row():
            answer_box = gr.Markdown(label="Answer")
            grounding_box = gr.Markdown(label="Grounding")

        # Gradio's event methods are attached dynamically and are missing
        # from its type annotations, so these three are asserted by hand.
        submit.click(  # type: ignore[attr-defined]
            answer, inputs=question, outputs=[answer_box, grounding_box]
        )
        question.submit(  # type: ignore[attr-defined]
            answer, inputs=question, outputs=[answer_box, grounding_box]
        )
    built: gr.Blocks = demo
    return built


if __name__ == "__main__":
    build().launch()

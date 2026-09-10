"""Render the tradeoff curve as a PNG for the README.

Reads the numbers from the database rather than taking them as arguments, so the
chart cannot drift from the table beside it. Regenerate with:

    uv run python scripts/plot_tradeoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anchor.config import get_settings
from anchor.ground.ground_run import ground_run

OUT = Path(__file__).resolve().parents[1] / "docs" / "tradeoff.png"


def main() -> int:
    settings = get_settings()
    # The judge costs money and moves no point on this curve, which thresholds
    # on attribution. Skipped so regenerating the chart is free.
    settings.ground.use_llm_judge = False
    report = ground_run(settings)

    points = report.points
    thresholds = [p.threshold for p in points]
    answer_rate = [(p.answer_rate or 0.0) * 100 for p in points]
    uncited = [(p.uncited_rate or 0.0) * 100 for p in points]
    unsupported = [(p.unsupported_rate or 0.0) * 100 for p in points]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.plot(thresholds, answer_rate, marker="o", lw=2, label="Answered (%)")
    ax.plot(thresholds, uncited, marker="s", lw=2, label="Delivered claims uncited (%)")
    ax.plot(
        thresholds,
        unsupported,
        marker="^",
        lw=2,
        label="Delivered claims cited but unsupported (%)",
    )

    ax.set_xlabel("Abstention threshold (share of claims that must be supported)")
    ax.set_ylabel("Percent")
    ax.set_title(
        f"Strictness versus completeness  ({report.answers} questions, {report.claims} claims)"
    )
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

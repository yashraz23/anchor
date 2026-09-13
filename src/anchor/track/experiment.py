"""Experiment tracking with Weights & Biases.

One W&B run per evaluation: the retrieval sweep, or the grounding sweep. What
gets logged is exactly what `runs.config_json` already holds plus the metrics
that came out of it, so the tracker and the database never disagree about what
produced a number.

Credentials are optional by design. With `WANDB_API_KEY` set the run syncs to
the cloud; without it the run is written offline to `data/wandb/` and can be
pushed later with `wandb sync`. Offline is a real run, not a stub — the same
config, the same tables, the same summary — which is what makes this
integration checkable by someone who has no account.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from typing import Any, Literal

from anchor.config import REPO_ROOT, Settings

logger = logging.getLogger(__name__)

# W&B writes run directories here rather than into the repo root, so `wandb/`
# does not appear next to `src/`. Offline runs are kept: they are the artefact
# you sync once an account exists.
_WANDB_DIR = REPO_ROOT / "data" / "wandb"


class _NullRun:
    """Stands in for a W&B run when tracking is off.

    Every logging call in this module goes through the same object whether or
    not W&B is live, so call sites never branch on whether tracking is enabled.
    """

    disabled = True

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@contextlib.contextmanager
def experiment(
    settings: Settings, job: str, config_extra: dict[str, object] | None = None
) -> Iterator[Any]:
    """Open a W&B run for one evaluation, or a no-op when tracking is off.

    `job` names the kind of evaluation ("retrieval-sweep", "grounding-sweep") and
    becomes the run's job type, so the two sweeps stay separable in the UI
    without needing two projects.
    """
    if not settings.track.use_wandb:
        yield _NullRun()
        return

    try:
        import wandb
    except ImportError:  # pragma: no cover - wandb is in the eval extra
        logger.warning("wandb not installed; skipping experiment tracking")
        yield _NullRun()
        return

    key = settings.wandb_api_key
    mode: Literal["online", "offline"] = "online" if key is not None else "offline"
    if mode == "offline":
        logger.info(
            "no WANDB_API_KEY; logging offline to %s (sync later with `wandb sync`)", _WANDB_DIR
        )

    config: dict[str, object] = dict(settings.run_config())
    config.update(config_extra or {})

    run = None
    try:
        _WANDB_DIR.mkdir(parents=True, exist_ok=True)
        if key is not None:
            # Authenticate explicitly rather than leaning on the ambient
            # environment: the key lives in Settings, and wandb.init has no
            # api_key parameter of its own.
            wandb.login(key=key.get_secret_value(), verify=False)
        run = wandb.init(
            project=settings.track.wandb_project,
            entity=settings.track.wandb_entity or None,
            job_type=job,
            dir=str(_WANDB_DIR),
            config=config,
            # mode belongs inside the Settings object, not beside it. Passing
            # `settings=` alongside a top-level `mode=` silently discards the
            # latter, and the offline fallback then fails asking for an API key
            # -- which is exactly the failure this fallback exists to avoid.
            settings=wandb.Settings(quiet=True, mode=mode),
        )
    except Exception as exc:
        # A tracking failure must not take an evaluation with it. Some of these
        # runs cost real money to produce.
        logger.warning("wandb init failed (%s); continuing untracked", exc)
        yield _NullRun()
        return

    try:
        yield run
    finally:
        with contextlib.suppress(Exception):
            run.finish()


def log_metrics(run: Any, metrics: dict[str, float | int | None]) -> None:
    """Send scalars to the run summary, dropping the ones that are missing.

    None means a metric could not be computed. It is dropped rather than sent as
    zero, for the same reason `cost_usd` returns None: a missing measurement
    that renders as 0.0 is a fabricated one.
    """
    if getattr(run, "disabled", False):
        return
    present = {k: v for k, v in metrics.items() if v is not None}
    with contextlib.suppress(Exception):
        run.summary.update(present)


def log_table(run: Any, name: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Log one table. Used for the recall grid and the tradeoff curve."""
    if getattr(run, "disabled", False):
        return
    try:
        import wandb

        run.log({name: wandb.Table(columns=list(columns), data=[list(r) for r in rows])})
    except Exception as exc:
        logger.warning("wandb table %r failed to log: %s", name, exc)


def log_tradeoff_curve(run: Any, points: Sequence[Any]) -> None:
    """Log the headline curve: answer rate against the two failure rates.

    The two failure modes are logged as separate series rather than summed.
    Folding them together is exactly the mistake that put the reported
    unsupported rate at 37% when the strict figure was under 1%.
    """
    if getattr(run, "disabled", False):
        return
    log_table(
        run,
        "tradeoff_curve",
        (
            "threshold",
            "answered",
            "abstained",
            "answer_rate",
            "delivered_claims",
            "unsupported_delivered",
            "unsupported_rate",
            "uncited_delivered",
            "uncited_rate",
            "contradicted_delivered",
        ),
        [
            (
                p.threshold,
                p.answered,
                p.abstained,
                p.answer_rate,
                p.delivered_claims,
                p.unsupported_delivered,
                p.unsupported_rate,
                p.uncited_delivered,
                p.uncited_rate,
                p.contradicted_delivered,
            )
            for p in points
        ],
    )

    # A step-indexed series as well as the table, so the curve is plottable in
    # the UI without the reader building a chart by hand.
    for p in points:
        with contextlib.suppress(Exception):
            run.log(
                {
                    "threshold": p.threshold,
                    "curve/answer_rate": p.answer_rate,
                    "curve/unsupported_rate": p.unsupported_rate,
                    "curve/uncited_rate": p.uncited_rate,
                }
            )

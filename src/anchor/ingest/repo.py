"""Acquire the vLLM source at a pinned commit.

Every document in the corpus records the commit it came from. That is what makes
staleness detection possible later: a chunk knows which version of vLLM it
describes, so retrieved guidance can be compared against the version the user is
actually running.

The clone is a blobless partial clone rather than a shallow one. Shallow clones
cannot resolve an arbitrary older commit without a second fetch, and the symbol
oracle needs the full source tree at exactly the pinned commit.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from anchor.config import IngestSettings


class RepoError(RuntimeError):
    """Raised when the vLLM checkout cannot be prepared."""


@dataclass(frozen=True)
class RepoSnapshot:
    """A pinned checkout. Everything ingested inherits these two fields."""

    root: Path
    commit_sha: str
    vllm_version: str
    repo_url: str

    def blob_url(self, source_path: str) -> str:
        """A permanent link to one file at this exact commit.

        Deliberately a GitHub blob URL rather than a docs-site URL: the mapping
        from a docs source path to its rendered page is guesswork, while this is
        exact, verifiable, and does not rot when the site is restructured.
        """
        base = self.repo_url.removesuffix(".git")
        return f"{base}/blob/{self.commit_sha}/{source_path}"


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run git and return stdout, raising RepoError with git's own message."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RepoError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RepoError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return result.stdout.strip()


_VERSION_RE = re.compile(r"^__version__\s*[:=]\s*[\"']([^\"']+)[\"']", re.MULTILINE)


def read_version(root: Path) -> str:
    """The vLLM version as recorded in the source tree, or 'unknown'.

    This is the fallback path. In a plain source checkout it usually finds
    nothing: vLLM's `vllm/version.py` imports from a `_version.py` that
    setuptools_scm generates at build time and does not commit, and its
    hardcoded fallback is the literal string "dev", which is useless for
    staleness comparison. Kept for clones fetched without tags.
    """
    for candidate in (
        root / "vllm" / "_version.py",
        root / "vllm" / "version.py",
        root / "vllm" / "__init__.py",
    ):
        if not candidate.is_file():
            continue
        match = _VERSION_RE.search(candidate.read_text(encoding="utf-8", errors="replace"))
        if match and match.group(1) != "dev":
            return match.group(1)
    return "unknown"


def resolve_version(root: Path) -> str:
    """The release this checkout belongs to.

    The nearest reachable tag, not `git describe`'s full form. A chunk needs a
    version a user can compare against the vLLM they have installed, and
    "v0.28.1rc0" answers that where "v0.28.1rc0-619-g65f3fca56" does not. Exact
    identity is not lost: `commit_sha` on the same row is precise.
    """
    try:
        tag = _git("describe", "--tags", "--abbrev=0", cwd=root)
    except RepoError:
        return read_version(root)
    return tag or read_version(root)


def prepare_checkout(settings: IngestSettings, commit: str | None = None) -> RepoSnapshot:
    """Clone or update the vLLM checkout and pin it to a commit.

    With no commit from either `commit` or `vllm_commit`, the tip of
    `vllm_branch` is resolved and pinned to whatever it is right now, and the
    resolved SHA is returned so the caller can reproduce the run later.

    Every stage after ingest must pass `commit` explicitly, with the value the
    corpus was ingested at. vLLM's main branch moves several times a day, so a
    later stage that re-resolves the tip silently checks out a different tree
    than the one `documents` describes, and every path lookup then misses.
    """
    if settings.local_source is not None:
        # A local corpus has no git history. The identity that matters
        # downstream is "which tree was this", so the path stands in for the
        # commit and is recorded on every document exactly as a SHA would be.
        root = settings.local_source.resolve()
        if not root.is_dir():
            raise RepoError(f"ingest.local_source is not a directory: {root}")
        digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()
        return RepoSnapshot(
            root=root,
            commit_sha=f"local-{digest[:32]}",
            vllm_version=resolve_version(root),
            repo_url=settings.vllm_repo_url,
        )

    root = settings.checkout_dir
    root.parent.mkdir(parents=True, exist_ok=True)

    if not (root / ".git").is_dir():
        _git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            settings.vllm_repo_url,
            str(root),
        )

    pinned = commit or settings.vllm_commit
    if pinned:
        target = pinned
        # An explicitly pinned commit may predate the local clone's last fetch.
        _git("fetch", "--filter=blob:none", "origin", target, cwd=root)
    else:
        _git("fetch", "--filter=blob:none", "origin", settings.vllm_branch, cwd=root)
        target = _git("rev-parse", "FETCH_HEAD", cwd=root)

    # Detached on purpose. This checkout is a read-only artifact, never a branch
    # anyone commits to.
    _git("checkout", "--detach", "--force", target, cwd=root)
    commit_sha = _git("rev-parse", "HEAD", cwd=root)

    return RepoSnapshot(
        root=root,
        commit_sha=commit_sha,
        vllm_version=resolve_version(root),
        repo_url=settings.vllm_repo_url,
    )

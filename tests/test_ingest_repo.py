"""Repo-snapshot tests. No network: version reading and URL construction are
pure functions over a temporary tree.
"""

from __future__ import annotations

from pathlib import Path

from anchor.ingest.repo import RepoSnapshot, read_version, resolve_version

SHA = "0123456789abcdef0123456789abcdef01234567"


def _snapshot(tmp_path: Path, version: str = "0.6.3") -> RepoSnapshot:
    return RepoSnapshot(
        root=tmp_path,
        commit_sha=SHA,
        vllm_version=version,
        repo_url="https://github.com/vllm-project/vllm.git",
    )


def test_blob_url_points_at_the_pinned_commit(tmp_path: Path) -> None:
    """The URL must pin the commit, not a branch, or it rots as docs move."""
    url = _snapshot(tmp_path).blob_url("docs/source/serving/openai.md")
    assert url == (f"https://github.com/vllm-project/vllm/blob/{SHA}/docs/source/serving/openai.md")
    assert ".git/blob" not in url


def test_read_version_from_version_py(tmp_path: Path) -> None:
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    (pkg / "version.py").write_text('__version__ = "0.6.3.post1"\n', encoding="utf-8")
    assert read_version(tmp_path) == "0.6.3.post1"


def test_read_version_falls_back_to_init(tmp_path: Path) -> None:
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "0.7.0"\n', encoding="utf-8")
    assert read_version(tmp_path) == "0.7.0"


def test_read_version_is_unknown_rather_than_raising(tmp_path: Path) -> None:
    """A missing version must land in the data as unknown, not abort a run."""
    assert read_version(tmp_path) == "unknown"


def test_read_version_rejects_the_dev_placeholder(tmp_path: Path) -> None:
    """vLLM's committed fallback is the literal "dev".

    setuptools_scm writes the real value into _version.py at build time, so a
    plain source checkout only carries the placeholder. Recording "dev" as a
    version would put a value in every row that cannot be compared against
    anything, which is worse than recording that it is unknown.
    """
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    (pkg / "version.py").write_text(
        'try:\n    from ._version import __version__\nexcept Exception:\n    __version__ = "dev"\n',
        encoding="utf-8",
    )
    assert read_version(tmp_path) == "unknown"


def test_read_version_prefers_the_generated_file(tmp_path: Path) -> None:
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    (pkg / "_version.py").write_text('__version__ = "0.11.0"\n', encoding="utf-8")
    (pkg / "version.py").write_text('__version__ = "dev"\n', encoding="utf-8")
    assert read_version(tmp_path) == "0.11.0"


def test_resolve_version_falls_back_when_there_are_no_tags(tmp_path: Path) -> None:
    """An untagged directory is not a git repo, so describe fails and the
    filesystem path takes over."""
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    (pkg / "_version.py").write_text('__version__ = "0.11.0"\n', encoding="utf-8")
    assert resolve_version(tmp_path) == "0.11.0"

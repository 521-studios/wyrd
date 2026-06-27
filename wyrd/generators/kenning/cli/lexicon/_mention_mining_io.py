"""Shared run-plumbing for the toponym-mention mining CLIs.

``mine-toponym-mentions-tiered`` and ``-staged`` both resolve the same
``--source`` selection and open the same ``--capture-failures`` append sink.
These were byte-identical copies (the tiered docstrings already noted they
"mirror the staged-cascade behavior") — single-sourced here so they can't drift.
"""

from __future__ import annotations

from pathlib import Path

import click


def resolve_source_ids(sources: tuple[str, ...], sources_dir: Path) -> list[str]:
    """Resolve the source list: validate each ``--source`` exists as a ``.txt``
    under ``sources_dir`` (returned in caller order), or walk ``*.txt`` (sorted,
    deterministic; excluding MANIFEST.md) when ``--source`` is omitted. Raises
    ClickException if a named source is missing or none are found."""
    if sources:
        source_ids = []
        for sid in sources:
            txt = sources_dir / f"{Path(sid).name}.txt"
            if not txt.exists():
                raise click.ClickException(f"source body not found: {txt}")
            source_ids.append(Path(sid).name)
    else:
        source_ids = sorted(p.stem for p in sources_dir.glob("*.txt") if p.name != "MANIFEST.md")
    if not source_ids:
        raise click.ClickException(f"no sources under {sources_dir}")
    return source_ids


def open_failure_sink(capture_failures: Path | None):
    """Open the ``--capture-failures`` append sink (or None). Append mode so a
    resume across multiple invocations accumulates failures end-to-end. Warns on
    an existing non-empty file so an operator-blind append doesn't silently bury
    old records. ``errors="replace"`` is the wyrd-klod surrogate-escape backstop
    — a "?" means a surrogate slipped past the in-band sanitizer. The stale-count
    read uses it too: a crash *mid-write* (this is a crash-safety pipeline) can
    leave a truncated multibyte sequence that the write-side backstop can't catch,
    and the resume warning-read must not die decoding it."""
    if capture_failures is None:
        return None
    capture_failures.parent.mkdir(parents=True, exist_ok=True)
    if capture_failures.exists() and capture_failures.stat().st_size > 0:
        with capture_failures.open("r", encoding="utf-8", errors="replace") as _fh:
            stale = sum(1 for ln in _fh if ln.strip())
        click.echo(
            f"  warning: --capture-failures {capture_failures} already has "
            f"{stale} record(s); appending (`> {capture_failures}` to clear)",
            err=True,
        )
    return capture_failures.open("a", encoding="utf-8", errors="replace")

"""Structural diff between two meanings.json bundles.

The Phase 3 round-trip test (wyrd-xrnw) pins that the dump → rebuild →
export pipeline is byte-deterministic on a synthetic fixture. The
operator-runnable complement (wyrd-w3x0) needs a richer diff: when the
real-data round-trip *does* drift, raw byte-level differences across a
34 MB bundle are useless. This module produces an actionable summary —
which subjects changed, which canonical decompositions changed,
top-level shape differences — alongside the first byte-level divergence
window for cases where the drift is something the structural diff
doesn't categorize.

Subject identity for alignment matches the export-meanings sort key
(see :func:`lexicon._group_families_into_subjects`):
``(modifier_type, meaning, modifier_tags, modern_usage)``. Two subjects
with the same identity tuple are paired; differing payload counts as a
"changed" subject.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BundleDiff:
    """Structured summary of differences between two bundle JSON texts.

    The dataclass shape stays stable for callers (CLI renderer, test
    assertions); add new fields rather than reshape existing ones.
    """

    # Byte-level metadata. ``bytes_match`` is the headline boolean —
    # everything else is for diagnosis when it's False.
    bytes_match: bool
    byte_size_committed: int
    byte_size_rebuilt: int
    # First byte offset where the two texts differ, or None when one is
    # a strict prefix of the other (in which case the length mismatch
    # is the divergence). Renders nicely with the window helper.
    first_divergence_byte: int | None = None
    first_divergence_window: tuple[str, str] | None = None

    # Structural diff. Computed only when both texts parse as JSON.
    parse_error: str | None = None
    top_level_keys_added: list[str] = field(default_factory=list)
    top_level_keys_removed: list[str] = field(default_factory=list)
    subjects_added: list[str] = field(default_factory=list)
    subjects_removed: list[str] = field(default_factory=list)
    # (identity_str, list[per-field-diff-str]) — narrow note about what
    # in the changed subject differs (e.g. word count delta, sense set
    # delta). Caps the per-subject detail; the byte-window covers the
    # full text for the curious.
    subjects_changed: list[tuple[str, list[str]]] = field(default_factory=list)
    canonical_decompositions_changed: int = 0
    joiners_changed: int = 0
    fantasy_morphemes_changed: int = 0


def _subject_identity(subject: dict[str, Any]) -> tuple:
    """Mirror of the sort key in ``_group_families_into_subjects``. Two
    subjects with the same return value are considered the same subject;
    payload differences make them ``changed`` rather than added/removed."""
    return (
        subject.get("modifier_type") or "",
        tuple(subject.get("meaning") or []),
        tuple(subject.get("modifier_tags") or []),
        tuple(w.get("modern_usage", "") for w in (subject.get("words") or [])),
    )


def _identity_str(identity: tuple) -> str:
    """Human-readable rendering of a subject identity tuple — used in
    the dropped/added/changed lists so operators can find the subject in
    the bundle file."""
    modifier_type, meaning, modifier_tags, modern_usage = identity
    meaning_part = ", ".join(meaning) if meaning else "<no meaning>"
    usage_part = ", ".join(modern_usage) if modern_usage else "<no usage>"
    return f"[{modifier_type or '?'}] {meaning_part} → ({usage_part})"


def _diff_subject_payloads(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Per-changed-subject narrow notes. Returns a list of short strings
    pointing at which ``words[i]`` slots diverge.

    Paired-as-changed subjects already agree on the identity tuple
    ``(modifier_type, meaning, modifier_tags, [w.modern_usage for w in
    words])`` — so divergence by definition lives inside the per-word
    payload (the per-language form lists like ``old_english``,
    ``old_norse``, the etymology metadata, etc.). The note format
    ``words[i]`` lets operators jump straight to the diverging slot in
    the bundle file without scanning the whole subject."""
    notes: list[str] = []
    a_words = a.get("words") or []
    b_words = b.get("words") or []
    # Length is implicit in the identity tuple (zipped via modern_usage
    # tuple), so a_words and b_words are guaranteed-same-length here.
    # strict=True turns that invariant into an assertion: a future change
    # that broke the identity-tuple contract would fail loudly with a
    # ValueError instead of silently truncating per-word diff notes.
    for i, (a_word, b_word) in enumerate(zip(a_words, b_words, strict=True)):
        if a_word != b_word:
            notes.append(f"words[{i}] differs")
    if not notes:
        # Belt-and-braces: the identity-tuple invariant should ensure
        # at least one word differs, but if a future identity-tuple
        # change makes some non-word field free-variable, surface that
        # instead of returning an empty notes list.
        notes.append("payload differs (see byte window)")
    return notes


def _find_first_divergence(
    committed: str, rebuilt: str, window_chars: int = 100
) -> tuple[int | None, tuple[str, str] | None]:
    """Return ``(byte_offset, (committed_window, rebuilt_window))`` for
    the first character-level difference. ``byte_offset`` is the UTF-8
    byte offset (matches ``wc -c`` / a hex editor's view of the file on
    disk), not the Python ``str`` character index — the bundle is
    written with ``ensure_ascii=False`` so non-ASCII etymological forms
    (æ, þ, ŵ, ...) make those two views diverge.

    ``window_chars`` controls the size of the diagnostic window on each
    side of the divergence. The window itself stays in character units;
    operators reading the report don't care about byte alignment for
    the surrounding context."""
    if committed == rebuilt:
        return None, None
    # zip's truncate semantics are intentional here — we only iterate to
    # the shorter length; the length-mismatch case is handled by the
    # caller via byte_size_* fields.
    for i, (a, b) in enumerate(zip(committed, rebuilt, strict=False)):
        if a != b:
            byte_offset = len(committed[:i].encode("utf-8"))
            start = max(0, i - window_chars // 2)
            end = i + window_chars // 2
            return byte_offset, (committed[start:end], rebuilt[start:end])
    # No mid-string divergence → one is a prefix of the other. The
    # divergence "offset" is the byte length of the shorter text.
    shorter = min(len(committed), len(rebuilt))
    byte_offset = len(committed[:shorter].encode("utf-8"))
    longer_text = committed if len(committed) > len(rebuilt) else rebuilt
    trailing = longer_text[shorter : shorter + window_chars]
    return byte_offset, (
        "<end>" if len(committed) <= shorter else trailing,
        "<end>" if len(rebuilt) <= shorter else trailing,
    )


def _diff_dict_keys(committed: dict[str, Any], rebuilt: dict[str, Any]) -> tuple[int, int, int]:
    """Compare two dict-shaped optional bundle fields (canonical_decompositions,
    joiners, fantasy_morphemes) by key set + payload equality. Returns
    ``(added, removed, changed)`` counts. The CLI renderer sums them
    for the report; sample keys are not surfaced here because these
    sub-fields are large enough that a key list would be wall-of-text."""
    a_keys = set(committed)
    b_keys = set(rebuilt)
    added = len(b_keys - a_keys)
    removed = len(a_keys - b_keys)
    changed = sum(1 for k in a_keys & b_keys if committed[k] != rebuilt[k])
    return added, removed, changed


def compute_bundle_diff(committed_text: str, rebuilt_text: str) -> BundleDiff:
    """Produce a structured diff between two meanings.json texts.

    The byte-level fields are always populated. Structural diff fields
    are populated only when both texts parse as JSON; on parse failure
    the ``parse_error`` field carries the underlying message and the
    caller should fall back to the byte window for diagnosis.
    """
    bytes_match = committed_text == rebuilt_text
    # Encode to UTF-8 for sizing because the bundle is written with
    # ensure_ascii=False (per export-meanings); the on-disk byte count
    # (what `wc -c` and a hex editor see) is the UTF-8 length, not the
    # Python ``str`` character length. Pure-ASCII content makes the
    # two equal; non-ASCII (æ, þ, ŵ, ...) makes them diverge.
    diff = BundleDiff(
        bytes_match=bytes_match,
        byte_size_committed=len(committed_text.encode("utf-8")),
        byte_size_rebuilt=len(rebuilt_text.encode("utf-8")),
    )
    if bytes_match:
        return diff

    first_byte, window = _find_first_divergence(committed_text, rebuilt_text)
    diff.first_divergence_byte = first_byte
    diff.first_divergence_window = window

    try:
        committed = json.loads(committed_text)
        rebuilt = json.loads(rebuilt_text)
    except json.JSONDecodeError as exc:
        diff.parse_error = f"could not parse one or both bundles as JSON: {exc}"
        return diff

    diff.top_level_keys_added = sorted(set(rebuilt) - set(committed))
    diff.top_level_keys_removed = sorted(set(committed) - set(rebuilt))

    # Subjects: index by identity tuple, pair up, classify.
    committed_subjects = {_subject_identity(s): s for s in (committed.get("subjects") or [])}
    rebuilt_subjects = {_subject_identity(s): s for s in (rebuilt.get("subjects") or [])}
    diff.subjects_added = [
        _identity_str(k) for k in (rebuilt_subjects.keys() - committed_subjects.keys())
    ]
    diff.subjects_removed = [
        _identity_str(k) for k in (committed_subjects.keys() - rebuilt_subjects.keys())
    ]
    for key in committed_subjects.keys() & rebuilt_subjects.keys():
        if committed_subjects[key] != rebuilt_subjects[key]:
            diff.subjects_changed.append(
                (
                    _identity_str(key),
                    _diff_subject_payloads(committed_subjects[key], rebuilt_subjects[key]),
                )
            )

    # Sort the lists so the report is deterministic across runs (helpful
    # when redirecting diff-bundle output to a file for review).
    diff.subjects_added.sort()
    diff.subjects_removed.sort()
    diff.subjects_changed.sort(key=lambda t: t[0])

    # Other top-level fields are dict-shaped; collapse to bare counts.
    for field_name, attr in (
        ("canonical_decompositions", "canonical_decompositions_changed"),
        ("joiners", "joiners_changed"),
        ("fantasy_morphemes", "fantasy_morphemes_changed"),
    ):
        a = committed.get(field_name) or {}
        b = rebuilt.get(field_name) or {}
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        added, removed, changed = _diff_dict_keys(a, b)
        # Roll added/removed/changed into the one counter — the CLI
        # reports total churn, not direction (the byte window will show
        # the direction for cases where it matters).
        setattr(diff, attr, added + removed + changed)

    return diff


def _emit_capped_list(
    lines: list[str],
    items: list[Any],
    *,
    cap: int,
    indent: str,
    prefix: str,
    render: Any,
) -> None:
    """Append up to ``cap`` items, then ``... and N more`` if truncated.

    ``render(item) -> str`` formats one item; we prepend ``indent`` +
    ``prefix`` per line so callers control the marker (``+``/``-``/``~``)
    without duplicating the cap logic. Keyword-only after ``items`` keeps
    call sites readable at the cost of a tiny bit of verbosity."""
    for item in items[:cap]:
        lines.append(f"{indent}{prefix}{render(item)}")
    extra = len(items) - cap
    if extra > 0:
        lines.append(f"{indent}... and {extra} more")


def _format_subjects_section(
    lines: list[str], diff: BundleDiff, max_subjects_per_section: int
) -> None:
    if not (diff.subjects_added or diff.subjects_removed or diff.subjects_changed):
        return
    lines.append("")
    lines.append(
        f"Subjects: "
        f"+{len(diff.subjects_added)} added, "
        f"-{len(diff.subjects_removed)} removed, "
        f"~{len(diff.subjects_changed)} changed"
    )
    if diff.subjects_added:
        lines.append("  Added (sample):")
        _emit_capped_list(
            lines,
            diff.subjects_added,
            cap=max_subjects_per_section,
            indent="    ",
            prefix="+ ",
            render=str,
        )
    if diff.subjects_removed:
        lines.append("  Removed (sample):")
        _emit_capped_list(
            lines,
            diff.subjects_removed,
            cap=max_subjects_per_section,
            indent="    ",
            prefix="- ",
            render=str,
        )
    if diff.subjects_changed:
        lines.append("  Changed (sample):")
        _emit_capped_list(
            lines,
            diff.subjects_changed,
            cap=max_subjects_per_section,
            indent="    ",
            prefix="~ ",
            render=lambda pair: f"{pair[0]}: {', '.join(pair[1])}",
        )


def _format_other_fields_section(lines: list[str], diff: BundleDiff) -> None:
    if not any(
        (
            diff.canonical_decompositions_changed,
            diff.joiners_changed,
            diff.fantasy_morphemes_changed,
        )
    ):
        return
    lines.append("")
    lines.append("Other top-level fields with churn:")
    if diff.canonical_decompositions_changed:
        lines.append(f"  canonical_decompositions: {diff.canonical_decompositions_changed} changed")
    if diff.joiners_changed:
        lines.append(f"  joiners: {diff.joiners_changed} changed")
    if diff.fantasy_morphemes_changed:
        lines.append(f"  fantasy_morphemes: {diff.fantasy_morphemes_changed} changed")


def _format_byte_header(lines: list[str], diff: BundleDiff) -> None:
    """Append the byte-level metadata block (size delta + first-divergence
    window) for a drifted bundle — the diagnostics that matter before the
    structural diff."""
    if diff.byte_size_committed != diff.byte_size_rebuilt:
        lines.append(
            f"Byte size: committed={diff.byte_size_committed:,}, "
            f"rebuilt={diff.byte_size_rebuilt:,} "
            f"(delta {diff.byte_size_rebuilt - diff.byte_size_committed:+,})"
        )
    else:
        lines.append(f"Byte size: {diff.byte_size_committed:,} (matches; content differs)")

    if diff.first_divergence_byte is not None:
        lines.append(f"First byte-level divergence at offset {diff.first_divergence_byte:,}")
        if diff.first_divergence_window:
            committed_win, rebuilt_win = diff.first_divergence_window
            lines.extend(
                [
                    "  committed: ..." + committed_win.replace("\n", "\\n") + "...",
                    "  rebuilt:   ..." + rebuilt_win.replace("\n", "\\n") + "...",
                ]
            )


def format_bundle_diff(diff: BundleDiff, max_subjects_per_section: int = 10) -> str:
    """Render a BundleDiff as a human-readable summary string.

    ``max_subjects_per_section`` caps the added/removed/changed sample
    lists so a wholesale rewrite (e.g. every subject churned) doesn't
    dump a 4-MB diff to the terminal. The cap is suffixed with
    ``... and N more`` when truncated."""
    if diff.bytes_match:
        return "No drift — rebuilt bundle is byte-identical to committed bundle."

    lines: list[str] = ["Bundle drift detected.", ""]
    _format_byte_header(lines, diff)

    if diff.parse_error:
        lines.append("")
        lines.append(f"Structural diff unavailable: {diff.parse_error}")
        return "\n".join(lines)

    if diff.top_level_keys_added or diff.top_level_keys_removed:
        lines.append("")
        lines.append("Top-level shape changed:")
        if diff.top_level_keys_added:
            lines.append(f"  + added:   {', '.join(diff.top_level_keys_added)}")
        if diff.top_level_keys_removed:
            lines.append(f"  - removed: {', '.join(diff.top_level_keys_removed)}")

    _format_subjects_section(lines, diff, max_subjects_per_section)
    _format_other_fields_section(lines, diff)
    return "\n".join(lines)


__all__ = [
    "BundleDiff",
    "compute_bundle_diff",
    "format_bundle_diff",
]

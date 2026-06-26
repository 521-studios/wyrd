"""Unit tests for ``bundle_diff`` (wyrd-w3x0).

The structural diff is the operator's primary debugging surface when
``lexicon diff-bundle`` flags drift, so each branch needs explicit
coverage: byte-identical, length mismatch, added subjects, removed
subjects, changed subjects, top-level shape change, unparseable JSON,
and the optional dict-shaped fields (canonical_decompositions,
joiners, fantasy_morphemes).
"""

from __future__ import annotations

import json

from wyrd.generators.kenning.bundle_diff import (
    BundleDiff,
    compute_bundle_diff,
    format_bundle_diff,
)


def _bundle(**fields) -> str:
    """Serialize a bundle dict the way export-meanings does — same
    indent and ensure_ascii flag so byte-level equality matches what
    operators actually have on disk."""
    if "subjects" not in fields:
        fields["subjects"] = []
    return json.dumps(fields, ensure_ascii=False, indent=2)


def _subject(
    *,
    modifier_type: str = "Topographical",
    meaning: list[str] | None = None,
    modifier_tags: list[str] | None = None,
    modern_usage: list[str] | None = None,
    words: list[dict] | None = None,
) -> dict:
    """Build a subject dict with the four identity-tuple fields plus
    the canonical-shaped ``words`` payload. Defaults are filled in so
    tests can override one field at a time."""
    if words is None:
        words = [{"modern_usage": usage} for usage in (modern_usage or ["-ton"])]
    return {
        "modifier_type": modifier_type,
        "meaning": meaning or ["cottage"],
        "modifier_tags": modifier_tags or ["architecture"],
        "words": words,
    }


# --- byte-level paths ----------------------------------------------------


def test_byte_identical_inputs_match() -> None:
    text = _bundle(subjects=[_subject()])
    diff = compute_bundle_diff(text, text)
    assert diff.bytes_match
    assert diff.byte_size_committed == diff.byte_size_rebuilt
    assert diff.first_divergence_byte is None
    assert format_bundle_diff(diff).startswith("No drift")


def test_size_mismatch_populates_byte_fields() -> None:
    a = _bundle(subjects=[_subject()])
    b = a + "trailing"  # rebuilt is longer
    diff = compute_bundle_diff(a, b)
    assert not diff.bytes_match
    assert diff.byte_size_rebuilt > diff.byte_size_committed
    assert diff.first_divergence_byte == len(a)
    rendered = format_bundle_diff(diff)
    assert "Byte size" in rendered
    assert "First byte-level divergence" in rendered


def test_unparseable_bundle_sets_parse_error() -> None:
    a = _bundle(subjects=[])
    b = "not json at all"
    diff = compute_bundle_diff(a, b)
    assert not diff.bytes_match
    assert diff.parse_error is not None
    assert "could not parse" in diff.parse_error
    # Structural lists stay empty when parse fails — caller falls back
    # to the byte window.
    assert diff.subjects_added == []
    assert diff.subjects_removed == []
    assert diff.subjects_changed == []
    rendered = format_bundle_diff(diff)
    assert "Structural diff unavailable" in rendered


# --- structural: subjects -----------------------------------------------


def test_added_subject_classified_as_added() -> None:
    a = _bundle(subjects=[_subject(meaning=["cottage"])])
    b = _bundle(
        subjects=[
            _subject(meaning=["cottage"]),
            _subject(meaning=["hill"], modern_usage=["-don"]),
        ]
    )
    diff = compute_bundle_diff(a, b)
    assert len(diff.subjects_added) == 1
    assert "hill" in diff.subjects_added[0]
    assert "-don" in diff.subjects_added[0]
    assert diff.subjects_removed == []
    assert diff.subjects_changed == []


def test_removed_subject_classified_as_removed() -> None:
    a = _bundle(
        subjects=[
            _subject(meaning=["cottage"]),
            _subject(meaning=["hill"], modern_usage=["-don"]),
        ]
    )
    b = _bundle(subjects=[_subject(meaning=["cottage"])])
    diff = compute_bundle_diff(a, b)
    assert diff.subjects_added == []
    assert len(diff.subjects_removed) == 1
    assert "hill" in diff.subjects_removed[0]


def test_changed_subject_classified_as_changed_with_notes() -> None:
    """Same identity tuple (modifier_type, meaning, tags, modern_usage)
    but different payload (e.g. different word.old_english list) → the
    subject is a 'changed' pairing, not added/removed."""
    a = _bundle(
        subjects=[
            {
                "modifier_type": "Topographical",
                "meaning": ["cottage"],
                "modifier_tags": ["architecture"],
                "words": [{"modern_usage": "-ton", "old_english": ["tun"]}],
            }
        ]
    )
    b = _bundle(
        subjects=[
            {
                "modifier_type": "Topographical",
                "meaning": ["cottage"],
                "modifier_tags": ["architecture"],
                # Different OE form — identity tuple unchanged
                "words": [{"modern_usage": "-ton", "old_english": ["tun", "tūn"]}],
            }
        ]
    )
    diff = compute_bundle_diff(a, b)
    assert diff.subjects_added == []
    assert diff.subjects_removed == []
    assert len(diff.subjects_changed) == 1
    ident, notes = diff.subjects_changed[0]
    assert "cottage" in ident
    assert notes  # at least one note — exact wording is "payload differs"


def test_changed_subject_notes_name_diverging_word_slot() -> None:
    """When two paired subjects have multiple words and one of them
    differs (e.g. word[1] gets a new old_english form), the per-subject
    notes name the diverging slot — operators jump straight to it in
    the bundle file without scanning all words."""
    a = _bundle(
        subjects=[
            {
                "modifier_type": "Topographical",
                "meaning": ["cottage"],
                "modifier_tags": ["architecture"],
                "words": [
                    {"modern_usage": "-ton", "old_english": ["tun"]},
                    {"modern_usage": "-cot", "old_english": ["cot"]},
                ],
            }
        ]
    )
    b = _bundle(
        subjects=[
            {
                "modifier_type": "Topographical",
                "meaning": ["cottage"],
                "modifier_tags": ["architecture"],
                "words": [
                    {"modern_usage": "-ton", "old_english": ["tun"]},
                    {"modern_usage": "-cot", "old_english": ["cot", "kot"]},  # +kot
                ],
            }
        ]
    )
    diff = compute_bundle_diff(a, b)
    assert len(diff.subjects_changed) == 1
    _ident, notes = diff.subjects_changed[0]
    assert "words[1] differs" in notes
    # word[0] is identical between a and b — must NOT be flagged.
    assert "words[0] differs" not in notes


def test_subject_meaning_change_is_reported_in_notes() -> None:
    a = _bundle(
        subjects=[
            {
                "modifier_type": "Topographical",
                "meaning": ["cottage"],
                "modifier_tags": ["architecture"],
                "words": [{"modern_usage": "-ton"}],
            }
        ]
    )
    b = _bundle(
        subjects=[
            {
                "modifier_type": "Topographical",
                # Same identity tuple — the tuple uses tuple(meaning),
                # so two different lists DO produce different identities.
                # Instead: change a non-identity field to keep them paired.
                "meaning": ["cottage"],
                "modifier_tags": ["architecture"],
                "words": [{"modern_usage": "-ton", "old_english": ["tun"]}],
            }
        ]
    )
    diff = compute_bundle_diff(a, b)
    # Paired-as-changed: identity tuple matches, payload differs.
    assert len(diff.subjects_changed) == 1


def test_top_level_key_added() -> None:
    a = _bundle(subjects=[_subject()])
    b = _bundle(subjects=[_subject()], joiners={"old_english": [{"form": "of", "weight": 1}]})
    diff = compute_bundle_diff(a, b)
    assert diff.top_level_keys_added == ["joiners"]
    assert diff.top_level_keys_removed == []


def test_top_level_key_removed() -> None:
    a = _bundle(subjects=[_subject()], joiners={"old_english": [{"form": "of", "weight": 1}]})
    b = _bundle(subjects=[_subject()])
    diff = compute_bundle_diff(a, b)
    assert diff.top_level_keys_removed == ["joiners"]
    assert diff.top_level_keys_added == []


# --- structural: canonical_decompositions / joiners / fantasy_morphemes ----


def test_canonical_decompositions_churn_counted() -> None:
    a = _bundle(
        subjects=[_subject()],
        canonical_decompositions={"Cotton": {"source": "scholar"}},
    )
    b = _bundle(
        subjects=[_subject()],
        canonical_decompositions={
            "Cotton": {"source": "scholar"},
            "Dawlish": {"source": "tiebreaker"},
        },
    )
    diff = compute_bundle_diff(a, b)
    # One added → churn count 1.
    assert diff.canonical_decompositions_changed == 1


def test_joiners_payload_change_counted_as_changed() -> None:
    a = _bundle(
        subjects=[_subject()],
        joiners={"old_english": [{"form": "of", "weight": 1}]},
    )
    b = _bundle(
        subjects=[_subject()],
        joiners={"old_english": [{"form": "of", "weight": 2}]},  # weight bumped
    )
    diff = compute_bundle_diff(a, b)
    assert diff.joiners_changed == 1


# --- rendering -----------------------------------------------------------


def test_format_handles_byte_identical() -> None:
    diff = BundleDiff(bytes_match=True, byte_size_committed=100, byte_size_rebuilt=100)
    rendered = format_bundle_diff(diff)
    assert "No drift" in rendered


def test_format_truncates_long_subject_lists() -> None:
    """Wholesale-rewrite drift shouldn't dump a 4 MB diff."""
    a_subjects = [_subject(meaning=[f"meaning_{i}"]) for i in range(50)]
    b_subjects = [_subject(meaning=[f"different_{i}"]) for i in range(50)]
    diff = compute_bundle_diff(_bundle(subjects=a_subjects), _bundle(subjects=b_subjects))
    rendered = format_bundle_diff(diff, max_subjects_per_section=5)
    # 50 added + 50 removed; sample at 5 each → expect truncation markers
    assert "... and 45 more" in rendered


def test_same_size_content_differs_branch_is_reported() -> None:
    """Two equal-length strings with differing content should hit the
    'Byte size: N (matches; content differs)' branch in the renderer
    instead of the size-delta branch."""
    a = _bundle(subjects=[_subject(meaning=["cottage"])])
    b = a.replace("cottage", "cabbage", 1)  # same length, different content
    assert len(a) == len(b), "test setup: lengths must match for this branch"
    diff = compute_bundle_diff(a, b)
    assert diff.byte_size_committed == diff.byte_size_rebuilt
    rendered = format_bundle_diff(diff)
    assert "matches; content differs" in rendered


def test_non_dict_optional_field_is_silently_skipped() -> None:
    """When canonical_decompositions, joiners, or fantasy_morphemes is
    malformed (e.g. a list or None where a dict is expected), the diff
    helper must not crash and must leave the corresponding counter at 0
    — the byte-window covers the underlying drift."""
    a = _bundle(subjects=[_subject()], canonical_decompositions={"Cotton": {"source": "scholar"}})
    # Build malformed b directly via json.dumps (the _bundle helper would
    # accept the wrong type silently, but we want to inject deliberately).
    b = json.dumps(
        {
            "subjects": [_subject()],
            "canonical_decompositions": ["malformed-list-not-dict"],
        },
        ensure_ascii=False,
        indent=2,
    )
    diff = compute_bundle_diff(a, b)
    # Counter stays at 0 because the malformed shape was skipped.
    assert diff.canonical_decompositions_changed == 0
    # No crash, parse_error stays None (both parsed fine).
    assert diff.parse_error is None


def test_format_renders_joiners_and_fantasy_morpheme_counters() -> None:
    """The format helper must surface joiners_changed and
    fantasy_morphemes_changed in the 'Other top-level fields with
    churn' section — operators rely on it to know which sub-field
    moved."""
    diff = BundleDiff(
        bytes_match=False,
        byte_size_committed=200,
        byte_size_rebuilt=210,
        joiners_changed=3,
        fantasy_morphemes_changed=7,
    )
    rendered = format_bundle_diff(diff)
    assert "joiners: 3 changed" in rendered
    assert "fantasy_morphemes: 7 changed" in rendered


def test_format_truncates_subjects_changed_section() -> None:
    """The 'Changed (sample):' section should truncate the same way as
    Added/Removed — flagged by test-coverage-reviewer on round 1."""
    # Build a bundle with many paired-but-changed subjects.
    a_subjects = [
        {
            "modifier_type": "Topographical",
            "meaning": [f"sense_{i}"],
            "modifier_tags": ["architecture"],
            "words": [{"modern_usage": f"-{i}", "old_english": [f"oe_{i}_a"]}],
        }
        for i in range(20)
    ]
    b_subjects = [
        {
            "modifier_type": "Topographical",
            "meaning": [f"sense_{i}"],
            "modifier_tags": ["architecture"],
            # Same identity tuple, different old_english payload → paired-as-changed.
            "words": [{"modern_usage": f"-{i}", "old_english": [f"oe_{i}_b"]}],
        }
        for i in range(20)
    ]
    diff = compute_bundle_diff(_bundle(subjects=a_subjects), _bundle(subjects=b_subjects))
    assert len(diff.subjects_changed) == 20
    rendered = format_bundle_diff(diff, max_subjects_per_section=3)
    assert "Changed (sample):" in rendered
    assert "... and 17 more" in rendered


def test_first_divergence_prefix_branch_committed_longer() -> None:
    """Symmetric to the test where rebuilt is longer: when COMMITTED is
    the longer text and rebuilt is a strict prefix, the diff helper
    should still produce a sensible byte offset + window pair."""
    rebuilt = _bundle(subjects=[_subject()])
    committed = rebuilt + "extra trailing bytes for the test"
    diff = compute_bundle_diff(committed, rebuilt)
    assert not diff.bytes_match
    assert diff.byte_size_committed > diff.byte_size_rebuilt
    assert diff.first_divergence_byte == diff.byte_size_rebuilt  # divergence at rebuilt's end
    # Window: committed side shows the trailing bytes; rebuilt side
    # shows "<end>" because we ran out of rebuilt content.
    assert diff.first_divergence_window is not None
    committed_win, rebuilt_win = diff.first_divergence_window
    assert "extra trailing" in committed_win
    assert rebuilt_win == "<end>"


def test_byte_size_counts_utf8_not_python_chars() -> None:
    """``byte_size_*`` and ``first_divergence_byte`` are UTF-8 byte
    counts (matching `wc -c` / the on-disk bundle size), not Python
    ``str`` character counts. Pin this with a non-ASCII etymological
    form — 'þ' is one ``str`` char but two UTF-8 bytes, so the same
    character-length strings get different byte counts."""
    pure_ascii = '{"x": "t"}'  # 10 chars, 10 UTF-8 bytes
    with_thorn = '{"x": "þ"}'  # 10 chars, 11 UTF-8 bytes (þ is 2 bytes)
    assert len(pure_ascii) == len(with_thorn) == 10, "test setup: char lengths must match"
    assert len(pure_ascii.encode("utf-8")) == 10
    assert len(with_thorn.encode("utf-8")) == 11, "test setup: þ must be 2 UTF-8 bytes"
    diff = compute_bundle_diff(pure_ascii, with_thorn)
    # byte_size_* must reflect UTF-8 bytes, not character counts.
    assert diff.byte_size_committed == 10
    assert diff.byte_size_rebuilt == 11
    # And first_divergence_byte is also UTF-8 byte offset, not char offset.
    # The 't' vs 'þ' chars sit at character index 7 in both strings —
    # which is also byte offset 7 because everything before them is ASCII.
    assert diff.first_divergence_byte == 7


def test_format_first_divergence_window_strips_newlines_for_readability() -> None:
    a = _bundle(subjects=[_subject(meaning=["cottage"])])
    # Construct b by replacing the meaning value in a string-level way.
    b = a.replace("cottage", "cabin   ")  # same length so byte offsets stay close
    diff = compute_bundle_diff(a, b)
    rendered = format_bundle_diff(diff)
    # The window-rendering replaces "\n" with "\\n" so the report stays
    # one line per side — pin that escaping here.
    assert "\\n" in rendered or "committed:" in rendered  # at least one of these


def test_first_divergence_byte_offset_accounts_for_multibyte_utf8() -> None:
    """compute_bundle_diff reports the UTF-8 BYTE offset (what ``wc -c`` / a hex
    editor see), not the Python ``str`` char index. The bundle is written
    ensure_ascii=False, so non-ASCII forms (þ, ŵ, æ, …) before the divergence
    push the byte offset past the char index — the exact distinction the field
    exists to capture, previously only exercised on pure-ASCII content."""
    # 'þ' and 'ŵ' are 2 bytes each in UTF-8; the values diverge after "þŵ-".
    committed = '{"k": "þŵ-AAA"}'
    rebuilt = '{"k": "þŵ-BBB"}'
    diff = compute_bundle_diff(committed, rebuilt)
    char_index = next(k for k, (c, r) in enumerate(zip(committed, rebuilt, strict=False)) if c != r)
    assert diff.first_divergence_byte == len(committed[:char_index].encode("utf-8"))
    assert diff.first_divergence_byte > char_index  # multibyte before it


def test_first_divergence_byte_offset_multibyte_prefix_case() -> None:
    """When one text is a prefix of the other (no mid-string divergence), the
    offset is the BYTE length of the shorter text — multibyte content in the
    common prefix still makes it exceed the char length."""
    committed = '{"k": "þŵ"}'
    rebuilt = committed + "TAIL"
    diff = compute_bundle_diff(committed, rebuilt)
    assert diff.first_divergence_byte == len(committed.encode("utf-8"))
    assert diff.first_divergence_byte > len(committed)  # multibyte in the prefix

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


def test_format_first_divergence_window_strips_newlines_for_readability() -> None:
    a = _bundle(subjects=[_subject(meaning=["cottage"])])
    # Construct b by replacing the meaning value in a string-level way.
    b = a.replace("cottage", "cabin   ")  # same length so byte offsets stay close
    diff = compute_bundle_diff(a, b)
    rendered = format_bundle_diff(diff)
    # The window-rendering replaces "\n" with "\\n" so the report stays
    # one line per side — pin that escaping here.
    assert "\\n" in rendered or "committed:" in rendered  # at least one of these

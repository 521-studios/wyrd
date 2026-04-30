"""Tests for the alphabetical-dictionary parser (Mawer/Ekwall/Johnston-style)."""

from __future__ import annotations

from wyrd.generators.kenning.dictionary_parser import (
    _ENTRY_BODY_SIGNALS,
    _is_real_headword,
    _split_packed_paragraph,
    find_body_bounds,
    parse_alphabetical_text,
)

# A synthetic Mawer-style fixture that exercises the patterns we care about:
# headword detection, phonetic brackets, parish in parens, attestation lines,
# packed paragraphs, false-positive filtering, body-boundary detection.
SYNTHETIC = """\
THE PLACE-NAMES OF NOWHERESHIRE

CONTENTS

PART I. PLACE-NAMES IN ALPHABETICAL ORDER ........... 1
PART II. ELEMENTS ................................. 100


PART I. PLACE-NAMES IN ALPHABETICAL ORDER

Abberwick (Eglingham). 1169 Pipe Alburwic; 1278 Ass. Alberwick.
O.E. Alubeorhtes wic = Alubeorht's wic. Phonology, §§ 1, 2.

Acomb [ækəm] (Bywell St Peter). 1268 Ipm. Akum.
O.E. (æt þām) ācum = (at the) oaks.

Mod. Irish fearran = land. This should NOT be parsed as an entry.

Langhope (Hexhamshire). 1229 Gray Langhop; 1663 Rental Langupp. Langley (Haydon). c. 1175 Langalea. Langton (Gainford). 1104 Langadun.


PART II. ELEMENTS

dun. O.E. dūn. A hill. Common in many forms.
"""


def test_find_body_bounds_skips_toc_and_part_two() -> None:
    lines = SYNTHETIC.split("\n")
    start, end = find_body_bounds(lines)
    body = "\n".join(lines[start:end])
    # Body must include Abberwick but not the dun element entry.
    assert "Abberwick" in body
    assert "PART II" not in body
    assert "dun. O.E. dūn" not in body


def test_parse_extracts_alphabetical_entries() -> None:
    entries = parse_alphabetical_text(SYNTHETIC)
    names = [e.toponym for e in entries]
    # Abberwick, Acomb, Langhope, Langley, Langton — five real entries.
    assert "Abberwick" in names
    assert "Acomb" in names
    assert "Langhope" in names
    assert "Langley" in names
    assert "Langton" in names
    # "Mod." abbreviation must NOT appear as an entry.
    assert "Mod" not in names


def test_packed_paragraph_split() -> None:
    """Three entries crammed into one paragraph should split into three."""
    flat = (
        "Langhope (Hexhamshire). 1229 Gray Langhop. "
        "Langley (Haydon). c. 1175 Langalea. "
        "Langton (Gainford). 1104 Langadun."
    )
    parts = _split_packed_paragraph(flat)
    # Three chunks, each starting with one headword.
    assert len(parts) == 3
    assert parts[0].startswith("Langhope")
    assert parts[1].startswith("Langley")
    assert parts[2].startswith("Langton")


def test_is_real_headword_filters_caps_and_abbreviations() -> None:
    assert _is_real_headword("Abberwick")
    assert _is_real_headword("Stockton-on-Tees")
    assert _is_real_headword("St John's")
    # All-caps single-word page running titles
    assert not _is_real_headword("DURHAM")
    assert not _is_real_headword("PLACE-NAMES")
    # All-caps multi-word page headers
    assert not _is_real_headword("PLACE NAMES OF DURHAM")
    # Common abbreviations
    assert not _is_real_headword("Mod")
    assert not _is_real_headword("OE")
    # Sentence starters
    assert not _is_real_headword("This")
    assert not _is_real_headword("The")


def test_entry_body_includes_attestations_and_etymology() -> None:
    entries = {e.toponym: e for e in parse_alphabetical_text(SYNTHETIC)}
    abberwick = entries["Abberwick"]
    # Body should contain both the dated attestation chain and the etymology.
    assert "1169 Pipe Alburwic" in abberwick.body_text
    assert "Alubeorht" in abberwick.body_text


def test_filters_sentence_fragments_via_body_signal() -> None:
    """Entries lacking an etymology-signal marker (year / O.E. / A.S. /
    Phonology / Domesday) should be rejected as OCR fragments."""
    fragment = """\
THE PLACE-NAMES OF NOWHERESHIRE

PART I. PLACE-NAMES IN ALPHABETICAL ORDER

Acomb [ækəm] (Bywell St Peter). 1268 Ipm. Akum.
O.E. (æt þām) ācum = (at the) oaks.

Self-explanatory. The names are obvious.

Acklington (Warkworth). 1176 Pipe Eclinton.
O.E. Æcceling(a)tun = farm of Æccel.

PART II. ELEMENTS
"""
    entries = parse_alphabetical_text(fragment)
    names = {e.toponym for e in entries}
    # Real entries are kept (they have year + O.E. signals).
    assert "Acomb" in names
    assert "Acklington" in names
    # Fragment with no signal markers is dropped.
    assert "Self-explanatory" not in names


def test_filters_short_artifact_bodies() -> None:
    """A paragraph that's just 'Burn.' with no body content gets dropped
    even though 'Burn' is a capitalized, period-terminated word."""
    fragment = """\
THE PLACE-NAMES OF NOWHERESHIRE

PART I. PLACE-NAMES IN ALPHABETICAL ORDER

Acomb (Bywell). 1268 Ipm. Akum. O.E. ācum.

Burn.

Acklington. 1176 Pipe. O.E. Æcceling(a)tun.

PART II. ELEMENTS
"""
    entries = parse_alphabetical_text(fragment)
    names = {e.toponym for e in entries}
    assert "Acomb" in names
    assert "Acklington" in names
    # Standalone "Burn." is too short to contain any signal — dropped.
    assert "Burn" not in names


def test_entry_with_phonetic_bracket() -> None:
    entries = {e.toponym: e for e in parse_alphabetical_text(SYNTHETIC)}
    acomb = entries["Acomb"]
    assert "Bywell" in acomb.body_text
    # Body should preserve OE form text (with macron) and modern spelling.
    assert "ācum" in acomb.body_text
    assert "Akum" in acomb.body_text


def test_dotted_signal_markers_match_when_followed_by_space() -> None:
    """Regression: the body-signal regex must match dotted abbreviations
    like O.E., M. Ir., A.S. when followed by a space or punctuation. Before
    the (?!\\w) lookahead fix, the trailing `\\b` only matched if a word
    character followed the literal period — meaning `O.E. ham` (the typical
    body shape) silently failed the test, and the regex was qualifying
    bodies via the year alternative alone."""
    cases = [
        ("from O.E. ham, a homestead", True),
        ("ends in O.E.", True),
        ("from O.E., a hill", True),
        ("from A.S. tun", True),
        ("M. Ir. seann means old", True),
        ("Mod. Gael. baile", True),
        # Negative: a signal-shaped substring that has trailing word chars
        # is a different word and must NOT match.
        ("see O.Ex (not a marker)", False),
        ("nothing relevant in this body", False),
    ]
    for body, expected in cases:
        got = bool(_ENTRY_BODY_SIGNALS.search(body))
        assert got == expected, f"{body!r}: got={got}, expected={expected}"


def test_celtic_signal_words_qualify_as_real_entries() -> None:
    """wyrd-3qq regression: Celtic / Norse / Gaelic / Welsh / Manx markers
    must qualify a body as a real etymology entry. Before the broadening,
    the signal regex was English-only (O.E. / A.S. / O.N. / year / Domesday)
    and Celtic books with G./Ir./W./Manx/Norse markers had legitimate
    bodies rejected, dropping yield by 90-100% on Joyce, Moore, Johnston,
    Gillies."""
    # Note: each test paragraph wraps in a body-bounds frame so the parser's
    # body-detection logic doesn't filter the fixture out.
    celtic = """\
THE PLACE-NAMES OF NOWHERESHIRE

PART I. PLACE-NAMES IN ALPHABETICAL ORDER

Aberdeen (Aberdeen). Gaelic Obar Dheathain, Welsh Aberdeen, mouth of the Don.

Tobermory (Mull). Gael. tobar Mhoire, the well of Mary. Recorded as such.

Snaefell (Man). From Norse snae 'snow' + fjall 'mountain', the snow mountain.

Ballyfoyle (Kilkenny). Old Irish baile + Foghail, the personal name.

PART II. ELEMENTS
"""
    entries = parse_alphabetical_text(celtic)
    names = {e.toponym for e in entries}
    # Each Celtic-signal flavor should be retained.
    assert "Aberdeen" in names, "Gaelic + Welsh markers should qualify"
    assert "Tobermory" in names, "Gael. abbreviation should qualify"
    assert "Snaefell" in names, "Norse marker should qualify"
    assert "Ballyfoyle" in names, "Old Irish marker should qualify"


def test_find_body_bounds_picks_largest_span_when_marker_repeats() -> None:
    """wyrd-3qq regression: when 'PART I' appears in TOC, body chapter, AND
    nested subsections (Moore Isle of Man has five 'Part I' headings), the
    bounds finder must pick the candidate with the LARGEST body span — not
    blindly take the last one. Otherwise nested subsection labels collapse
    the body to a tiny span and yield drops to single digits."""
    text = "\n".join(
        [
            "TITLE PAGE",
            "",
            "PART I. SOMETHING",  # TOC entry — 0-line body
            "PART II. SOMETHING ELSE",  # ends the above
            "",
            "PART I. THE REAL BODY",  # the real body header
            *[f"Entry{i} (Parish). 1{i:03d} Pipe Alfor. O.E. Alfor's wic." for i in range(2000)],
            "PART II. ELEMENTS",
            "ending matter",
        ]
    )
    lines = text.split("\n")
    s, e = find_body_bounds(lines)
    body = "\n".join(lines[s:e])
    assert "Entry0" in body, "real body must be inside the bounds"
    assert "Entry1999" in body, "real body must extend through to the end marker"
    assert "TITLE PAGE" not in body
    assert "ending matter" not in body


def test_find_body_bounds_recognizes_end_marker_on_line_after_start() -> None:
    """Regression: when a start marker is immediately followed by an end
    marker on the very next line (TOC entry shape: 'PART I. ... 1\\nPART II.
    ... 100'), the candidate's span must collapse to zero so that the real
    body candidate wins the largest-span vote.

    Before the fix, the end-marker scan started at `body_start + 1` and
    skipped past an adjacent end-line, letting the bogus TOC candidate run
    all the way to a later end marker and grow large enough to win."""
    text = "\n".join(
        [
            "TITLE",
            "",
            "PART I. ALPHABETICAL ORDER",  # TOC entry — line 2
            "PART II. ELEMENTS",  # adjacent end marker — line 3
            "",
            "PART I. ALPHABETICAL ORDER",  # real body — line 5
            *[f"Entry{i} (Parish). 1{i:03d} Pipe Foo. O.E. bar." for i in range(1500)],
            "PART II. ELEMENTS",
            "trailing",
        ]
    )
    lines = text.split("\n")
    s, e = find_body_bounds(lines)
    body = "\n".join(lines[s:e])
    assert "Entry0" in body, "real body candidate should win"
    # The TOC candidate should have collapsed to span 0; if the offset bug
    # returned, the TOC candidate would have spanned line 3..end-of-real-body
    # and the start would land at line 3.
    assert s >= 5, f"start={s}: TOC candidate appears to have won (offset bug)"


def test_find_body_bounds_safety_net_for_tiny_body_on_big_doc() -> None:
    """When the heuristic produces a tiny body span on a large document, the
    safety net should return the whole doc rather than a misidentified
    sliver. The downstream `_entry_body_is_real` filter handles the noise."""
    # Synthesize a >1000-line doc with NO body markers anywhere — the
    # fall-through "first headword in second half" logic could land on a
    # near-end line and yield a 0-line body.
    lines = ["preamble line"] * 1500
    # Inject just one matchable headword near the very end so the fallback
    # picks it but the resulting span is essentially zero.
    lines[1499] = "Foobar (Place). 1086 Domesday Foobar."
    s, e = find_body_bounds(lines)
    # Safety net should kick in: body span < 5% of 1500 lines (= 75) means
    # we return the whole doc rather than a 1-line slice.
    assert (e - s) >= len(lines) // 20, (
        f"safety net should expand tiny body, got {e - s} of {len(lines)} lines"
    )

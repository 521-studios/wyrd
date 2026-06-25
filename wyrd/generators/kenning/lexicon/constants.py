"""Lexicon constants — language-field mappings + OCR ligature normalization.

Two groups of constants currently live here:

* ``LANGUAGE_FIELDS`` / ``NON_LANGUAGE_FIELDS`` — map meanings.json
  source-language field names to the lexicon's canonical language
  codes, plus the per-entry-flag field names that aren't language
  slots. Used by ingestion to route ``meanings.json`` data into the
  etymon table without losing the wave-2 (Hebrew, Arabic, Sanskrit,
  etc.) bundle fields. Keep in sync with ``_LEGEND`` in the kenning
  generator.

* ``OCR_LIGATURE_MAP`` + ``normalize_ocr_form`` — OE ligature (æ, ð, þ,
  œ, ȳ, ē) normalization for OCR-mangled forms. The repeated-replace
  shape and its performance rationale is documented inline; don't
  refactor without re-reading the docstring on ``normalize_ocr_form``.

This module is referenced as ``lexicon.constants`` (the canonical
import) and re-exported from ``lexicon/__init__.py`` for back-compat
with code that imports ``from wyrd.generators.kenning.lexicon import
LANGUAGE_FIELDS`` etc.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.morpheme_surface import _BOUNDARY_DASHES

# Single source of truth for "what counts as a boundary position-dash" — the same
# set ``normalize_morpheme_surface`` trims from a stored surface (D45). As a
# prefix/suffix tuple for ``str.startswith``/``endswith`` (empty-string safe,
# unlike a substring test). Shared so ``position_from_usage`` derives position
# from the SAME notion of "dash" the surface de-dash uses.
_BOUNDARY_DASH_AFFIXES = tuple(_BOUNDARY_DASHES)

# Map meanings.json source-language field names to the lexicon's
# canonical language codes. Keep in sync with _LEGEND in the kenning
# generator. The misspellings ('old_scandanavian', 'old _english')
# are real entries in the bundled meanings.json — normalized here so
# we don't lose data.
LANGUAGE_FIELDS: dict[str, str] = {
    "old_english": "old-english",
    "old _english": "old-english",
    "old_scandinavian": "old-norse",
    "old_scandanavian": "old-norse",
    "old_french": "norman-french",
    "celtic_mix": "celtic",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "modern_english": "modern-english",
    "biblical": "biblical",
    # wyrd-vsrn Phase 2c: wave-2 bundle fields — round-trip on
    # ingestion routes 'hebrew'/'arabic'/etc. JSON keys to the
    # canonical-language code on the etymon table. Multiple lexicon
    # codes (e.g. he + hbo + sem-pro) emit into the same bundle
    # field via _LANG_CODE_TO_JSON_FIELD; ingestion direction loses
    # that detail and treats every entry as the canonical lang.
    "hebrew": "he",
    "arabic": "ar",
    "persian": "fa",
    "sanskrit": "sa",
    "akkadian": "akk",
    "egyptian": "egy",
    "aramaic": "arc",
    "armenian": "axm",
}

# Names of ``word`` keys that are NOT source-language form arrays.
# Lexicon ingest must treat these as non-language data: ``modern_usage``
# is the canonical surface form (read elsewhere to derive a reflex
# position, not stored as an etymon), and ``source_known`` is a per-
# entry boolean flag the ingester has historically ignored. The set is
# used by tests as an exhaustiveness check against the LANGUAGE_FIELDS
# enumeration; the bundle loader treats anything not in either set as
# a new language slot and routes it accordingly.
NON_LANGUAGE_FIELDS: set[str] = {"modern_usage", "source_known", "morpheme_id"}


# ---------------------------------------------------------------------------
# OCR ligature normalization
# ---------------------------------------------------------------------------
#
# OE ligature characters (æ, ð, þ, œ, ȳ, ē, etc.) are routinely
# mangled by OCR engines. The same etymon ends up under many
# spellings: "Hædan", "Hcsdan", "Hsedan", "Haedan", "Hædann"...
# breaking consensus counts.
#
# We normalize by:
#   1. Mapping common OCR confusions back to ASCII-equivalent OE chars.
#   2. Lowercasing.
#   3. Stripping leading/trailing dashes (which mark position, not form).
#
# This produces a canonical key that's robust to OCR variation.

# OCR-confused digraphs that should map to OE ligatures (in ASCII).
# The repeated-replace shape was investigated as a perf hot-spot under
# wyrd-0ke (Gemini PR #53 round-5 concern about M·N allocations on
# multi-MB OCR bodies). Benchmarked alternatives (str.translate +
# str.replace, str.translate + re.sub, single re.sub over an
# alternation pattern) all came out 5-15x SLOWER on 1/10/50 MB bodies
# and produced identical memory peaks (intermediate strings are GC'd
# between replace calls so the peak is dominated by the single largest
# in-flight string, not the running allocation count). Python's
# str.replace is a heavily-optimized C path with a memchr-fast
# negative-scan; replacing it with translate or regex adds Python-side
# dispatch overhead that swamps the per-mapping allocation savings on
# the bodies this function actually sees. Keep the loop; documented
# here so a future drive-by perf review doesn't redo the same work.
OCR_LIGATURE_MAP: list[tuple[str, str]] = [
    # Order matters — longer/more-specific first. "cs" inside an OE form
    # is almost always a misread æ; same for "ce" in many positions. We
    # keep the ASCII-friendly equivalents (ae, dh, th, oe) in normalized
    # form.
    ("æ", "ae"),
    ("ð", "dh"),
    ("þ", "th"),
    ("œ", "oe"),
    ("ȳ", "y"),
    ("ē", "e"),
    ("ī", "i"),
    ("ō", "o"),
    ("ū", "u"),
    ("ā", "a"),
    # Common OCR confusions for æ
    ("cs", "ae"),  # "Hcsdan" → "haedan"
    ("ce", "ae"),  # "Hcedan" → "haedan" (only in OE context — risky in modern)
    # Common OCR confusions for ð
    ("§", "dh"),
]


def normalize_ocr_form(form: str) -> str:
    """Normalize an etymon form against common OCR confusions.

    Returns a lowercased, dash-stripped, ligature-normalized form
    suitable as a clustering key. The result is NOT meant to be
    displayed — it's a join key.

    Conservative: we only apply the most reliable mappings. "ce" →
    "ae" is genuinely OE-context-dependent and applied here; if it
    produces false positives in modern-english entries we'd want a
    per-language rule set, but for now the lexicon is dominated by
    historical forms.

    The repeated-replace shape is intentional. See the rationale next
    to ``OCR_LIGATURE_MAP`` — translate / regex alternatives all
    benched slower on the call-site body sizes (1-50 MB) we actually
    see, and the memory-peak claim that motivated wyrd-0ke didn't
    hold up under measurement either.
    """
    s = form.strip().strip("-").lower()
    for src, dst in OCR_LIGATURE_MAP:
        s = s.replace(src, dst)
    return s


def position_from_usage(modern_usage: str) -> str:
    """Derive a reflex position from the dash markers on a modern_usage string.

    Starts and ends with '-' → inner, ends with '-' → pre, otherwise
    post.

    NB (wyrd-vpri): this INTENTIONALLY diverges from
    ``Meaning._set_location``, which now maps a no-dash usage to the
    distinct ``bare`` location. This function feeds the reflex
    ``position`` DB column, whose CHECK constraint is only
    ('pre','post','inner') — there is no 'bare' position there — and
    the value is consumed solely as an ORDER BY sort key, never as a
    runtime eligibility gate (the runtime re-derives location from the
    dashed surface form). So no-dash → 'post' is the correct DB-side
    mapping; don't "fix" it to 'bare' or the constraint rejects it.
    """
    # Recognize ANY boundary dash, not just ASCII "-": a marker drawn with an
    # en-dash / U+2010 must derive the SAME position as its ASCII twin, or a
    # Unicode-dash reflex forks from the ASCII one on the position axis (the
    # surface is already de-dashed via ``normalize_morpheme_surface``, D45).
    if modern_usage.startswith(_BOUNDARY_DASH_AFFIXES) and modern_usage.endswith(
        _BOUNDARY_DASH_AFFIXES
    ):
        return "inner"
    if modern_usage.endswith(_BOUNDARY_DASH_AFFIXES):
        return "pre"
    return "post"

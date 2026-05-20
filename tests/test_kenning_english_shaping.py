"""Tests for wyrd-ha9q Phase 2b: english_shaped derivation."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as kenning_cli
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.english_shaping import (
    KNOWN_FORM_OVERRIDES,
    PHASE2A_NON_LATIN_LANGS,
    _apply_digraphs,
    _apply_single_chars,
    _ipa_to_english_fallback,
    _looks_english_readable,
    _strip_transliteration,
    derive_english_shaped,
    derive_english_shaped_all,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Lexicon DB with the schema created and migrated."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


# ---------------------------------------------------------------------
# Diacritic-stripping primitives
# ---------------------------------------------------------------------


def test_apply_digraphs_replaces_semitic_two_letter_marks() -> None:
    """Semitic-transliteration digraphs (ṯ→th, ḫ→kh, š→sh, ǧ→j, ġ→gh,
    ḏ→dh) produce two-letter output. str.replace replaces ALL
    occurrences within a string."""
    assert _apply_digraphs("ṯalāṯah") == "thalāthah"
    assert _apply_digraphs("ḫamis") == "khamis"
    assert _apply_digraphs("šayṭān") == "shayṭān"  # ṭ stays (handled by single-char)
    assert _apply_digraphs("ǧinn") == "jinn"
    assert _apply_digraphs("ġūl") == "ghūl"  # ū stays for single-char pass
    assert _apply_digraphs("ḏikr") == "dhikr"


def test_apply_digraphs_replaces_sanskrit_sibilants_to_sh() -> None:
    """Both ś and ṣ map to 'sh' (and 'Sh') — Sanskrit retroflex /
    palatal sibilants. rakṣasa convention is rakshasa, not rakssasa."""
    assert _apply_digraphs("rakṣasa") == "rakshasa"
    assert _apply_digraphs("śiva") == "shiva"
    assert _apply_digraphs("Śakti") == "Shakti"


def test_apply_single_chars_handles_sanskrit_palatal_velar_nasals() -> None:
    """Sanskrit palatal ñ and velar ṅ map to bare 'n' (single-char) —
    not 'ny'/'ng' digraphs. The following consonant supplies the
    palatal/velar quality, so digraphing would double-count:
        liṅga → 'linga' (correct) not 'lingga'
        añjali → 'anjali' (correct) not 'anyjali'
        gaṅgā → 'ganga' (correct) not 'ganggā'
    Pre-fix this used a digraph map; pinned here so a regression
    that re-introduces ng/ny would fail."""
    assert _apply_single_chars("añjali") == "anjali"
    assert _apply_single_chars("liṅga") == "linga"
    assert _apply_single_chars("gaṅgā") == "ganga"


def test_apply_single_chars_strips_emphatics_and_long_vowels() -> None:
    """Single-char map: emphatics (ḍ→d, ḥ→h, ṭ→t), long vowels (ā→a,
    ī→i, ū→u), retroflex single-chars (ṛ→r, ṇ→n, ḷ→l). The single-char
    pass intentionally does NOT touch digraph chars (ṣ, š, ǧ, ...) —
    those are handled by `_apply_digraphs` upstream."""
    assert _apply_single_chars("ḍāl") == "dal"
    assert _apply_single_chars("ḥikma") == "hikma"
    assert _apply_single_chars("ṭāhir") == "tahir"
    assert _apply_single_chars("ṛ ṇ ḷ ā ī ū ē ō") == "r n l a i u e o"


def test_apply_single_chars_silences_glottals() -> None:
    """ʿ ʾ ʔ → empty string. Acute / grave accents → bare letter."""
    assert _apply_single_chars("ʿifrīt") == "ifrit"
    assert _apply_single_chars("ʾāl") == "al"
    assert _apply_single_chars("rāʔ") == "ra"
    # Hebrew stress-acute drops:
    assert _apply_single_chars("káp") == "kap"
    assert _apply_single_chars("bósem") == "bosem"


def test_strip_transliteration_runs_full_pipeline() -> None:
    """End-to-end strip: digraphs → single-chars → cleanup.
    Verify a Hebrew + Arabic + Sanskrit sample each."""
    assert _strip_transliteration("ʿifrīt") == "ifrit"
    assert _strip_transliteration("rakṣasa") == "rakshasa"
    assert _strip_transliteration("kɛ́lɛḇ, kélev") == "kelev, kelev"  # comma stays; English-readable
    assert _strip_transliteration("Ba'al Zvuv") == "Baal Zvuv"
    # ṯ → 'th' is the rule output ("shabbath" is the older English form).
    # The modern English "shabbat" convention is a known-form override
    # rather than a rule case — kept here as the rule baseline.
    assert _strip_transliteration("šabbāṯ") == "shabbath"


def test_strip_transliteration_collapses_ipa_brackets_and_subscripts() -> None:
    """IPA-style brackets / slashes / length / stress markers are
    stripped. Akkadian subscripts (ar-ga-man-nu, GIR₄) too. The
    d͡ʒ ligature collapses to 'dzh' since ʒ goes through the digraph
    map to 'zh' before the tie-bar gets stripped."""
    assert _strip_transliteration("/d͡ʒɪn/") == "dzhin"
    assert _strip_transliteration("ar-ga-man-nu₂") == "ar-ga-man-nu"
    # ħ→h, ɔ→o, θ→th, brackets+stress stripped → "hothul"
    assert _strip_transliteration("[ħɔˈθul]") == "hothul"


# ---------------------------------------------------------------------
# _looks_english_readable predicate
# ---------------------------------------------------------------------


def test_looks_english_readable_accepts_clean_ascii() -> None:
    assert _looks_english_readable("rakshasa") is True
    assert _looks_english_readable("baal zvuv") is True
    assert _looks_english_readable("ba'al-zvuv") is True


def test_looks_english_readable_rejects_residual_diacritics() -> None:
    """If our maps missed a character, the residual non-ASCII char
    means the output isn't ready — return False so the caller gets
    None and falls back to canonical_form."""
    assert _looks_english_readable("ifrit") is True
    assert _looks_english_readable("ʿifrīt") is False  # both ʿ and ī survive
    assert _looks_english_readable("kelev_") is False  # underscore not in allowed set
    assert _looks_english_readable("123") is False  # digits-only, no letter
    assert _looks_english_readable("") is False


# ---------------------------------------------------------------------
# IPA fallback
# ---------------------------------------------------------------------


def test_ipa_to_english_fallback_strips_brackets_and_stress() -> None:
    """`/d͡ʒɪnː/` → 'dzhin' (length mark and tie-bar removed; ʒ→zh)."""
    assert _ipa_to_english_fallback("/d͡ʒɪnː/") == "dzhin"
    assert _ipa_to_english_fallback("[kalb]") == "kalb"


def test_ipa_to_english_fallback_returns_none_on_unstrippable() -> None:
    """If even after stripping the result has non-ASCII residuals,
    return None — caller falls back to canonical_form."""
    # An IPA string with phonemes we don't map: returns None
    assert _ipa_to_english_fallback("/ʕʕʕ/") is None  # only ayin, all silenced → empty


# ---------------------------------------------------------------------
# Top-level derive_english_shaped
# ---------------------------------------------------------------------


def test_derive_skips_latin_script_languages() -> None:
    """Old English / Latin / Welsh / Old French / Proto-Germanic etc.
    return None — canonical_form is already English-readable."""
    for lang in ("old-english", "latin", "old-french", "welsh", "proto-germanic"):
        assert (
            derive_english_shaped(
                canonical_form="tūn",
                language=lang,
                transliteration="tun",
                pronunciation_ipa=None,
            )
            is None
        ), f"failed for {lang}"


def test_derive_known_form_override_wins_over_rule_strip() -> None:
    """Even when the diacritic-strip would produce a valid output,
    the cultural-precedent override fires first. rakṣasa → rakshasa
    (the override) not 'rakshasa' (which happens to be the same here)
    or rakshasa via single-char rules. Use a case where the override
    differs: 'jinn' override is short 'jinn', not the digraph-stripped
    'jinn' (which would also be jinn — same answer). Use 'apsaras'
    override (override = 'apsara', singular)."""
    assert (
        derive_english_shaped(
            canonical_form="अप्सरस्",
            language="sa",
            transliteration="apsaras",
            pronunciation_ipa=None,
        )
        == "apsara"
    )


def test_derive_known_form_override_case_insensitive() -> None:
    """Override match is case-insensitive: 'RAKṢASA', 'Rakṣasa',
    'rakṣasa' all resolve."""
    for case in ("rakṣasa", "Rakṣasa", "RAKṢASA"):
        assert (
            derive_english_shaped(
                canonical_form="रक्षस",
                language="sa",
                transliteration=case,
                pronunciation_ipa=None,
            )
            == "rakshasa"
        ), f"failed for {case}"


def test_derive_falls_through_to_strip_when_no_override() -> None:
    """A transliteration with no override goes through the strip
    pipeline. Sample: Hebrew 'shabát' → 'shabat'."""
    assert (
        derive_english_shaped(
            canonical_form="שבת",
            language="he",
            transliteration="shabát",
            pronunciation_ipa=None,
        )
        == "shabat"
    )


def test_derive_falls_through_to_ipa_when_no_transliteration() -> None:
    """When transliteration is absent, IPA is the fallback."""
    assert (
        derive_english_shaped(
            canonical_form="جن",
            language="ar",
            transliteration=None,
            pronunciation_ipa="/d͡ʒɪnː/",
        )
        == "dzhin"
    )


def test_derive_returns_none_on_no_inputs() -> None:
    """All sources NULL → None."""
    assert (
        derive_english_shaped(
            canonical_form="جن",
            language="ar",
            transliteration=None,
            pronunciation_ipa=None,
        )
        is None
    )


def test_derive_returns_none_when_strip_leaves_residuals() -> None:
    """If the transliteration uses characters our maps don't cover,
    the strip output won't pass _looks_english_readable, and we
    fall through to IPA / None instead of surfacing a half-stripped
    value."""
    # Greek θ is in the digraph map (→th), so use a character that
    # really isn't in any map. Combining accents that survived NFD:
    assert (
        derive_english_shaped(
            canonical_form="x",
            language="he",
            transliteration="́́",  # bare combining accents, no base char
            pronunciation_ipa=None,
        )
        is None
    )


# ---------------------------------------------------------------------
# Per-language smoke tests against the override + strip pipeline
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "translit,language,expected",
    [
        # Sanskrit creature canon
        ("rakṣasa", "sa", "rakshasa"),
        ("nāga", "sa", "naga"),
        ("garuḍa", "sa", "garuda"),
        ("yakṣa", "sa", "yaksha"),
        # Arabic
        ("ʿifrīt", "ar", "ifrit"),
        ("marīd", "ar", "marid"),
        ("ǧinn", "ar", "jinn"),
        ("šayṭān", "ar", "shaitan"),
        ("ġūl", "ar", "ghoul"),
        # Hebrew
        ("gōlem", "he", "golem"),
        ("behēmōṯ", "he", "behemoth"),
        ("līwyāṯān", "he", "leviathan"),
        ("lilīṯ", "he", "lilith"),
        # Persian
        ("parī", "fa", "peri"),
        ("dīv", "fa", "div"),
        ("sīmurġ", "fa", "simurgh"),
        # Akkadian
        ("tiāmat", "akk", "tiamat"),
        # Egyptian (Faulkner-style)
        ("anpw", "egy", "anpu"),
    ],
)
def test_derive_canonical_creature_names_match_known_forms(
    translit: str, language: str, expected: str
) -> None:
    """Spot-check the well-known English forms for each wave-2 language.
    These are the names that drove the design (rakshasa, jinn, golem,
    ifrit, marid, shaitan, sphinx) — they must come out exactly."""
    actual = derive_english_shaped(
        canonical_form="…",
        language=language,
        transliteration=translit,
        pronunciation_ipa=None,
    )
    assert actual == expected, f"{translit!r} → {actual!r} (expected {expected!r})"


def test_arabic_sad_strips_to_s_not_sh() -> None:
    """Arabic ṣād is the same Unicode char as Sanskrit ṣa but
    English-renders as 's', not 'sh'. Pinned via the per-language
    digraph override map — Arabic-specific.
        ṣabr → sabr (Arabic patience)
        ṣalāt → salat (Arabic prayer)
        ṣaḥrā → sahra (Arabic desert)
    Without the override the rule path produced 'shabr' / 'shalat',
    affecting ~11k Arabic rows in the wave-2 backfill (caught by
    Claude review of PR #99)."""
    assert (
        derive_english_shaped(
            canonical_form="صبر",
            language="ar",
            transliteration="ṣabr",
            pronunciation_ipa=None,
        )
        == "sabr"
    )
    # Sanskrit (and other languages) keep ṣ → 'sh' as the default.
    assert (
        derive_english_shaped(
            canonical_form="राक्षस",
            language="sa",
            transliteration="rakṣas",
            pronunciation_ipa=None,
        )
        # Note: 'rakṣas' is in KNOWN_FORM_OVERRIDES → 'rakshasa'.
        # Use a Sanskrit word NOT in the overrides to exercise the rule:
    ) == "rakshasa"
    # Direct rule exercise (no override match):
    assert (
        derive_english_shaped(
            canonical_form="दृष्टि",
            language="sa",
            transliteration="dṛṣṭi",
            pronunciation_ipa=None,
        )
        == "drshti"
    )


def test_derive_falls_through_to_ipa_when_strip_unreadable_but_ipa_clean() -> None:
    """Three-way path: transliteration is present but its strip
    output keeps non-ASCII residuals (so _looks_english_readable
    rejects it), AND IPA is present and yields a clean result.
    Verifies the fallthrough chain — pre-fix this was untested."""

    # Construct a transliteration with a character no map covers
    # (Maltese 'ċ' is NOT in our digraph or single-char maps).
    # IPA is clean and yields readable output.
    res = derive_english_shaped(
        canonical_form="…",
        language="ar",
        transliteration="ċafr",  # ċ unmapped → "ċafr" survives → unreadable
        pronunciation_ipa="/d͡ʒɪnː/",
    )
    assert res == "dzhin", f"expected 'dzhin' from IPA fallback, got {res!r}"


@pytest.mark.parametrize(
    "translit,expected",
    [
        # Aramaic — caught by test-coverage agent's note that arc was
        # missing from the original parametrized smoke set.
        ("šēḏ", "shed"),
        ("shedah", "shedah"),
    ],
)
def test_derive_aramaic_creature_names(translit: str, expected: str) -> None:
    """Aramaic (`arc`) wasn't represented in the per-language smoke
    table. wyrd-ami's Aramaic creature corpus is small but
    'shed'/'shedah' (the demonic spirit) is a canonical name."""
    assert (
        derive_english_shaped(
            canonical_form="…",
            language="arc",
            transliteration=translit,
            pronunciation_ipa=None,
        )
        == expected
    )


@pytest.mark.parametrize("override_key,override_value", list(KNOWN_FORM_OVERRIDES.items()))
def test_every_known_form_override_round_trips_through_derive(
    override_key: str, override_value: str
) -> None:
    """Every (key, value) in KNOWN_FORM_OVERRIDES must:
    (1) match itself when passed as the transliteration to
    derive_english_shaped, and
    (2) produce its registered value as the english_shaped output.

    Pre-fix this only validated dict shape (lowercase keys, ASCII
    values) without exercising the lookup path. A typo in a key —
    e.g. '"behemot"' instead of '"behēmōṯ"' — would silently never
    fire."""
    actual = derive_english_shaped(
        canonical_form="…",
        language="he",  # any non-Latin lang triggers the override path
        transliteration=override_key,
        pronunciation_ipa=None,
    )
    assert actual == override_value, (
        f"override {override_key!r} should yield {override_value!r}, got {actual!r}"
    )


def test_known_form_overrides_keys_are_lowercase_or_typeable() -> None:
    """All KNOWN_FORM_OVERRIDES keys must be lowercase (the lookup
    lowercases before matching). Catches typos that would make an
    override silently never fire."""
    for key in KNOWN_FORM_OVERRIDES:
        assert key == key.lower(), f"override key {key!r} has uppercase letters"


def test_known_form_overrides_values_are_clean_ascii() -> None:
    """Override VALUES must be ASCII (the whole point is English-
    readable) and look like reasonable English words."""
    for key, value in KNOWN_FORM_OVERRIDES.items():
        assert value.isascii(), f"override value for {key!r} has non-ASCII: {value!r}"
        assert value, f"override for {key!r} is empty"
        assert _looks_english_readable(value), (
            f"override value {value!r} for {key!r} doesn't pass readability check"
        )


# ---------------------------------------------------------------------
# CLI: lexicon derive-english-shaped
# ---------------------------------------------------------------------


def _seed_etymon_with_translit(
    db: LexiconDB,
    *,
    canonical_form: str,
    language: str,
    transliteration: str | None = None,
    pronunciation_ipa: str | None = None,
) -> int:
    cur = db.conn.execute(
        """INSERT INTO etymon
           (canonical_form, language, transliteration, pronunciation_ipa)
           VALUES (?, ?, ?, ?)""",
        (canonical_form, language, transliteration, pronunciation_ipa),
    )
    return cur.lastrowid


def _english_shaped(db: LexiconDB, etymon_id: int) -> str | None:
    return db.conn.execute(
        "SELECT english_shaped FROM etymon WHERE id = ?", (etymon_id,)
    ).fetchone()["english_shaped"]


def test_cli_derive_english_shaped_dry_run_does_not_write(fresh_db: Path) -> None:
    """Dry-run mode (no --apply): walks rows + reports counts but
    leaves english_shaped NULL. Pinned because the CLI is the
    production entry point that wrote 48k rows to the live DB
    on the first PR pass."""
    with LexiconDB(fresh_db) as db:
        eid = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "derive-english-shaped", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert _english_shaped(db, eid) is None


def test_cli_derive_english_shaped_apply_writes_rows(fresh_db: Path) -> None:
    """--apply: english_shaped values are derived and UPDATEd onto the
    matching rows. End-to-end smoke covering the override path
    (jinn) + the rule path (ifrit from ʿifrīt)."""
    with LexiconDB(fresh_db) as db:
        jinn_id = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        ifrit_id = _seed_etymon_with_translit(
            db, canonical_form="عفريت", language="ar", transliteration="ʿifrīt"
        )
        # Latin-script row — must be left NULL (the CLI's SQL filter
        # excludes it via _PHASE2A_NON_LATIN_LANGS).
        oe_id = _seed_etymon_with_translit(db, canonical_form="tūn", language="old-english")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "derive-english-shaped", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert _english_shaped(db, jinn_id) == "jinn"
        assert _english_shaped(db, ifrit_id) == "ifrit"
        # Latin-script row stays NULL.
        assert _english_shaped(db, oe_id) is None


def test_cli_derive_english_shaped_language_filter_is_parameterized(fresh_db: Path) -> None:
    """The --language flag uses parameterized SQL (not f-string
    interpolation) so a malicious value can't inject UPDATEs / DROPs
    into the WHERE clause. Pre-fix the CLI built `language = '{val}'`
    via f-string. Pinned by passing a deliberately-malicious string
    that would corrupt the DB if the bug came back; verifying the
    language filter narrows correctly AND the DB stays intact."""
    with LexiconDB(fresh_db) as db:
        ar_id = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        he_id = _seed_etymon_with_translit(
            db, canonical_form="גולם", language="he", transliteration="gōlem"
        )
        db.commit()

    runner = CliRunner()
    # Sanity: --language filter narrows to one language.
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "derive-english-shaped",
            "--db",
            str(fresh_db),
            "--language",
            "ar",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert _english_shaped(db, ar_id) == "jinn"
        # The Hebrew row was filtered OUT — still NULL.
        assert _english_shaped(db, he_id) is None

    # Adversarial: a SQL-injection-shaped value is treated as a
    # literal language code (which doesn't match any row). Both
    # rows remain unchanged; no UPDATE is forged.
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "derive-english-shaped",
            "--db",
            str(fresh_db),
            "--language",
            "x'; UPDATE etymon SET english_shaped = 'pwned'; --",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        # Hebrew row is still NULL (the malicious string didn't update it).
        assert _english_shaped(db, he_id) is None
        # Arabic row keeps its earlier-written value.
        assert _english_shaped(db, ar_id) == "jinn"


def test_cli_derive_english_shaped_reshape_preserves_value_when_derive_returns_none(
    fresh_db: Path,
) -> None:
    """`--reshape` MUST NOT NULL out an existing english_shaped value
    just because the row's transliteration / IPA inputs are now empty.
    A subset of the live ~48k rows have NULL transliteration AND NULL
    IPA — derive_english_shaped returns None for those — and the CLI
    must skip the UPDATE entirely rather than overwrite a previously-
    written value with NULL.

    Pinned because the production --reshape pass runs over 133k rows
    and even a brief regression where None → NULL UPDATE'd would
    corrupt the live table."""
    with LexiconDB(fresh_db) as db:
        eid = _seed_etymon_with_translit(
            db,
            canonical_form="גג",
            language="he",
            transliteration=None,  # nothing to derive from
            pronunciation_ipa=None,
        )
        # Pre-set english_shaped to a value that came from a richer
        # earlier ingest (analogous to a row that had transliteration
        # the first time and got cleared / lost it later).
        db.conn.execute(
            "UPDATE etymon SET english_shaped = 'preserved-value' WHERE id = ?",
            (eid,),
        )
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "derive-english-shaped",
            "--db",
            str(fresh_db),
            "--apply",
            "--reshape",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        # The pre-set value survives the --reshape pass because
        # derive_english_shaped returned None (no inputs) and the
        # CLI's None-branch increments skipped_no_input rather than
        # issuing an UPDATE.
        assert _english_shaped(db, eid) == "preserved-value"


def test_cli_derive_english_shaped_reshape_redoes_non_null_rows(fresh_db: Path) -> None:
    """Default behavior leaves non-NULL rows alone; --reshape
    re-derives even rows that already have a value. Pinned so a
    future change to the override table or the rule maps can flow
    to existing rows via --reshape without manual SQL UPDATEs."""
    with LexiconDB(fresh_db) as db:
        eid = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        # Pre-set english_shaped to a stale value.
        db.conn.execute("UPDATE etymon SET english_shaped = 'STALE' WHERE id = ?", (eid,))
        db.commit()

    runner = CliRunner()

    # Default --apply: stale value is preserved (the row isn't NULL).
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "derive-english-shaped", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert _english_shaped(db, eid) == "STALE"

    # --reshape: re-derives + overwrites.
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "derive-english-shaped",
            "--db",
            str(fresh_db),
            "--apply",
            "--reshape",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert _english_shaped(db, eid) == "jinn"


# ---------------------------------------------------------------------
# derive_english_shaped_all (L3 wrapper)
# ---------------------------------------------------------------------
#
# wyrd-s9z3: pin the wrapper's gates separately from the row-by-row
# transformer. The wrapper drives the enrichment-chain entry point; if
# its language filter, reshape gate, or by_language counter shape
# regresses, the lower-level derive_english_shaped tests would still
# pass while the chain output silently shifted.


def test_derive_english_shaped_all_apply_writes_only_non_latin_rows(
    fresh_db: Path,
) -> None:
    """Default languages filter is PHASE2A_NON_LATIN_LANGS — Latin-script
    languages (old-english, latin, welsh) are SELECTed out before
    derive_english_shaped sees them. Pins the cheap-pre-filter contract:
    the wrapper avoids walking the 1.4M ModE rows whose canonical_form is
    already English-readable."""
    with LexiconDB(fresh_db) as db:
        ar = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        oe = _seed_etymon_with_translit(
            db, canonical_form="tūn", language="old-english", transliteration="tun"
        )
        db.commit()
        result = derive_english_shaped_all(db, apply=True)
        ar_shaped = _english_shaped(db, ar)
        oe_shaped = _english_shaped(db, oe)
    assert ar_shaped == "jinn"
    assert oe_shaped is None
    assert result["written"] == 1
    # candidates count reflects the SELECT scope, not the etymon table.
    assert result["candidates"] == 1


def test_derive_english_shaped_all_dry_run_skips_writes(fresh_db: Path) -> None:
    """apply=False walks the candidate set + returns counts but does NOT
    UPDATE the rows. Pins the dry-run gate parity with the apply=True
    branch (which DOES persist)."""
    with LexiconDB(fresh_db) as db:
        eid = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        db.commit()
        result = derive_english_shaped_all(db, apply=False)
    with LexiconDB(fresh_db) as db:
        assert _english_shaped(db, eid) is None
    assert result["applied"] == 0
    assert result["written"] == 1  # would-write count is reported even on dry-run


def test_derive_english_shaped_all_default_reshape_false_skips_already_shaped(
    fresh_db: Path,
) -> None:
    """reshape=False is the default — the SELECT WHERE includes
    ``english_shaped IS NULL``, so a row whose english_shaped was set
    by a prior pass is silently skipped on re-run. Pins the idempotency
    gate that lets the enrichment chain re-run cheaply."""
    with LexiconDB(fresh_db) as db:
        eid = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        # Pre-populate the column with a sentinel an override-only run
        # would have left alone.
        db.conn.execute(
            "UPDATE etymon SET english_shaped = ? WHERE id = ?",
            ("preserved-sentinel", eid),
        )
        db.commit()
        result = derive_english_shaped_all(db, apply=True)
        shaped = _english_shaped(db, eid)
    assert shaped == "preserved-sentinel"
    assert result["candidates"] == 0
    assert result["written"] == 0


def test_derive_english_shaped_all_reshape_true_revisits_populated_rows(
    fresh_db: Path,
) -> None:
    """reshape=True drops the IS NULL filter — every wave-2 row is
    re-derived. Lets an operator re-run the pass after a rule change."""
    with LexiconDB(fresh_db) as db:
        eid = _seed_etymon_with_translit(
            db, canonical_form="جن", language="ar", transliteration="ǧinn"
        )
        db.conn.execute(
            "UPDATE etymon SET english_shaped = ? WHERE id = ?",
            ("stale-value", eid),
        )
        db.commit()
        result = derive_english_shaped_all(db, apply=True, reshape=True)
        shaped = _english_shaped(db, eid)
    assert shaped == "jinn"
    assert result["candidates"] == 1


def test_derive_english_shaped_all_languages_override_restricts_select(
    fresh_db: Path,
) -> None:
    """Passing a languages tuple overrides the default. Lets the CLI's
    --language filter reuse this wrapper without copy-pasting the
    SELECT."""
    with LexiconDB(fresh_db) as db:
        _seed_etymon_with_translit(db, canonical_form="جن", language="ar", transliteration="ǧinn")
        he_id = _seed_etymon_with_translit(
            db, canonical_form="גולם", language="he", transliteration="gōlem"
        )
        db.commit()
        result = derive_english_shaped_all(db, apply=True, languages=("he",))
        ar_shaped = db.conn.execute(
            "SELECT english_shaped FROM etymon WHERE language = 'ar'"
        ).fetchone()["english_shaped"]
        he_shaped = _english_shaped(db, he_id)
    assert he_shaped == "golem"
    assert ar_shaped is None
    assert result["candidates"] == 1


def test_derive_english_shaped_all_by_language_counter_shape(
    fresh_db: Path,
) -> None:
    """``by_language`` is keyed by source-language code and counts
    only WRITTEN rows (not candidates). Pins the shape so the
    format_enrichment_run renderer can iterate it deterministically."""
    with LexiconDB(fresh_db) as db:
        _seed_etymon_with_translit(db, canonical_form="جن", language="ar", transliteration="ǧinn")
        _seed_etymon_with_translit(
            db, canonical_form="גולם", language="he", transliteration="gōlem"
        )
        # A row with no usable input — counts as candidate but not written.
        _seed_etymon_with_translit(db, canonical_form="x", language="he", transliteration=None)
        db.commit()
        result = derive_english_shaped_all(db, apply=True)
    by_lang = result["by_language"]
    assert by_lang == {"ar": 1, "he": 1}
    assert result["skipped_no_input"] == 1
    assert result["written"] == 2


def test_derive_english_shaped_all_phase2a_default_is_immutable() -> None:
    """PHASE2A_NON_LATIN_LANGS is exposed as a public constant so the
    CLI + wrapper can share it. Pinning the set membership prevents a
    drift between the wrapper's default and the CLI's --language enum."""
    assert "he" in PHASE2A_NON_LATIN_LANGS
    assert "ar" in PHASE2A_NON_LATIN_LANGS
    assert "sa" in PHASE2A_NON_LATIN_LANGS
    # Latin-script families must NOT be in the set — they'd be skipped
    # by derive_english_shaped anyway, but inclusion would bloat the
    # candidate SELECT pointlessly.
    assert "old-english" not in PHASE2A_NON_LATIN_LANGS
    assert "latin" not in PHASE2A_NON_LATIN_LANGS

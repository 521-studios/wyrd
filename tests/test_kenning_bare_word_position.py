"""wyrd-rogd.13: bare word-sequence placement (per-WORD-position stats).

Some bare standalone words have a strong word-sequence position bias —
``Saint`` is essentially always the FIRST word (Saint Albans), Latin
postpositives like ``Parva`` the LAST (Wigston Parva) — that the general
``single_usages`` pool can't express. These tests pin the whole pipeline:

* build side — ``Name.get_bare_word_positions`` tally + the
  ``bare_word_positions`` key ``proportions_from`` emits (RAW counts);
* L4 round-trip — ``proportions_bare_word_position`` table emit + adapter
  read (and graceful degradation on pre-rogd.13 DBs);
* load side — the naked↔structured threshold
  (``WYRD_BARE_POSITION_THRESHOLD``) and the per-word-position buckets
  ``MeaningGenerator.load_bare_word_positions`` registers;
* vector path — ``_flatten_struct_slots`` deriving ``wp-*`` bucket keys
  from the struct's word order, ``_resolve_slot_usage_frequency``'s
  legacy fallback, and the end-to-end placement guarantee (a structured
  word never sampled at a word position it isn't attested in; naked
  words eligible everywhere).

NOTE the two distinct position vocabularies: the WITHIN-WORD morpheme
position (D39/D40 dashes: ``pre``/``inner``/``post``/``bare``) vs the
WITHIN-NAME word position this feature adds (which word slot of a
multi-word name). Bare words have morpheme position ``bare`` and gain a
word position.
"""

from __future__ import annotations

import random
import sqlite3

from wyrd.generators.kenning.lexicon.proportions_builder import proportions_from
from wyrd.generators.kenning.lexicon.runtime_db_export import _insert_bare_word_positions
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.name import Name
from wyrd.generators.kenning.runtime.proportions import (
    MeaningGenerator,
    _flatten_struct_slots,
    bare_position_threshold,
    load_proportions,
)
from wyrd.generators.kenning.runtime.runtime_db_adapter import (
    _read_proportions_bare_word_positions,
)
from wyrd.generators.kenning.runtime.vector_name_select import (
    _resolve_slot_usage_frequency,
)
from wyrd.generators.kenning.runtime.word import word_position_for
from wyrd.generators.kenning.vectors.schemas import (
    EligibilityGate,
    EmpiricalPriors,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
)

# ---- helpers ---------------------------------------------------------------


def _meaning(usage: str, tags=None) -> Meaning:
    return Meaning(
        usage=usage,
        tags=tags if tags is not None else ["x"],
        meanings=["gloss"],
        sources={"old_english": [usage.lower().replace("-", "")]},
    )


def _bare_db(*usages: str) -> dict[str, list[Meaning]]:
    return {u: [_meaning(u)] for u in usages}


# ---- word_position_for -----------------------------------------------------


def test_word_position_for_single_word_name_has_no_position():
    assert word_position_for(0, 1) is None


def test_word_position_for_two_word_name_is_pre_post():
    assert word_position_for(0, 2) == "pre"
    assert word_position_for(1, 2) == "post"


def test_word_position_for_three_word_name_interior_is_inner():
    assert word_position_for(0, 3) == "pre"
    assert word_position_for(1, 3) == "inner"
    assert word_position_for(2, 3) == "post"


# ---- Name.get_bare_word_positions -------------------------------------------


def test_bare_word_positions_two_word_name():
    db = _bare_db("Mount", "Pleasant")
    name = Name("Mount Pleasant")
    name.find_meaning(db)
    assert name.get_bare_word_positions() == {("pre", "mount"), ("post", "pleasant")}


def test_bare_word_positions_three_word_name_records_inner():
    db = _bare_db("Newton", "Saint", "Loe")
    name = Name("Newton Saint Loe")
    name.find_meaning(db)
    assert name.get_bare_word_positions() == {
        ("pre", "newton"),
        ("inner", "saint"),
        ("post", "loe"),
    }


def test_bare_word_positions_single_word_name_is_empty():
    """No solo case — a 1-word name carries no word-sequence evidence."""
    db = _bare_db("Bath")
    name = Name("Bath")
    name.find_meaning(db)
    assert name.get_bare_word_positions() == set()


def test_bare_word_positions_skips_multi_morpheme_words():
    """Only BARE (single-morpheme) words tally; a compound word ('Higham'
    = High- + -ham) records nothing here — its morphemes are pre/post
    WITHIN the word, not bare."""
    db = {
        "High-": [_meaning("High-")],
        "-ham": [_meaning("-ham")],
        "Green": [_meaning("Green")],
    }
    name = Name("Higham Green")
    name.find_meaning(db)
    assert name.get_bare_word_positions() == {("post", "green")}


def test_bare_word_positions_repeated_word_records_both_positions():
    db = _bare_db("Newton")
    name = Name("Newton Newton")
    name.find_meaning(db)
    assert name.get_bare_word_positions() == {("pre", "newton"), ("post", "newton")}


def test_bare_word_positions_case_twins_record_one_observation():
    """The bundle routinely stores case twins of one surface (``Ghyll`` /
    ``ghyll``) that both parse the same word. The tally records the
    morpheme IDENTITY (bare lowercase surface, D40) — one observation per
    name, not one per stored variant — so twins can't double-count AND
    split the naked↔structured threshold."""
    db = {"Ghyll": [_meaning("Ghyll")], "ghyll": [_meaning("ghyll")], "Ash": [_meaning("Ash")]}
    name = Name("Ash Ghyll")
    name.find_meaning(db)
    assert name.get_bare_word_positions() == {("pre", "ash"), ("post", "ghyll")}


# ---- proportions_from emit ---------------------------------------------------


def _decomposed(name_str: str, db) -> Name:
    name = Name(name_str)
    name.find_meaning(db)
    return name


def test_proportions_from_emits_raw_bare_word_positions():
    db = _bare_db("Mount", "Pleasant", "Green")
    names = [
        _decomposed("Mount Pleasant", db),
        _decomposed("Mount Green", db),
        _decomposed("Green", db),  # solo: feeds single_usages, NOT positions
    ]
    out = proportions_from(names)
    assert out["bare_word_positions"] == {
        "post": {"green": 1, "pleasant": 1},
        "pre": {"mount": 2},
    }
    # the solo observation still lands in the general bare pool
    assert out["single_usages"]["Green"] == 2  # one solo + one positional


def test_proportions_from_no_multi_word_names_emits_empty_positions():
    db = _bare_db("Bath")
    out = proportions_from([_decomposed("Bath", db)])
    assert out["bare_word_positions"] == {}


# ---- L4 emit + adapter round-trip --------------------------------------------


def _bare_position_table_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE proportions_bare_word_position ("
        "culture TEXT NOT NULL, position TEXT NOT NULL, "
        "usage_key TEXT NOT NULL, weight INTEGER NOT NULL, "
        "PRIMARY KEY (culture, position, usage_key))"
    )
    return conn


def test_bare_word_positions_l4_round_trip():
    conn = _bare_position_table_db()
    written = _insert_bare_word_positions(
        conn,
        "english",
        {"pre": {"Mount": 2, "Zero": 0}, "post": {"Pleasant": 1}},
    )
    assert written == 2  # the zero-weight row is skipped
    out = _read_proportions_bare_word_positions(conn, "english")
    assert out == {"post": {"Pleasant": 1}, "pre": {"Mount": 2}}
    # other cultures see nothing
    assert _read_proportions_bare_word_positions(conn, "welsh") == {}


def test_adapter_degrades_gracefully_without_the_table():
    """Pre-rogd.13 L4 DBs have no proportions_bare_word_position table —
    the adapter returns {} so the runtime keeps legacy behavior."""
    conn = sqlite3.connect(":memory:")
    assert _read_proportions_bare_word_positions(conn, "english") == {}


# ---- threshold env resolution -------------------------------------------------


def test_threshold_defaults_to_three(monkeypatch):
    monkeypatch.delenv("WYRD_BARE_POSITION_THRESHOLD", raising=False)
    assert bare_position_threshold() == 3


def test_threshold_reads_env(monkeypatch):
    monkeypatch.setenv("WYRD_BARE_POSITION_THRESHOLD", "7")
    assert bare_position_threshold() == 7


def test_threshold_malformed_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WYRD_BARE_POSITION_THRESHOLD", "lots")
    assert bare_position_threshold() == 3


def test_threshold_clamps_to_one(monkeypatch):
    """0 would make every bare word 'structured' — including words with NO
    positional observations, which would then be eligible nowhere."""
    monkeypatch.setenv("WYRD_BARE_POSITION_THRESHOLD", "0")
    assert bare_position_threshold() == 1


# ---- MeaningGenerator.load_bare_word_positions ---------------------------------


def _bucket_usages(mg: MeaningGenerator, key: tuple) -> dict[str, float]:
    gen = mg.generators.get(key)
    return dict(gen.elements) if gen else {}


def test_structured_word_only_in_attested_position_buckets():
    db = _bare_db("Parva", "Great", "Mount")
    single = {"Parva": 5, "Great": 5, "Mount": 1}
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts(single, "single")
    mg.load_bare_word_positions(
        {"pre": {"Great": 5, "Mount": 1}, "post": {"Parva": 5}},
        single,
        threshold=3,
    )
    pre = _bucket_usages(mg, ("bare", "single", "wp-pre"))
    inner = _bucket_usages(mg, ("bare", "single", "wp-inner"))
    post = _bucket_usages(mg, ("bare", "single", "wp-post"))
    # structured words carry their per-position weight, only where attested
    assert pre["Great"] == 5
    assert "Great" not in inner and "Great" not in post
    assert post["Parva"] == 5
    assert "Parva" not in pre and "Parva" not in inner
    # naked word (1 obs < threshold) is everywhere at its GENERAL weight
    assert pre["Mount"] == inner["Mount"] == post["Mount"] == 1
    # the general bucket is untouched (the legacy fallback target)
    general = _bucket_usages(mg, ("bare", "single"))
    assert general == {"Parva": 5, "Great": 5, "Mount": 1}


def test_naked_word_general_weight_beats_positional_count():
    """A naked word's weight in every wp bucket is its single_usages
    weight (which includes solo-name observations), not its thin
    positional count."""
    db = _bare_db("Bath")
    single = {"Bath": 9}  # 8 solo observations + 1 positional
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts(single, "single")
    mg.load_bare_word_positions({"pre": {"Bath": 1}}, single, threshold=3)
    assert _bucket_usages(mg, ("bare", "single", "wp-pre"))["Bath"] == 9
    assert _bucket_usages(mg, ("bare", "single", "wp-post"))["Bath"] == 9


def test_threshold_boundary_structures_at_exactly_threshold():
    db = _bare_db("Edge")
    single = {"Edge": 3}
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts(single, "single")
    # 2 + 1 = 3 total positional observations == threshold → structured
    mg.load_bare_word_positions({"pre": {"Edge": 2}, "post": {"Edge": 1}}, single, threshold=3)
    assert _bucket_usages(mg, ("bare", "single", "wp-pre"))["Edge"] == 2
    assert _bucket_usages(mg, ("bare", "single", "wp-post"))["Edge"] == 1
    assert "Edge" not in _bucket_usages(mg, ("bare", "single", "wp-inner"))


def test_case_twin_form_keys_aggregate_for_the_threshold():
    """Operator-supplied proportions JSON may carry stored-form keys
    (case twins). The naked↔structured decision aggregates by surface —
    'Ghyll' 2 + 'ghyll' 2 = 4 >= 3 → structured post-only — so split
    keys can't dilute the threshold. Both form keys enter only the
    attested position's bucket (the slot scorer sums them by surface)."""
    db = {"Ghyll": [_meaning("Ghyll")], "ghyll": [_meaning("ghyll")]}
    single = {"Ghyll": 2, "ghyll": 2}
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts(single, "single")
    mg.load_bare_word_positions({"post": {"Ghyll": 2, "ghyll": 2}}, single, threshold=3)
    post = _bucket_usages(mg, ("bare", "single", "wp-post"))
    assert post == {"Ghyll": 2, "ghyll": 2}
    assert _bucket_usages(mg, ("bare", "single", "wp-pre")) == {}
    assert _bucket_usages(mg, ("bare", "single", "wp-inner")) == {}


def test_empty_stats_register_no_wp_buckets():
    db = _bare_db("Bath")
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts({"Bath": 1}, "single")
    mg.load_bare_word_positions({}, {"Bath": 1}, threshold=3)
    assert not any("wp-" in str(k) for k in mg.generators)


def test_saint_flagged_bare_words_bucket_with_their_flag():
    """The wp dimension applies uniformly across bare bucket families —
    a saint-flagged bare word lands in ('bare', 'saint', 'single',
    'wp-…'), matching the struct keys word_to_key produces.

    The fixture mirrors the live bundle's Saint subject: the SAINT flag in
    ``Meaning.key()`` is USAGE-derived (surface == 'saint'), while the
    tags stay common-noun (``religious``/``social``) — a ``saint``-TAGGED
    pure-proper-noun subject would be base-pool excluded (wyrd-57d8) and
    never reach any frequency bucket."""
    saint = Meaning(
        usage="Saint-",
        tags=["religious", "social"],
        meanings=["holy"],
        sources={"old_french": ["saint"]},
    )
    db = {"Saint-": [saint]}
    single = {"Saint": 5}
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts(single, "single")
    mg.load_bare_word_positions({"pre": {"Saint": 5}}, single, threshold=3)
    assert _bucket_usages(mg, ("bare", "saint", "single", "wp-pre"))["Saint"] == 5
    assert ("bare", "saint", "single", "wp-post") not in mg.generators


# ---- _flatten_struct_slots wp-key derivation -----------------------------------


def test_flatten_extends_bare_slots_with_word_position():
    struct = ((("bare", "single"),), (("bare", "single"),))
    positions, qualifiers, bucket_keys = _flatten_struct_slots(struct)
    assert positions == ["bare", "bare"]
    assert qualifiers == [None, None]
    assert bucket_keys == [
        ("bare", "single", "wp-pre"),
        ("bare", "single", "wp-post"),
    ]


def test_flatten_three_word_struct_marks_interior_inner():
    struct = ((("bare", "single"),), (("bare", "saint", "single"),), (("bare", "single"),))
    _, qualifiers, bucket_keys = _flatten_struct_slots(struct)
    assert qualifiers == [None, "saint", None]
    assert bucket_keys[1] == ("bare", "saint", "single", "wp-inner")


def test_flatten_leaves_non_bare_slots_unextended():
    """A multi-word struct mixing a compound word with a bare word only
    extends the BARE slot — pre/post morpheme slots keep their keys
    (word position is a bare-word concept; spec scope is bare only)."""
    struct = ((("pre",), ("post",)), (("bare", "single"),))
    positions, _, bucket_keys = _flatten_struct_slots(struct)
    assert positions == ["pre", "post", "bare"]
    assert bucket_keys == [("pre",), ("post",), ("bare", "single", "wp-post")]


def test_flatten_single_word_struct_is_unchanged():
    """No solo case: a 1-word struct's bare slot keeps the un-extended
    key (single-morpheme structures are load-excluded anyway, wyrd-g1hj —
    this pins the no-derivation contract for any that slip through)."""
    struct = ((("bare", "single"),),)
    _, _, bucket_keys = _flatten_struct_slots(struct)
    assert bucket_keys == [("bare", "single")]


# ---- _resolve_slot_usage_frequency fallback -------------------------------------


def test_resolve_uses_wp_bucket_when_present():
    freq_map = {
        ("bare", "single"): {"Parva": 5.0, "Great": 5.0},
        ("bare", "single", "wp-pre"): {"Great": 5.0},
    }
    out = _resolve_slot_usage_frequency(freq_map, ("bare", "single", "wp-pre"))
    assert out == {"Great": 5.0}


def test_resolve_falls_back_to_general_bucket_without_wp_data():
    """Legacy bundle (or a bucket family with no positional stats): the
    wp-extended key misses and resolution falls back to the general
    bare bucket — pre-rogd.13 behavior, bit-stable."""
    freq_map = {("bare", "single"): {"Parva": 5.0, "Great": 5.0}}
    out = _resolve_slot_usage_frequency(freq_map, ("bare", "single", "wp-pre"))
    assert out == {"Parva": 5.0, "Great": 5.0}


def test_resolve_non_wp_key_missing_still_returns_empty():
    """The wyrd-bol9 contract is unchanged for non-wp keys: a missing
    bucket means unreachable, NOT a fallback."""
    freq_map = {("bare", "single"): {"Parva": 5.0}}
    assert _resolve_slot_usage_frequency(freq_map, ("pre",)) == {}


def test_resolve_opted_out_map_still_returns_none():
    assert _resolve_slot_usage_frequency(None, ("bare", "single", "wp-pre")) is None


# ---- end-to-end: placement through the vector path ------------------------------


def _two_bare_word_data() -> dict:
    """Synthetic culture: 2-word names of standalone words, where Great is
    pre-only structured, Parva post-only structured, Mount naked."""
    return {
        "usages": {},
        "single_usages": {"Great": 5, "Parva": 5, "Mount": 1},
        "structures": [
            {
                "proportion": 10,
                "words": [[{"location": "bare"}], [{"location": "bare"}]],
            }
        ],
        "tag_marginal": {},
        "tag_cooccurrence": {},
        "attested_languages": {},
        "bare_word_positions": {"pre": {"Great": 5, "Mount": 1}, "post": {"Parva": 5}},
    }


def _request() -> RequestVector:
    return RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="test", semantic_tags={"x": 1.0}),
        weights=ScoringWeights(),
    )


def _generate_names(data, seeds=None) -> list[list[str]]:
    seeds = range(40) if seeds is None else seeds
    db = _bare_db("Great", "Parva", "Mount")
    gen = load_proportions(data, db, {})
    names = []
    for seed in seeds:
        new_name = gen.select_via_vector(
            random.Random(seed),
            request=_request(),
            priors=EmpiricalPriors(),
        )
        assert new_name is not None
        names.append(str(new_name).split(" "))
    return names


def test_structured_words_never_sampled_at_unattested_word_position():
    """The placement guarantee: across many seeds, Parva (post-only)
    never opens a name and Great (pre-only) never closes one. Mount
    (naked) stays eligible at both."""
    names = _generate_names(_two_bare_word_data())
    firsts = {words[0] for words in names}
    lasts = {words[-1] for words in names}
    assert "Parva" not in firsts
    assert "Great" not in lasts
    # the eligible words actually surface (the pools aren't degenerate)
    assert "Great" in firsts
    assert "Parva" in lasts
    assert "Mount" in firsts | lasts


def test_legacy_data_without_positions_keeps_any_slot_behavior():
    """Same culture WITHOUT the positional stats: the wp keys fall back
    to the general bare bucket and every word is eligible at every word
    position — exactly pre-rogd.13 output."""
    data = _two_bare_word_data()
    del data["bare_word_positions"]
    names = _generate_names(data)
    firsts = {words[0] for words in names}
    lasts = {words[-1] for words in names}
    assert "Parva" in firsts  # the bug this feature fixes…
    assert "Great" in lasts  # …still present on legacy data, by design

"""Tests for ``wyrd kenning lexicon export-runtime-db --dev`` (wyrd-d90t PR 2).

PR 2 adds:
* ``select_dev_subset()`` — deterministic top-N-by-weight slice of the
  L4 inputs.
* ``--dev`` CLI flag that runs the slice, pins ``built_at`` +
  ``source_lexicon_db`` to fixed sentinels for byte-stable output, and
  rejects non-default upstream filters that would compromise the seed's
  reproducibility across operators.
* ``wyrd/generators/kenning/data/seed-runtime.db`` — the committed
  dev-mode seed; tests in this module verify the schema version + dev
  sentinels + every-table-has-rows.

The full "drift detection" claim from the d90t design (PR 2 spec):
byte-equal re-emit verification against an L3 fixture is not exercised
here — committing a fixture L3 alongside the seed would double the
binary footprint without proportional fidelity gain. Schema-version
drift is caught by ``test_committed_seed_carries_current_schema_version``;
data drift is left to the operator's local ``--dev`` regeneration cycle.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.cli.lexicon.export_runtime_db import SCHEMA_VERSION
from wyrd.generators.kenning.lexicon.runtime_db_export import (
    DEV_BUILT_AT,
    DEV_SOURCE_LEXICON_SENTINEL,
    _top_n_by_weight,
    select_dev_subset,
    write_runtime_db,
)

# ---------- select_dev_subset ----------


def _subject(usage: str, language: str = "old_english") -> dict:
    return {
        "meaning": ["Test"],
        "modifier_tags": [],
        "modifier_type": None,
        "words": [{"modern_usage": usage, language: [usage]}],
    }


def _proportions(usages: dict[str, int], single_usages: dict[str, int] | None = None) -> dict:
    return {
        "usages": usages,
        "single_usages": single_usages or {},
        "structures": [{"proportion": 1, "words": [[{"location": "pre"}]]}],
        "tag_marginal": {"water": 1},
        "tag_cooccurrence": {"water|tree": 1},
    }


def test_top_n_by_weight_orders_weight_desc_then_alpha() -> None:
    """Tiebreaker on weight ties is alphabetical key, so re-runs against
    the same dict yield the same top-N regardless of insertion order."""
    items = {"-zzz": 5, "-aaa": 10, "-bbb": 10, "-ccc": 3}
    assert _top_n_by_weight(items, 3) == [("-aaa", 10), ("-bbb", 10), ("-zzz", 5)]


def test_top_n_by_weight_caps_at_n() -> None:
    items = {f"-key{i}": i for i in range(20)}
    out = _top_n_by_weight(items, 5)
    assert len(out) == 5
    # Highest weights first: 19, 18, 17, 16, 15
    assert [w for _, w in out] == [19, 18, 17, 16, 15]


def test_select_dev_subset_drops_unreferenced_meanings() -> None:
    """Any subject whose every word's modern_usage falls outside the
    per-culture top-N union is dropped entirely from the L4."""
    subjects = [
        _subject("-ham-"),  # kept (in top-N below)
        _subject("-orphan-"),  # dropped (not in any top-N)
    ]
    proportions = {
        "english": _proportions({"-ham-": 100, "-ton-": 50}),
    }
    subs_out, _, _, props_out = select_dev_subset(
        subjects,
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=2,
    )
    assert [s["words"][0]["modern_usage"] for s in subs_out] == ["-ham-"]
    assert set(props_out["english"]["usages"].keys()) == {"-ham-", "-ton-"}


def test_select_dev_subset_keeps_words_whose_usage_appears_in_any_culture() -> None:
    """A morpheme used by Welsh-only generation should survive the
    cross-culture union of keep-sets, even though English's top-N
    doesn't include it."""
    subjects = [_subject("caer-", "celtic_mix")]
    proportions = {
        "english": _proportions({"-ham-": 100}),
        "welsh": _proportions({"caer-": 50}),
    }
    subs_out, _, _, _ = select_dev_subset(
        subjects,
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=1,
    )
    assert [s["words"][0]["modern_usage"] for s in subs_out] == ["caer-"]


def test_select_dev_subset_prunes_unrenderable_structure_after_trim(caplog) -> None:
    """wyrd-9eqk: a structure whose only filler usage is trimmed by the
    top-N cut would render an empty name. select_dev_subset must drop such
    fully-unrenderable structures (and warn), while keeping a structure
    that still has a fillable slot. Guards against the trimming-induced
    orphan class recurring on a future seed re-export."""
    import logging

    subjects = [
        # high-weight pre morpheme — survives top_n=1
        {
            "meaning": ["Aa"],
            "modifier_tags": ["topography"],
            "modifier_type": None,
            "words": [{"modern_usage": "Aa-", "old_english": ["aa"]}],
        },
        # low-weight bare PERSONAL NAME — trimmed at top_n=1, so the
        # ('bare','name','single') bucket it would fill goes unbuilt
        {
            "meaning": ["Bob"],
            "modifier_tags": ["male name"],
            "modifier_type": None,
            "words": [{"modern_usage": "Bob", "old_english": ["bob"]}],
        },
    ]
    proportions = {
        "english": {
            "usages": {"Aa-": 10},
            "single_usages": {"Aa-": 5, "Bob": 1},
            "structures": [
                # partially renderable (pre slot filled by Aa-) — KEEP
                {"proportion": 10, "words": [[{"location": "pre"}, {"location": "post"}]]},
                # fully unrenderable once Bob is trimmed — DROP (orphan)
                {"proportion": 1, "words": [[{"location": "bare", "name": True}]]},
            ],
            "tag_marginal": {},
            "tag_cooccurrence": {},
            "attested_languages": {},
        }
    }
    with caplog.at_level(
        logging.WARNING, logger="wyrd.generators.kenning.lexicon.runtime_db_export"
    ):
        _, _, _, props_out = select_dev_subset(
            subjects,
            fantasy_morphemes={},
            canonical_decompositions={},
            proportions_by_culture=proportions,
            top_n_per_culture=1,
        )
    structs = props_out["english"]["structures"]
    # The bare-name orphan is gone; the partially-renderable pre+post stays.
    assert {"proportion": 1, "words": [[{"location": "bare", "name": True}]]} not in structs
    assert any(
        w == [{"location": "pre"}, {"location": "post"}] for s in structs for w in [s["words"][0]]
    )
    assert any("wyrd-9eqk" in r.getMessage() for r in caplog.records)


def test_select_dev_subset_keeps_renderable_structures_silently(caplog) -> None:
    """No-op + silent on a clean build: every structure whose slots are
    fillable by a kept usage survives, and no wyrd-9eqk warning fires."""
    import logging

    subjects = [
        {
            "meaning": ["Aa"],
            "modifier_tags": ["topography"],
            "modifier_type": None,
            "words": [{"modern_usage": "Aa-", "old_english": ["aa"]}],
        },
    ]
    proportions = {
        "english": {
            "usages": {"Aa-": 10},
            "single_usages": {"Aa-": 5},
            "structures": [{"proportion": 10, "words": [[{"location": "pre"}]]}],
            "tag_marginal": {},
            "tag_cooccurrence": {},
            "attested_languages": {},
        }
    }
    with caplog.at_level(
        logging.WARNING, logger="wyrd.generators.kenning.lexicon.runtime_db_export"
    ):
        _, _, _, props_out = select_dev_subset(
            subjects,
            fantasy_morphemes={},
            canonical_decompositions={},
            proportions_by_culture=proportions,
            top_n_per_culture=1,
        )
    assert props_out["english"]["structures"] == [
        {"proportion": 10, "words": [[{"location": "pre"}]]}
    ]
    assert not any("wyrd-9eqk" in r.getMessage() for r in caplog.records)


def test_select_dev_subset_prune_preserves_order_and_is_per_culture(caplog) -> None:
    """wyrd-9eqk: the prune is an order-preserving filter (the committed
    seed must stay byte-stable) and per-culture isolated. Keep A, drop the
    orphan B, keep C → surviving list is exactly [A, C] in source order;
    a second clean culture is untouched and draws no warning; the warning
    names only the pruned culture."""
    import logging

    subjects = [
        {
            "meaning": ["Aa"],
            "modifier_tags": ["topography"],
            "modifier_type": None,
            "words": [{"modern_usage": "Aa-", "old_english": ["aa"]}],
        },
        {
            "meaning": ["Bob"],
            "modifier_tags": ["male name"],
            "modifier_type": None,
            "words": [{"modern_usage": "Bob", "old_english": ["bob"]}],
        },
        {
            "meaning": ["Caer"],
            "modifier_tags": ["topography"],
            "modifier_type": None,
            "words": [{"modern_usage": "caer-", "celtic_mix": ["caer"]}],
        },
    ]
    struct_a = {"proportion": 9, "words": [[{"location": "pre"}, {"location": "post"}]]}
    struct_b = {"proportion": 2, "words": [[{"location": "bare", "name": True}]]}
    struct_c = {"proportion": 5, "words": [[{"location": "pre"}]]}
    proportions = {
        "english": {
            "usages": {"Aa-": 10},
            "single_usages": {"Aa-": 5, "Bob": 1},  # Bob trimmed at top_n=1
            "structures": [struct_a, struct_b, struct_c],  # B is the orphan
            "tag_marginal": {},
            "tag_cooccurrence": {},
            "attested_languages": {},
        },
        "welsh": {
            "usages": {"caer-": 7},
            "single_usages": {"caer-": 3},
            "structures": [{"proportion": 7, "words": [[{"location": "pre"}]]}],  # renderable
            "tag_marginal": {},
            "tag_cooccurrence": {},
            "attested_languages": {},
        },
    }
    with caplog.at_level(
        logging.WARNING, logger="wyrd.generators.kenning.lexicon.runtime_db_export"
    ):
        _, _, _, props_out = select_dev_subset(
            subjects,
            fantasy_morphemes={},
            canonical_decompositions={},
            proportions_by_culture=proportions,
            top_n_per_culture=1,
        )
    # Order preserved: orphan B dropped, A and C kept in their source order.
    assert props_out["english"]["structures"] == [struct_a, struct_c]
    # Other culture untouched.
    assert props_out["welsh"]["structures"] == [{"proportion": 7, "words": [[{"location": "pre"}]]}]
    # Exactly one warning, naming the pruned culture and not the clean one.
    cj_warnings = [r.getMessage() for r in caplog.records if "wyrd-9eqk" in r.getMessage()]
    assert len(cj_warnings) == 1
    assert "'english'" in cj_warnings[0]
    assert "welsh" not in cj_warnings[0]


def test_select_dev_subset_passes_fantasy_and_canonical_through_sorted() -> None:
    """Fantasy + canonical decompositions aren't trimmed (they're small
    enough that the seed shouldn't lose coverage of common names) but
    ARE re-sorted by key so the SQL insertion order is byte-stable
    across operators regardless of source-dict iteration order."""
    fantasy = {"djinni": {"input_name": "Djinni"}, "harpy": {"input_name": "Harpy"}}
    canonical = {
        "Stratford": {"signature": "abc", "source": "scholar"},
        "Birmingham": {"signature": "def", "source": "scholar"},
    }
    _, fantasy_out, canon_out, _ = select_dev_subset(
        subjects=[],
        fantasy_morphemes=fantasy,
        canonical_decompositions=canonical,
        proportions_by_culture={},
        top_n_per_culture=1,
    )
    # Same data, sorted keys — pin the deterministic order.
    assert list(fantasy_out.keys()) == ["djinni", "harpy"]
    assert list(canon_out.keys()) == ["Birmingham", "Stratford"]
    assert fantasy_out == fantasy
    assert canon_out == canonical


def test_select_dev_subset_keeps_partial_survivor_words() -> None:
    """A multi-word subject where SOME words pass the keep-set and others
    don't must keep the surviving words and drop the rejected ones in
    place — without losing the subject. Catches an in-place-mutation
    regression that the drop-everything / keep-everything tests wouldn't
    see."""
    subjects = [
        {
            "meaning": ["Test"],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [
                {"modern_usage": "-ham-", "old_english": ["ham"]},
                {"modern_usage": "-orphan-", "old_english": ["orphan"]},
                {"modern_usage": "-ton-", "old_english": ["ton"]},
            ],
        }
    ]
    proportions = {"english": _proportions({"-ham-": 100, "-ton-": 50})}
    subs_out, _, _, _ = select_dev_subset(
        subjects,
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=10,
    )
    assert len(subs_out) == 1
    surviving = [w["modern_usage"] for w in subs_out[0]["words"]]
    assert surviving == ["-ham-", "-ton-"]


def test_select_dev_subset_keep_from_single_usages() -> None:
    """``single_usages`` (single-element place names) must contribute to
    the keep-set independently — a morpheme that only appears in
    single_usages should still survive the meaning filter."""
    subjects = [_subject("-castle")]
    proportions = {
        "english": {
            "usages": {"-other-": 100},
            "single_usages": {"-castle": 50},
            "structures": [],
            "tag_marginal": {},
            "tag_cooccurrence": {},
        }
    }
    subs_out, _, _, props_out = select_dev_subset(
        subjects,
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=5,
    )
    assert [s["words"][0]["modern_usage"] for s in subs_out] == ["-castle"]
    assert "-castle" in props_out["english"]["single_usages"]


def test_select_dev_subset_drops_subject_with_no_surviving_words() -> None:
    """When every word in a subject is filtered out, the whole subject
    is dropped — keeps the meaning row count tight."""
    subjects = [
        {
            "meaning": ["Test"],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [
                {"modern_usage": "-out-1", "old_english": ["x"]},
                {"modern_usage": "-out-2", "old_english": ["y"]},
            ],
        }
    ]
    proportions = {"english": _proportions({"-ham-": 100})}
    subs_out, _, _, _ = select_dev_subset(
        subjects,
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=10,
    )
    assert subs_out == []


# ---------- --dev CLI flag ----------


def _seed_minimal_lexicon(db_path: Path) -> None:
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema

    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(id="skeat-1901", title="Skeat 1901", year=1901)
        db.commit()


def _write_proportions_fixture(directory: Path, culture: str = "english") -> None:
    import json

    payload = _proportions({"-ham-": 100, "-ton-": 50}, {"-castle": 20})
    (directory / f"{culture}_proportions.json").write_text(json.dumps(payload))


def test_dev_flag_pins_built_at_sentinel(tmp_path: Path) -> None:
    """--dev replaces the wall-clock built_at with DEV_BUILT_AT so the
    seed-runtime.db is byte-stable across emits."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(str(out_path))
    try:
        built_at = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'built_at'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert built_at == DEV_BUILT_AT


def test_dev_flag_produces_byte_stable_output(tmp_path: Path) -> None:
    """Two --dev emits with identical inputs must produce byte-equal
    output — the foundation of the seed-runtime.db drift detection."""
    db_path = tmp_path / "lexicon.db"
    out1 = tmp_path / "runtime1.db"
    out2 = tmp_path / "runtime2.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    for out in (out1, out2):
        result = runner.invoke(
            cli_root,
            [
                "lexicon",
                "export-runtime-db",
                "--dev",
                "--db",
                str(db_path),
                "--proportions-dir",
                str(tmp_path),
                "--output",
                str(out),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    assert out1.read_bytes() == out2.read_bytes()


def test_dev_flag_pins_source_lexicon_sentinel(tmp_path: Path) -> None:
    """--dev replaces the resolved-absolute lexicon path with a sentinel
    so the seed-runtime.db is byte-stable across operators / CI."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(str(out_path))
    try:
        source = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'source_lexicon_db'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert source == DEV_SOURCE_LEXICON_SENTINEL


# ---------- generation_subset (wyrd-ukq0): the production cold-start trim ----------


def test_select_dev_subset_none_keeps_all_referenced_usages() -> None:
    """``top_n_per_culture=None`` (--generation-subset) keeps EVERY proportion-
    referenced usage — not just the top N — so a low-weight usage that
    ``top_n=1`` would drop survives, while a truly unreferenced subject is
    still dropped. This is the load-bearing difference from the --dev slice."""
    subjects = [
        _subject("-ham-"),  # high weight
        _subject("-ton-"),  # low weight — top_n=1 would drop it, None keeps it
        _subject("-orphan-"),  # unreferenced — dropped either way
    ]
    proportions = {"english": _proportions({"-ham-": 100, "-ton-": 50})}
    subs_out, _, _, props_out = select_dev_subset(
        subjects,
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_by_culture=proportions,
        top_n_per_culture=None,
    )
    assert {s["words"][0]["modern_usage"] for s in subs_out} == {"-ham-", "-ton-"}
    assert set(props_out["english"]["usages"]) == {"-ham-", "-ton-"}


def test_write_runtime_db_rejects_dev_and_generation_subset(tmp_path: Path) -> None:
    """The two subset modes are mutually exclusive — guarded at the library
    layer (defense-in-depth backstop for non-CLI callers), raised before any
    expensive proportions work."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        write_runtime_db(
            output_path=tmp_path / "out.db",
            subjects=[],
            fantasy_morphemes={},
            canonical_decompositions={},
            proportions_dir=tmp_path,
            source_lexicon_db=tmp_path / "lexicon.db",
            dev_subset=True,
            generation_subset=True,
        )


def test_generation_subset_carries_real_metadata(tmp_path: Path) -> None:
    """--generation-subset stamps REAL built_at + source_lexicon_db (NOT the
    --dev byte-stability sentinels) — the production DB must be self-identifying."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--generation-subset",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(str(out_path))
    try:
        meta = dict(conn.execute("SELECT key, value FROM bundle_metadata").fetchall())
    finally:
        conn.close()

    assert meta["built_at"] != DEV_BUILT_AT
    assert meta["source_lexicon_db"] != DEV_SOURCE_LEXICON_SENTINEL
    assert str(db_path.resolve()) == meta["source_lexicon_db"]


def test_cli_rejects_dev_and_generation_subset_together(tmp_path: Path) -> None:
    """--dev + --generation-subset is a friendly UsageError (exit≠0, no
    traceback), parity with the --dev filter rejection."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--generation-subset",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.db"),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_dev_flag_rejects_non_default_filters(tmp_path: Path) -> None:
    """--dev must hard-fail when combined with any non-default upstream
    filter — silently honoring a custom --min-witnesses would produce a
    seed that differs from the committed one without the operator
    noticing."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--min-witnesses",
            "5",  # non-canonical
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert "--dev requires canonical defaults" in result.output
    assert "--min-witnesses=5" in result.output


def test_dev_flag_rejects_no_include_rando(tmp_path: Path) -> None:
    """Same guard covers boolean filter overrides — not just numeric."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--no-include-rando",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert "--no-include-rando" in result.output


def test_dev_flag_rejects_rando_min_corroborators(tmp_path: Path) -> None:
    """wyrd-fssn: --rando-min-corroborators is a new upstream filter on
    the rando-port admit path; the --dev guard must reject non-default
    values for it the same way it rejects the other filter overrides
    (so the committed seed-runtime.db stays byte-stable across
    operators regardless of which fssn-era flag they touch)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--rando-min-corroborators",
            "1",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code != 0
    assert "--rando-min-corroborators" in result.output


def test_dev_flag_respects_top_n_cap(tmp_path: Path) -> None:
    """--dev-top-n trims usages per culture; verify a low cap produces
    a smaller proportions_usage row count."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    # 20 usages, cap to 5 → only 5 rows land per culture.
    import json

    payload = _proportions({f"-key{i}-": 100 - i for i in range(20)})
    (tmp_path / "english_proportions.json").write_text(json.dumps(payload))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--dev",
            "--dev-top-n",
            "5",
            "--db",
            str(db_path),
            "--proportions-dir",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(str(out_path))
    try:
        usage_rows = conn.execute(
            "SELECT COUNT(*) FROM proportions_usage WHERE culture = 'english'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert usage_rows == 5


# ---------- committed seed-runtime.db ----------


def _seed_path() -> Path:
    """Return the on-disk path of the committed seed-runtime.db."""
    return Path(str(resources.files("wyrd.generators.kenning.data").joinpath("seed-runtime.db")))


@pytest.mark.skipif(
    not _seed_path().is_file(),
    reason="seed-runtime.db is not present; regenerate via "
    "`wyrd kenning lexicon export-runtime-db --dev --output "
    "wyrd/generators/kenning/data/seed-runtime.db`",
)
def test_committed_seed_carries_current_schema_version() -> None:
    """bundle_metadata.schema_version on the committed seed must equal
    the runtime's SCHEMA_VERSION constant. When the constant bumps
    (incompatible schema change), the operator must regenerate the
    seed or this test fails — the drift signal."""
    conn = sqlite3.connect(str(_seed_path()))
    try:
        schema_version = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert schema_version == SCHEMA_VERSION, (
        f"committed seed-runtime.db is at schema {schema_version!r} but "
        f"the runtime expects {SCHEMA_VERSION!r}. Regenerate via "
        "`wyrd kenning lexicon export-runtime-db --dev --output "
        "wyrd/generators/kenning/data/seed-runtime.db`."
    )


@pytest.mark.skipif(
    not _seed_path().is_file(),
    reason="seed-runtime.db is not present",
)
def test_committed_seed_carries_dev_built_at() -> None:
    """The committed seed must have been emitted with --dev (so the
    fixed sentinel timestamp is present). Catches operator-error where
    a non-dev emit accidentally clobbered the committed seed."""
    conn = sqlite3.connect(str(_seed_path()))
    try:
        built_at = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'built_at'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert built_at == DEV_BUILT_AT, (
        f"committed seed has built_at={built_at!r}, expected {DEV_BUILT_AT!r}. "
        "Did the operator regenerate without --dev?"
    )


@pytest.mark.skipif(
    not _seed_path().is_file(),
    reason="seed-runtime.db is not present",
)
def test_committed_seed_carries_dev_source_sentinel() -> None:
    """source_lexicon_db on the committed seed must be the dev sentinel
    (not an operator-specific absolute path). Catches an operator who
    re-runs without --dev or who somehow lands their local /home/... path
    in the committed file."""
    conn = sqlite3.connect(str(_seed_path()))
    try:
        source = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'source_lexicon_db'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert source == DEV_SOURCE_LEXICON_SENTINEL, (
        f"committed seed has source_lexicon_db={source!r}, expected "
        f"{DEV_SOURCE_LEXICON_SENTINEL!r}. Regenerate with --dev."
    )


@pytest.mark.skipif(
    not _seed_path().is_file(),
    reason="seed-runtime.db is not present",
)
def test_committed_seed_has_data_in_every_l4_table() -> None:
    """Smoke check: every L4 table the runtime queries should have at
    least one row in the committed seed, so local dev / unit tests have
    something to exercise the generator surface against."""
    conn = sqlite3.connect(str(_seed_path()))
    try:
        for table in (
            "meaning",
            "fantasy_morpheme",
            "canonical_decomposition",
            "proportions_usage",
            "proportions_single_usage",
            "proportions_structure",
            "proportions_tag_marginal",
            "proportions_tag_cooccurrence",
            # wyrd-pfoo: per-Meaning attestation per culture. Empty
            # would mean the splitter dropped the data on the floor;
            # populated is the runtime's culture-bleed defence at the
            # Meaning level.
            "proportions_attested_language",
        ):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count > 0, f"committed seed has no rows in {table}"
    finally:
        conn.close()

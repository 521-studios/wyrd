"""wyrd-04oi: the pickled-matcher cache must be a transparent speed-up — a
loaded trie produces byte-identical decompositions to a freshly-built one, and
every can't-trust-it path returns None so the caller rebuilds (never a crash,
never a wrong result)."""

from __future__ import annotations

import pickle

import pytest

from wyrd.generators.kenning.runtime import matcher_cache
from wyrd.generators.kenning.runtime.matcher_cache import (
    dump_matcher,
    load_matcher,
    matcher_sidecar_path,
    matcher_stamp,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.trie_matcher import (
    MorphemeTrie,
    all_decompositions,
    build_morpheme_trie,
)


def _m(usage: str) -> Meaning:
    # english source so build_morpheme_trie's phonogram filter keeps it.
    return Meaning(usage, [], [], {"modern_english": ["x"]})


def _db() -> dict[str, list[Meaning]]:
    return {
        "bridg": [_m("bridg")],
        "water": [_m("water")],
        "by": [_m("by"), _m("by")],  # two senses on one surface
        "ham": [_m("ham")],
    }


def _surfaces(word: str, trie: MorphemeTrie) -> list[list[str]]:
    """Decompositions reduced to usage-strings (Meaning) / fragments (str), so
    two tries can be compared without relying on Meaning object identity."""
    return [
        [e if isinstance(e, str) else e.usage for e in d] for d in all_decompositions(word, trie)
    ]


def test_roundtrip_produces_identical_decompositions(tmp_path):
    db = _db()
    trie = build_morpheme_trie(db)
    stamp = matcher_stamp(db)
    path = tmp_path / "x.db.matcher.pkl"
    dump_matcher(trie, path, stamp=stamp)

    loaded = load_matcher(path, stamp=stamp)
    assert isinstance(loaded, MorphemeTrie)
    assert loaded is not trie  # genuinely deserialized, not the same object
    for word in ("Bridgwater", "byham", "waterby", "xyz", "ham"):
        assert _surfaces(word, loaded) == _surfaces(word, trie), word
    assert loaded.morpheme_count == trie.morpheme_count


def test_missing_file_returns_none(tmp_path):
    assert load_matcher(tmp_path / "absent.pkl", stamp="anything") is None


def test_stamp_mismatch_returns_none(tmp_path):
    db = _db()
    trie = build_morpheme_trie(db)
    path = tmp_path / "x.matcher.pkl"
    dump_matcher(trie, path, stamp=matcher_stamp(db))
    # A different corpus → different stamp → rebuild.
    other = dict(db)
    other["new"] = [_m("new")]
    assert load_matcher(path, stamp=matcher_stamp(other)) is None


def test_stamp_changes_with_corpus():
    db = _db()
    s1 = matcher_stamp(db)
    # adding a usage changes the stamp
    db2 = dict(db)
    db2["new"] = [_m("new")]
    assert matcher_stamp(db2) != s1
    # changing a sense count (same keys) also changes the stamp
    db3 = dict(db)
    db3["by"] = [_m("by")]  # was 2 senses, now 1
    assert matcher_stamp(db3) != s1
    # identical content → identical stamp (deterministic)
    assert matcher_stamp(_db()) == s1


def test_python_version_mismatch_returns_none(tmp_path, monkeypatch):
    db = _db()
    trie = build_morpheme_trie(db)
    stamp = matcher_stamp(db)
    path = tmp_path / "x.matcher.pkl"
    dump_matcher(trie, path, stamp=stamp)
    # Simulate loading on a different interpreter minor version.
    monkeypatch.setattr(matcher_cache, "_PY", "9.99")
    assert load_matcher(path, stamp=stamp) is None


def test_format_version_mismatch_returns_none(tmp_path, monkeypatch):
    db = _db()
    trie = build_morpheme_trie(db)
    stamp = matcher_stamp(db)
    path = tmp_path / "x.matcher.pkl"
    dump_matcher(trie, path, stamp=stamp)
    monkeypatch.setattr(matcher_cache, "_MATCHER_PICKLE_FORMAT", 999)
    assert load_matcher(path, stamp=stamp) is None


def test_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "corrupt.matcher.pkl"
    path.write_bytes(b"not a pickle \x00\x01\x02")
    assert load_matcher(path, stamp="x") is None


def test_non_dict_payload_returns_none(tmp_path):
    path = tmp_path / "wrong.matcher.pkl"
    with path.open("wb") as fh:
        pickle.dump(["not", "the", "envelope"], fh)
    assert load_matcher(path, stamp="x") is None


def test_dump_is_atomic_no_partial_on_failure(tmp_path):
    # An unpicklable payload must not leave a half-written target file.
    path = tmp_path / "x.matcher.pkl"

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("nope")

    bad = MorphemeTrie()
    bad.forward.terminals.append(Unpicklable())
    with pytest.raises(RuntimeError):
        dump_matcher(bad, path, stamp="x")
    assert not path.exists()  # atomic: target never created
    assert not list(tmp_path.glob("*.matcher.tmp"))  # temp cleaned up


def test_sidecar_path_keeps_db_name():
    p = matcher_sidecar_path("/tmp/wyrd-runtime-abc123.db")
    assert p.name == "wyrd-runtime-abc123.db.matcher.pkl"


# --- cold-load wiring (runtime_matcher_path + _decomposition_matcher) --------


def test_runtime_matcher_path_resolves_sidecar(tmp_path, monkeypatch):
    from wyrd.generators.kenning.runtime import runtime_db

    monkeypatch.setattr(runtime_db, "_resolved_db_path", None)
    assert runtime_db.runtime_matcher_path() is None  # no DB opened yet

    db_file = tmp_path / "y.db"
    db_file.write_bytes(b"")
    monkeypatch.setattr(runtime_db, "_resolved_db_path", db_file)
    assert runtime_db.runtime_matcher_path() is None  # DB but no sidecar

    sidecar = tmp_path / "y.db.matcher.pkl"
    sidecar.write_bytes(b"x")
    assert runtime_db.runtime_matcher_path() == sidecar


def test_decomposition_matcher_uses_pickle_when_present(tmp_path, monkeypatch):
    from wyrd.generators.kenning import _decomposition_matcher
    from wyrd.generators.kenning.runtime import name, runtime_db

    db = _db()
    trie = build_morpheme_trie(db)
    db_file = tmp_path / "z.db"
    db_file.write_bytes(b"")
    dump_matcher(trie, matcher_sidecar_path(db_file), stamp=matcher_stamp(db))
    monkeypatch.setattr(runtime_db, "_resolved_db_path", db_file)

    # The pickle matches → must NOT rebuild.
    def _boom(_db):
        raise AssertionError("_trie_for should not be called when the pickle loads")

    monkeypatch.setattr(name, "_trie_for", _boom)
    got = _decomposition_matcher(db)
    assert isinstance(got, MorphemeTrie)
    assert _surfaces("Bridgwater", got) == _surfaces("Bridgwater", trie)


def test_decomposition_matcher_rebuilds_without_sidecar(tmp_path, monkeypatch):
    from wyrd.generators.kenning import _decomposition_matcher
    from wyrd.generators.kenning.runtime import name, runtime_db

    db = _db()
    monkeypatch.setattr(runtime_db, "_resolved_db_path", None)  # no sidecar path
    sentinel = build_morpheme_trie(db)
    monkeypatch.setattr(name, "_trie_for", lambda d: sentinel)
    assert _decomposition_matcher(db) is sentinel  # fell back to build


def test_decomposition_matcher_rebuilds_on_stamp_mismatch(tmp_path, monkeypatch):
    from wyrd.generators.kenning import _decomposition_matcher
    from wyrd.generators.kenning.runtime import name, runtime_db

    db = _db()
    db_file = tmp_path / "w.db"
    db_file.write_bytes(b"")
    # dump a pickle stamped for a DIFFERENT corpus
    other = dict(db)
    other["extra"] = [_m("extra")]
    dump_matcher(
        build_morpheme_trie(other), matcher_sidecar_path(db_file), stamp=matcher_stamp(other)
    )
    monkeypatch.setattr(runtime_db, "_resolved_db_path", db_file)
    sentinel = build_morpheme_trie(db)
    monkeypatch.setattr(name, "_trie_for", lambda d: sentinel)
    assert _decomposition_matcher(db) is sentinel  # stamp mismatch → rebuild


def test_build_sidecar_against_seed_db_roundtrips(tmp_path, monkeypatch):
    """End-to-end: build the matcher from the real bundled seed corpus via the
    export helper, then load it back and confirm it decomposes a name the same
    as a freshly-built trie."""
    from wyrd.generators.kenning import _load_meanings
    from wyrd.generators.kenning.cli.lexicon.export_runtime_db import _build_matcher_sidecar
    from wyrd.generators.kenning.runtime import runtime_db
    from wyrd.generators.kenning.runtime.runtime_db import _bundled_seed_path

    seed = _bundled_seed_path()
    if not seed.exists():
        pytest.skip("bundled seed runtime DB not present")
    # Build the sidecar from the seed corpus (helper points WYRD_RUNTIME_DB at
    # it, loads, dumps, and resets caches in finally).
    dest = tmp_path / "seed.db"
    dest.write_bytes(seed.read_bytes())
    sidecar = _build_matcher_sidecar(dest)
    assert sidecar.exists()

    # Now load against the same DB and confirm a real decomposition matches a
    # freshly-built trie over the seed corpus.
    monkeypatch.setenv("WYRD_RUNTIME_DB", str(dest))
    runtime_db.reset_runtime_db_cache()
    _load_meanings.cache_clear()
    try:
        meaning_db, _ = _load_meanings()
        loaded = load_matcher(sidecar, stamp=matcher_stamp(meaning_db))
        assert isinstance(loaded, MorphemeTrie)
        fresh = build_morpheme_trie(meaning_db)
        for word in ("Bridgwater", "Whitby", "Ashford"):
            assert _surfaces(word, loaded) == _surfaces(word, fresh), word
    finally:
        monkeypatch.delenv("WYRD_RUNTIME_DB", raising=False)
        runtime_db.reset_runtime_db_cache()
        _load_meanings.cache_clear()

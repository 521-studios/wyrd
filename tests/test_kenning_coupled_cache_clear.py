"""Pin the wyrd-h3ls cache-coupling: ``_load_meanings.cache_clear()``
must also drop the ``_load_culture`` cache.

Why this matters: ``_load_culture`` independently lru-caches a
``NameGenerator`` parameterised on the ``meaning_db`` returned by
``_load_meanings``. A test that mutates the bundle and clears only
``_load_meanings`` would leave ``_load_culture``'s cached generator
holding a reference to the old ``meaning_db``. Pre-wyrd-h3ls this was
a latent footgun — no test triggered it, but a future mutation-test
author would have hit it without warning. Coupling the clears
eliminates the footgun by making the obvious call do the right thing.
"""

from __future__ import annotations

from wyrd.generators.kenning import (
    _coupled_cache_clear,
    _load_culture,
    _load_joiners,
    _load_meanings,
)
from wyrd.generators.kenning.era import rewind


def test_load_meanings_cache_clear_is_wired_to_coupled_helper() -> None:
    """Pin the wiring itself — ``_load_meanings.cache_clear`` must
    BE ``_coupled_cache_clear``. The other tests here exercise the
    coupling's effect (per-culture cache empty, identity preserved),
    but they call ``_load_culture.cache_clear()`` explicitly at
    setup, so a regression that drops the line-273 monkey-patch
    would still pass them. This identity check pins the patch
    survival itself."""
    assert _load_meanings.cache_clear is _coupled_cache_clear, (
        "_load_meanings.cache_clear must be the coupled-clear helper; "
        "the monkey-patch at the bottom of __init__.py looks dropped"
    )


def test_load_meanings_cache_clear_also_clears_load_culture() -> None:
    """The standard test-side pattern is
    ``_load_meanings.cache_clear()``. After wyrd-h3ls that call must
    also drop the per-culture cache. Pin via the lru_cache stats:
    after a culture load + meaning_db clear, the per-culture cache
    should be empty (currsize == 0)."""
    _load_meanings.cache_clear()
    _load_culture.cache_clear()

    # Prime both caches.
    _load_meanings()
    _load_culture("english")
    assert _load_culture.cache_info().currsize == 1, "english culture should be cached after load"

    # The coupled clear: caller hits only the meanings cache.
    _load_meanings.cache_clear()

    assert _load_culture.cache_info().currsize == 0, (
        "_load_meanings.cache_clear() should also drop _load_culture's cache; "
        "pre-wyrd-h3ls this assertion failed because the two caches were "
        "independent and the per-culture generator held a reference to the "
        "old meaning_db"
    )


def test_culture_load_returns_a_consistent_meaning_db_after_clear() -> None:
    """After clearing _load_meanings's cache, the next call to
    _load_culture must return a NameGenerator built over the
    freshly-loaded meaning_db, not a cached one. The test-witness
    is identity: word_db and culture's NameGenerator.meaning_gen
    must agree on the meaning_db they reference.

    Without the coupled clear, _load_culture's cache would serve a
    NameGenerator wrapped over the OLD meaning_db while
    _load_meanings serves the NEW one — the generator's internals
    would be silently inconsistent with the freshly-loaded bundle."""
    _load_meanings.cache_clear()
    word_db_before, _ = _load_meanings()
    name_gen_before, _ = _load_culture("english")

    # Clear and re-load.
    _load_meanings.cache_clear()
    word_db_after, _ = _load_meanings()
    name_gen_after, _ = _load_culture("english")

    # Identity check: the word_db reference _load_culture's
    # NameGenerator wraps must be the SAME object _load_meanings
    # returns. Pre-coupled-clear, the per-culture cache hit would
    # produce a NameGenerator referencing the stale word_db.
    assert id(name_gen_before.meaning_gen.meaning_db) == id(word_db_before)
    assert id(name_gen_after.meaning_gen.meaning_db) == id(word_db_after)
    # And the two clears produced distinct meaning_db objects (no
    # cache leak across the explicit clear).
    assert id(word_db_before) != id(word_db_after)


def test_load_meanings_cache_clear_also_clears_load_joiners() -> None:
    """wyrd-0l2g: ``_load_meanings.cache_clear()`` must ALSO drop the
    ``_load_joiners`` cache. The two read the same bundle file; if
    one's cache stays warm while the other's drops, a test that
    mutates the bundle gets inconsistent views."""
    _load_meanings.cache_clear()
    _load_joiners.cache_clear()

    # Prime both caches.
    _load_meanings()
    _load_joiners()
    assert _load_joiners.cache_info().currsize == 1, (
        "joiners should be cached after _load_joiners() call"
    )

    # The coupled clear should drop joiners too.
    _load_meanings.cache_clear()
    assert _load_joiners.cache_info().currsize == 0, (
        "_load_meanings.cache_clear() should also drop _load_joiners' cache"
    )


def test_coupled_clear_also_clears_rewind_stripped_index() -> None:
    """The rewind anchor resolver's stripped-usage index is the SAME
    id(meaning_db)-keyed bundle cache as proportions._SURFACE_INDEX_CACHE, with
    the same stale-on-reload hazard: after a bundle reload CPython can reuse the
    freed meaning_db's id() for a new bundle's db, and the cache would then serve
    the OLD bundle's index. The coupled clear must drop it in lockstep with
    _load_meanings (it previously did not — the cache was absent from the
    coupling)."""
    rewind._STRIPPED_INDEX_CACHE.clear()
    rewind._get_stripped_index({"-ham": ["m"]})
    assert rewind._STRIPPED_INDEX_CACHE  # populated
    _load_meanings.cache_clear()  # the standard coupled entry point
    assert rewind._STRIPPED_INDEX_CACHE == {}, (
        "_load_meanings.cache_clear() must also drop the rewind stripped-index "
        "cache; it is an id(meaning_db)-keyed bundle cache and goes stale on reload"
    )


def test_get_stripped_index_rebuilds_on_id_recycle() -> None:
    """Identity-check guard: if id(meaning_db) was recycled (a freed db's id now
    belongs to a NEW db), the cache must NOT return the stale prior entry — the
    stored (meaning_db, index) back-reference forces a rebuild. Simulated by
    planting a different-db entry under the new db's id key."""
    rewind._STRIPPED_INDEX_CACHE.clear()
    db_new = {"-ton": ["m1"]}
    rewind._STRIPPED_INDEX_CACHE[id(db_new)] = ({"-stale": []}, {"stale": ["WRONG"]})
    assert rewind._get_stripped_index(db_new) == {"ton": ["m1"]}  # rebuilt, not stale

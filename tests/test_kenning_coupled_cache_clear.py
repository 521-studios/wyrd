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

from wyrd.generators.kenning import _load_culture, _load_meanings


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

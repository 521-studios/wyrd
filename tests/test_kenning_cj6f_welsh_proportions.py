"""Regression tests for wyrd-cj6f: welsh proportions generation crashed
with ``KeyError: ('bare', 'name', 'single')`` when a structure slot
referenced a (position, tag, count) bucket that had no registered
Generator.

Root cause: ``MeaningGenerator.load_parts`` only registers a Generator
for a bucket when a usage exists in BOTH the proportions table and the
meaning_db. Structures, however, are kept in full — so a structure slot
can reference a bucket whose populating usage(s) are absent (bundle /
proportions drift) or trimmed (the ``--dev`` seed keeps top-N usages but
all structures). ``MeaningGenerator.select`` then indexed
``self.generators[key]`` directly and raised KeyError mid-generation.

Fix: ``select`` returns None for an unregistered bucket (mirroring
``load_parts``' wyrd-van9 skip-don't-crash policy and ``bucket_keys``'
existing ``.get()``). None is already a valid per-slot result that the
callers handle by leaving the element unfilled.

Note: in the committed ``--dev`` seed welsh has one orphaned single-slot
structure whose bucket was trimmed, so the rare seeds that pick it now
render an empty name instead of crashing. That is a dev-seed-only
artifact — the full production bundle populates the bucket, so no empty
names there — and a non-crash is the contract this ticket fixes.

Tests use the committed bundle via ``_load_meanings`` / a real Kenning
(a fixture, not the live ~/.wyrd lexicon DB).
"""

from __future__ import annotations

from wyrd.generators.kenning import _load_meanings
from wyrd.generators.kenning.generators.kenning import Kenning
from wyrd.generators.kenning.runtime.proportions import MeaningGenerator
from wyrd.seed import rng_for


def test_meaning_generator_select_missing_bucket_returns_none(caplog):
    """The fix: a bucket key absent from ``self.generators`` returns None
    rather than raising KeyError (the seed-806 crash site), warns exactly
    once per key (D24 observability), and leaves the present-key path
    working (the no-regression half of the change)."""
    import logging

    meaning_db, tag_db = _load_meanings()
    known = next(iter(meaning_db))
    mg = MeaningGenerator(meaning_db, tag_db, {known: 100})

    # Present key still routes through and returns a real usage.
    present_key = next(iter(mg.generators))
    assert mg.select(rng_for(0), present_key) is not None

    # Missing key returns None (was KeyError) and warns once, not per call.
    bogus_key = ("bare", "__no_such_tag__", "single")
    assert bogus_key not in mg.generators
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.runtime.proportions"):
        assert mg.select(rng_for(0), bogus_key) is None
        assert mg.select(rng_for(1), bogus_key) is None  # repeat → no second warning
    cj6f_warnings = [r for r in caplog.records if "wyrd-cj6f" in r.getMessage()]
    assert len(cj6f_warnings) == 1, "missing-bucket drift should warn exactly once per key"


def test_welsh_proportions_seed_806_does_not_crash():
    """Original reproducer: welsh proportions at seed 806 KeyError'd.
    Post-fix it returns a valid GenerationResult without raising. (The
    rendered name may be empty for this dev-seed orphan structure — see
    module docstring — but it must not crash and must stay a str.)"""
    k = Kenning()
    result = k.generate({"culture": "welsh", "scoring_mode": "proportions"}, seed=806)
    assert hasattr(result, "result")
    assert isinstance(result.result, str)


def test_welsh_proportions_no_crash_across_seed_range():
    """Robustness guard spanning the known crashing seed (806): welsh
    proportions generation must not raise for any seed in the range, so a
    future seed-mapping shift can't silently reintroduce the crash."""
    k = Kenning()
    for seed in range(750, 870):
        k.generate({"culture": "welsh", "scoring_mode": "proportions"}, seed=seed)

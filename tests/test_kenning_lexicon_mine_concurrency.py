"""Tests for the parallelized mining loop in `_mine_entries` (wyrd-l0r).

Covers the serial path's bit-stability (concurrency=1 == pre-PR behavior)
and the parallel path's correctness invariants:

- input-order preservation in `accepted_entries`
- accept/decline/reject counts match across serial and parallel runs
- per-entry validation (D3) survives the fan-out (extract_one is invoked
  exactly once per entry, never batched)
- exceptions on individual entries don't kill sibling tasks
"""

from __future__ import annotations

import threading
import time

from wyrd.generators.kenning.cli import _mine_entries
from wyrd.generators.kenning.llm_extractor import LLMResult, ValidationFailure


class _FakeEntry:
    """Stand-in for ParsedEntry used as the extract_one return payload.
    Only the .elements truthiness is consulted by `_mine_entries` so a
    minimal stub suffices."""

    def __init__(self, label: str, has_elements: bool = True) -> None:
        self.label = label
        self.elements = ["x"] if has_elements else []


class _FakeParsed:
    """Stand-in for the parser output: just the three attributes
    `_mine_entries` reads."""

    def __init__(self, toponym: str, body_text: str = "body", suffix: str | None = None) -> None:
        self.toponym = toponym
        self.body_text = body_text
        self.section_suffix = suffix


def _make_extractor(plan):
    """Build an extract_one stub that returns a pre-baked LLMResult per
    toponym based on ``plan`` (dict mapping toponym → 'accept' | 'decline'
    | ('reject', failure_reason))."""

    def fake(client, toponym, body, suffix_hint):  # noqa: ARG001
        verdict = plan[toponym]
        if verdict == "accept":
            return LLMResult(
                entry=_FakeEntry(toponym),
                raw_response={},
                failures=[],
                accepted=True,
            )
        if verdict == "decline":
            return LLMResult(
                entry=_FakeEntry(toponym, has_elements=False),
                raw_response={},
                failures=[],
                accepted=True,
            )
        # ('reject', reason)
        _, reason = verdict
        return LLMResult(
            entry=None,
            raw_response={},
            failures=[ValidationFailure(reason)],
            accepted=False,
        )

    return fake


def test_mine_entries_serial_matches_parallel_on_same_inputs():
    """concurrency=1 and concurrency=8 must produce the same accepted
    entries (in the same input order), declined count, rejected count,
    and by_failure breakdown when run against the same plan."""
    parsed = [_FakeParsed(f"t{i:02d}") for i in range(20)]
    plan = {p.toponym: "accept" for p in parsed}
    # Inject a few non-accepted verdicts at varied positions.
    plan["t03"] = "decline"
    plan["t07"] = ("reject", "form_not_in_body")
    plan["t14"] = ("reject", "bad_language")
    plan["t17"] = ("reject", "form_not_in_body")
    extract_one = _make_extractor(plan)

    serial = _mine_entries(
        parsed=parsed,
        client=None,
        extract_one=extract_one,
        concurrency=1,
        progress_start=time.time(),
    )
    parallel = _mine_entries(
        parsed=parsed,
        client=None,
        extract_one=extract_one,
        concurrency=8,
        progress_start=time.time(),
    )

    s_accepted, s_declined, s_rejected, s_failures = serial
    p_accepted, p_declined, p_rejected, p_failures = parallel

    # Accepted entries are pinned by input order on both paths so the
    # downstream ingest sees the same insertion sequence regardless of
    # task-completion order.
    assert [e.label for e in s_accepted] == [e.label for e in p_accepted]
    assert [e.label for e in s_accepted] == [
        f"t{i:02d}" for i in range(20) if i not in (3, 7, 14, 17)
    ]
    assert s_declined == p_declined == 1
    assert s_rejected == p_rejected == 3
    assert s_failures == p_failures == {"form_not_in_body": 2, "bad_language": 1}


def test_mine_entries_parallel_runs_extract_one_exactly_once_per_entry():
    """D3 invariant: per-entry validation must not be batched. Pin via a
    counter — every parsed entry should produce exactly one extract_one
    invocation regardless of concurrency. A regression that, say,
    submitted the same future twice would surface as a count mismatch."""
    parsed = [_FakeParsed(f"t{i:02d}") for i in range(50)]
    call_count: dict[str, int] = {}
    lock = threading.Lock()

    def extract_one(client, toponym, body, suffix_hint):  # noqa: ARG001
        with lock:
            call_count[toponym] = call_count.get(toponym, 0) + 1
        return LLMResult(
            entry=_FakeEntry(toponym),
            raw_response={},
            failures=[],
            accepted=True,
        )

    _mine_entries(
        parsed=parsed,
        client=None,
        extract_one=extract_one,
        concurrency=8,
        progress_start=time.time(),
    )

    assert set(call_count) == {p.toponym for p in parsed}
    assert all(n == 1 for n in call_count.values())


def test_mine_entries_parallel_actually_overlaps_calls():
    """Sanity check that ThreadPoolExecutor is dispatching concurrently
    rather than silently serializing. Each fake call sleeps briefly; with
    concurrency=8 and 16 entries × 50ms each the wall-clock should be
    near 100ms, well under the 800ms a serial run would take."""
    parsed = [_FakeParsed(f"t{i:02d}") for i in range(16)]

    def extract_one(client, toponym, body, suffix_hint):  # noqa: ARG001
        time.sleep(0.05)
        return LLMResult(
            entry=_FakeEntry(toponym),
            raw_response={},
            failures=[],
            accepted=True,
        )

    start = time.time()
    _mine_entries(
        parsed=parsed,
        client=None,
        extract_one=extract_one,
        concurrency=8,
        progress_start=start,
    )
    elapsed = time.time() - start
    # Wide upper bound to absorb scheduler jitter on slow CI hosts.
    assert elapsed < 0.4, f"parallel run took {elapsed:.3f}s — likely serializing"


def test_mine_entries_concurrency_one_is_bit_stable_with_serial_loop():
    """The concurrency=1 path must produce IDENTICAL output to the
    pre-PR serial loop. Strong correctness pin against a refactor that
    inadvertently routes the serial path through the executor (which
    would change ordering / scheduling semantics).

    Equivalent of the cohesion / era-filter bit-stability tests
    elsewhere in the suite."""
    parsed = [_FakeParsed(f"t{i:02d}") for i in range(10)]
    plan = {p.toponym: "accept" for p in parsed}
    plan["t02"] = ("reject", "form_not_in_body")
    plan["t05"] = "decline"

    accepted, declined, rejected, failures = _mine_entries(
        parsed=parsed,
        client=None,
        extract_one=_make_extractor(plan),
        concurrency=1,
        progress_start=time.time(),
    )

    assert [e.label for e in accepted] == [
        "t00",
        "t01",
        "t03",
        "t04",
        "t06",
        "t07",
        "t08",
        "t09",
    ]
    assert declined == 1
    assert rejected == 1
    assert failures == {"form_not_in_body": 1}


def test_mine_entries_parallel_propagates_extract_one_exceptions():
    """If extract_one raises (e.g. an unhandled HTTP error), the
    exception should propagate up rather than being silently dropped.
    Better to fail loudly mid-mining than to ingest a partial run that
    looks complete."""
    import pytest

    parsed = [_FakeParsed(f"t{i:02d}") for i in range(10)]

    def extract_one(client, toponym, body, suffix_hint):  # noqa: ARG001
        if toponym == "t05":
            raise RuntimeError("simulated transport blow-up")
        return LLMResult(
            entry=_FakeEntry(toponym),
            raw_response={},
            failures=[],
            accepted=True,
        )

    with pytest.raises(RuntimeError, match="simulated transport blow-up"):
        _mine_entries(
            parsed=parsed,
            client=None,
            extract_one=extract_one,
            concurrency=4,
            progress_start=time.time(),
        )


def test_mine_entries_empty_parsed_returns_empty_results():
    """Edge case: no parsed entries → empty accepted list, zero
    counts, empty failure breakdown. Guard so the helper doesn't
    divide-by-zero or trip the executor's empty-iterable path."""
    accepted, declined, rejected, failures = _mine_entries(
        parsed=[],
        client=None,
        extract_one=_make_extractor({}),
        concurrency=4,
        progress_start=time.time(),
    )
    assert accepted == []
    assert declined == 0
    assert rejected == 0
    assert failures == {}

"""Tests for ``mine-skeat``'s ``_mine_entries`` extract loop (wyrd-l0r).

Pins the wyrd-8uvi-extracted helpers (``_MineState`` / ``_process`` /
``_emit_progress`` / ``_mine_entries_parallel``): per-result tallying
(accepted / declined / rejected + failure reasons), INPUT-order slotting of
accepted entries (deterministic regardless of completion order), and the
every-``log_every`` progress echo. ``extract_one`` is stubbed — no LLM.
"""

from __future__ import annotations

from types import SimpleNamespace

from wyrd.generators.kenning.cli.lexicon.mine_skeat import _mine_entries


def _parsed(n: int) -> list:
    """n entries cycling accepted-with-elements (A) / declined (D) / rejected (R)
    by section_suffix, so the stub extractor can branch deterministically."""
    return [
        SimpleNamespace(
            toponym=f"T{i}{'ADR'[i % 3]}", body_text=f"b{i}", section_suffix="ADR"[i % 3]
        )
        for i in range(n)
    ]


def _extract_one(client, *, toponym, body, suffix_hint):
    if suffix_hint == "A":
        return SimpleNamespace(
            accepted=True, entry=SimpleNamespace(elements=[1], name=toponym), failures=[]
        )
    if suffix_hint == "D":  # accepted but no elements → declined
        return SimpleNamespace(
            accepted=True, entry=SimpleNamespace(elements=[], name=toponym), failures=[]
        )
    return SimpleNamespace(
        accepted=False, entry=None, failures=[SimpleNamespace(reason="bad_form")]
    )


def test_mine_entries_serial_tallies_and_input_order() -> None:
    parsed = _parsed(9)  # 3×(A, D, R)
    accepted, declined, rejected, by_failure = _mine_entries(
        parsed=parsed, client=None, extract_one=_extract_one, concurrency=1, progress_start=0.0
    )
    assert [e.name for e in accepted] == ["T0A", "T3A", "T6A"]  # only A, in input order
    assert declined == 3
    assert rejected == 3
    assert by_failure == {"bad_form": 3}


def test_mine_entries_parallel_matches_serial_result() -> None:
    parsed = _parsed(25)
    serial = _mine_entries(
        parsed=parsed, client=None, extract_one=_extract_one, concurrency=1, progress_start=0.0
    )
    parallel = _mine_entries(
        parsed=parsed, client=None, extract_one=_extract_one, concurrency=4, progress_start=0.0
    )
    # Accepted entries come back in INPUT order on BOTH paths (slotted by index),
    # so the return value is identical regardless of completion order.
    assert [e.name for e in serial[0]] == [e.name for e in parallel[0]]
    assert serial[1:] == parallel[1:]  # declined, rejected, by_failure


def test_mine_entries_emits_progress_every_log_every(capsys) -> None:
    parsed = _parsed(25)
    _mine_entries(
        parsed=parsed,
        client=None,
        extract_one=_extract_one,
        concurrency=1,
        progress_start=0.0,
        log_every=10,
    )
    err = capsys.readouterr().err
    # Progress lines at completed=10 and 20 (not at 25 — only multiples of 10).
    assert "[10/25]" in err
    assert "[20/25]" in err
    assert "[25/25]" not in err

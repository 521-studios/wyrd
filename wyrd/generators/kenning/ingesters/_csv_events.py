"""Shared CSV → JSONL event builder for the per-source toponym ingesters.

``build_events`` was duplicated byte-for-byte across ``os_open_names``,
``speed_1611``, ``hundred_rolls`` and ``hearth_tax`` — each copy differed only in
the two per-source pieces it closed over: the source-metadata row (``_SOURCE_ROW``)
and the row mapper (``_normalize_row``). Those are injected here so the
toponym-dedupe + source-row + attestation-emit orchestration lives in one place
and can't drift across sources.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.jsonl.log import write_jsonl


def build_events(
    csv_rows: Iterable[dict[str, str]],
    *,
    source_row: dict[str, Any],
    normalize_row: Callable[[dict[str, str]], tuple[dict[str, Any], dict[str, Any]] | None],
    include_source_row: bool = True,
) -> list[dict[str, Any]]:
    """Pure: CSV rows → JSONL events. Source row first (optional); toponym refs
    dedupe; attestations don't (each source entry is independently auditable).

    ``source_row`` is the source's metadata event and ``normalize_row`` maps one
    CSV row to a ``(toponym_event, attestation_event)`` pair (or ``None`` to skip
    it) — the only two things that vary per source.
    """
    out: list[dict[str, Any]] = []
    if include_source_row:
        out.append(dict(source_row))
    seen_toponym_refs: set[str] = set()
    for row in csv_rows:
        pair = normalize_row(row)
        if pair is None:
            continue
        toponym_event, attestation_event = pair
        ref = toponym_event["ref"]
        if ref not in seen_toponym_refs:
            out.append(toponym_event)
            seen_toponym_refs.add(ref)
        out.append(attestation_event)
    return out


def ingest(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    iter_csv_rows: Callable[[Path], Iterable[dict[str, str]]],
    build_events: Callable[[Iterable[dict[str, str]]], list[dict[str, Any]]],
    apply: bool = False,
) -> dict[str, int]:
    """Read CSV → write JSONL events (dry-run by default), counting rows + events.

    Shared by the per-source CSV ingesters — ``iter_csv_rows`` (the source's CSV
    reader) and ``build_events`` (its event builder) are injected, the only two
    pieces that vary per source.
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    counts = {
        "csv_rows_scanned": 0,
        "toponym_events": 0,
        "attestation_events": 0,
    }

    def _counting_rows() -> Iterable[dict[str, str]]:
        for row in iter_csv_rows(csv_path):
            counts["csv_rows_scanned"] += 1
            yield row

    events = build_events(_counting_rows())
    for event in events:
        if event["_type"] == "toponym":
            counts["toponym_events"] += 1
        elif event["_type"] == "attestation":
            counts["attestation_events"] += 1

    if apply:
        write_jsonl(out_path, events)
        counts["written"] = 1
    return counts

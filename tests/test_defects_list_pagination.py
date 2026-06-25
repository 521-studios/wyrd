"""Pagination/ordering contract for ``defects.list_defects`` (no moto needed).

The DynamoDB-backed tests in ``test_defects.py`` use moto; these exercise the
pure pagination *loop* via an injected fake table, so they run without the AWS
mock and pin the newest-first contract across multiple result pages.

Regression target: ``list_defects(status='all')`` scans in ARBITRARY order, so it
must exhaust the table BEFORE the newest-first sort — stopping early at ``limit``
(as the GSI-query path legitimately does) would let a newer row on an unscanned
page be missed. The query path (``status != 'all'``) returns rows newest-first, so
it still early-exits once ``limit`` matches are held.
"""

from __future__ import annotations

from typing import Any

from wyrd import defects


def _item(
    id_: str, created_at: str, *, status: str = "new", generator: str = "g"
) -> dict[str, Any]:
    return {
        "id": id_,
        "status": status,
        "created_at": created_at,
        "generator": generator,
        "name": id_,
        "reason": "r",
    }


class _FakeTable:
    """Serves canned pages keyed by a synthetic ExclusiveStartKey; counts calls."""

    def __init__(self, pages: list[list[dict]]):
        self._pages = pages
        self.calls = 0

    def _serve(self, kwargs: dict) -> dict:
        self.calls += 1
        sk = kwargs.get("ExclusiveStartKey")
        idx = 0 if sk is None else sk["page"]
        resp: dict[str, Any] = {"Items": self._pages[idx]}
        if idx + 1 < len(self._pages):
            resp["LastEvaluatedKey"] = {"page": idx + 1}
        return resp

    def scan(self, **kwargs):
        return self._serve(kwargs)

    def query(self, **kwargs):
        return self._serve(kwargs)


def test_list_defects_all_exhausts_pages_before_newest_first_sort(monkeypatch):
    """status='all' must page through the whole (arbitrary-order) scan before
    sorting, or a newer row on a later page is dropped. Here the NEWEST row is on
    page 1 while page 0 (scanned first) holds an older row; limit=1 must still
    return the page-1 newest. Fails on the pre-fix early-exit (only page 0 seen)."""
    old = _item("old", "2026-01-01T00:00:00+00:00")
    new = _item("new", "2026-06-01T00:00:00+00:00")
    fake = _FakeTable([[old], [new]])
    monkeypatch.setattr(defects, "_defects_table", lambda **kw: fake)

    result = defects.list_defects(status="all", limit=1, env="staging")

    assert [r["id"] for r in result] == ["new"]
    assert fake.calls == 2  # both pages scanned


def test_list_defects_query_still_early_exits(monkeypatch):
    """The GSI query path returns rows newest-first, so once `limit` matches are
    held it must NOT keep paging — the early-exit is preserved by the fix."""
    newest = _item("n", "2026-06-01T00:00:00+00:00")
    older = _item("o", "2026-01-01T00:00:00+00:00")
    fake = _FakeTable([[newest], [older]])  # page 0 newest (GSI order), page 1 older
    monkeypatch.setattr(defects, "_defects_table", lambda **kw: fake)

    result = defects.list_defects(status="new", limit=1, env="staging")

    assert [r["id"] for r in result] == ["n"]
    assert fake.calls == 1  # did not page into the older page


def test_list_defects_query_pages_until_limit_matches_under_generator_filter(monkeypatch):
    """The early-exit counts POST-filter matches: when page 0's rows are all
    excluded by the generator filter, the GSI path must page on to page 1 to reach
    `limit`, rather than stopping at the (zero-match) first page."""
    p0 = _item("a", "2026-05-01T00:00:00+00:00", generator="other")  # filtered out
    p1 = _item("b", "2026-04-01T00:00:00+00:00", generator="kenning")  # matches
    fake = _FakeTable([[p0], [p1]])
    monkeypatch.setattr(defects, "_defects_table", lambda **kw: fake)

    result = defects.list_defects(status="new", generator="kenning", limit=1, env="staging")

    assert [r["id"] for r in result] == ["b"]
    assert fake.calls == 2  # paged past the all-filtered page 0 to collect the match

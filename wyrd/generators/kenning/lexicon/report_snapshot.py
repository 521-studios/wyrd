"""Port file-based dashboard reports into the queryable ``report_snapshot``
/ ``report_metric`` tables (wyrd-d24).

A "report" is a directory under ``data/mining/reports/<label>/`` holding the
dashboard artifacts (``stats.txt``, ``enrichment_status.txt``,
``era_coverage.txt``, ``rando_port_readiness.txt``, ``empirical_priors.json``,
``drift_*.json``). :func:`parse_report_dir` turns the structured ones into a
flat list of ``(category, metric, value_num, value_text)`` rows so before/after
comparisons become a SQL JOIN instead of a ``diff`` over text files.

Parsing is tolerant by construction: each file is parsed independently and a
missing or unparseable file simply contributes no rows (the dirs differ in
which artifacts they carry). Numbers may contain thousands separators
(``22,590``) and percent signs; both are normalized into ``value_num`` while
the verbatim token is preserved in ``value_text``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metric:
    category: str
    metric: str
    value_num: float | None
    value_text: str | None


def _to_num(token: str) -> float | None:
    """Parse a numeric token tolerant of ``,`` separators and a trailing
    ``%``. Returns None when the token isn't numeric."""
    cleaned = token.strip().rstrip("%").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_stats(text: str) -> list[Metric]:
    """``stats.txt`` lines: ``<table_name><spaces><count>``."""
    out: list[Metric] = []
    for line in text.splitlines():
        m = re.match(r"^\s*([a-zA-Z_][\w]*)\s+([\d,]+)\s*$", line)
        if m:
            out.append(Metric("stats", m.group(1), _to_num(m.group(2)), m.group(2).strip()))
    return out


def _parse_enrichment(text: str) -> list[Metric]:
    """``enrichment_status.txt``: ``- Total etymons: N`` and
    ``- `col`: N / M ( P%)`` lines."""
    out: list[Metric] = []
    total = re.search(r"Total etymons:\s*([\d,]+)", text)
    if total:
        out.append(Metric("enrichment", "total_etymons", _to_num(total.group(1)), total.group(1)))
    # - `cognate_id`:   657500 / 2370418 ( 27.7%)
    for col, count, pct in re.findall(
        r"^\s*-\s*`([\w]+)`:\s*([\d,]+)\s*/\s*[\d,]+\s*\(\s*([\d.]+)%\)",
        text,
        re.MULTILINE,
    ):
        out.append(Metric("enrichment", f"{col}_count", _to_num(count), count))
        out.append(Metric("enrichment", f"{col}_pct", _to_num(pct), f"{pct}%"))
    return out


def _parse_era(text: str) -> list[Metric]:
    """``era_coverage.txt`` bulleted totals."""
    out: list[Metric] = []
    patterns = {
        "total_toponyms": r"Total toponyms:\s*([\d,]+)",
        "toponyms_with_attestation": r"with ≥1 attestation:\s*([\d,]+)",
    }
    for metric, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out.append(Metric("era", metric, _to_num(m.group(1)), m.group(1)))
    return out


def _parse_rando(text: str) -> list[Metric]:
    """``rando_port_readiness.txt`` markdown gate table — per-language
    coverage (the ``Coverage`` column) + the ``C1`` verdict, plus the
    overall PASS/FAIL line."""
    out: list[Metric] = []
    overall = re.search(r"\*\*Overall:\s*(PASS|FAIL)", text)
    if overall:
        out.append(
            Metric("rando", "overall", 1.0 if overall.group(1) == "PASS" else 0.0, overall.group(1))
        )
    # Data rows look like: | old_english | 1396 | 555 | ... | 74.4% | ✗ | ✓ | ✓ |
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0].lower() in ("language", "---") or "---" in cells[0]:
            continue
        lang = cells[0]
        if not re.match(r"^[a-z_]+$", lang):
            continue
        # The gate table carries TWO percent columns in order:
        # Scholar-only then Coverage. Capture both — coverage is the
        # C1-gated headline; scholar-only shows how much rests on the
        # empirical (vs scholar) pipeline.
        pcts = [c for c in cells if c.endswith("%")]
        if len(pcts) >= 1:
            out.append(Metric("rando", f"{lang}.scholar_only_pct", _to_num(pcts[0]), pcts[0]))
        if len(pcts) >= 2:
            out.append(Metric("rando", f"{lang}.coverage", _to_num(pcts[-1]), pcts[-1]))
        # C1 is the first ✓/✗ verdict cell after the numeric columns.
        c1 = next((c for c in cells if c in ("✓", "✗")), None)
        if c1 is not None:
            out.append(Metric("rando", f"{lang}.c1", 1.0 if c1 == "✓" else 0.0, c1))
    return out


def _parse_priors(text: str) -> list[Metric]:
    """``empirical_priors.json``: native / loan cell counts."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    out: list[Metric] = []
    for key in ("native", "loan"):
        if isinstance(data.get(key), list):
            n = len(data[key])
            out.append(Metric("priors", f"{key}_cells", float(n), str(n)))
    return out


def _parse_drift(text: str, culture: str) -> list[Metric]:
    """``drift_<culture>.json``: per-culture decomposition rates."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    out: list[Metric] = []
    for key in ("decomposition_rate_a", "decomposition_rate_b"):
        if isinstance(data.get(key), (int, float)):
            out.append(Metric("drift", f"{culture}.{key}", float(data[key]), str(data[key])))
    return out


def parse_report_dir(report_dir: Path) -> list[Metric]:
    """Parse every recognized artifact in ``report_dir`` into Metric rows.

    Returns a deduplicated, deterministically-sorted list. Unknown /
    missing / unparseable files contribute nothing."""
    report_dir = Path(report_dir)
    metrics: list[Metric] = []

    simple = {
        "stats.txt": _parse_stats,
        "enrichment_status.txt": _parse_enrichment,
        "era_coverage.txt": _parse_era,
        "rando_port_readiness.txt": _parse_rando,
        "empirical_priors.json": _parse_priors,
    }
    for fname, parser in simple.items():
        path = report_dir / fname
        if path.is_file():
            metrics.extend(parser(path.read_text(encoding="utf-8")))

    for drift_path in sorted(report_dir.glob("drift_*.json")):
        culture = drift_path.stem[len("drift_") :]
        metrics.extend(_parse_drift(drift_path.read_text(encoding="utf-8"), culture))

    # Dedup on (category, metric) — last wins — then sort for determinism.
    by_key: dict[tuple[str, str], Metric] = {}
    for met in metrics:
        by_key[(met.category, met.metric)] = met
    return sorted(by_key.values(), key=lambda m: (m.category, m.metric))


def ingest_snapshot(
    conn: sqlite3.Connection,
    *,
    label: str,
    captured_at: str,
    metrics: list[Metric],
    source_dir: str | None = None,
    notes: str | None = None,
) -> int:
    """Upsert one snapshot + its metrics. Idempotent: re-ingesting the same
    label replaces its metric rows. Returns the metric row count written."""
    conn.execute(
        "INSERT INTO report_snapshot (label, captured_at, source_dir, notes) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(label) DO UPDATE SET captured_at=excluded.captured_at, "
        "source_dir=excluded.source_dir, notes=excluded.notes",
        (label, captured_at, source_dir, notes),
    )
    conn.execute("DELETE FROM report_metric WHERE snapshot_label = ?", (label,))
    conn.executemany(
        "INSERT INTO report_metric (snapshot_label, category, metric, value_num, value_text) "
        "VALUES (?, ?, ?, ?, ?)",
        [(label, m.category, m.metric, m.value_num, m.value_text) for m in metrics],
    )
    return len(metrics)


def ingest_snapshot_to_db(
    db_path: str | Path,
    *,
    label: str,
    captured_at: str,
    metrics: list[Metric],
    source_dir: str | None = None,
    notes: str | None = None,
) -> int:
    """Path-based wrapper over :func:`ingest_snapshot` that owns the
    write connection (foreign_keys ON, commit, close) — so CLI callers
    stay glue and don't open raw sqlite3 connections (package-boundary
    rule). Returns the metric row count written."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        n = ingest_snapshot(
            conn,
            label=label,
            captured_at=captured_at,
            metrics=metrics,
            source_dir=source_dir,
            notes=notes,
        )
        conn.commit()
    finally:
        conn.close()
    return n


@dataclass(frozen=True)
class MetricDelta:
    category: str
    metric: str
    a_text: str | None
    b_text: str | None
    delta_num: float | None


def diff_snapshots(conn: sqlite3.Connection, label_a: str, label_b: str) -> list[MetricDelta]:
    """Full-outer-join two snapshots' metrics; return rows where the value
    differs (or exists in only one), sorted by category/metric."""
    a = {
        (r[0], r[1]): (r[2], r[3])
        for r in conn.execute(
            "SELECT category, metric, value_num, value_text FROM report_metric "
            "WHERE snapshot_label = ?",
            (label_a,),
        )
    }
    b = {
        (r[0], r[1]): (r[2], r[3])
        for r in conn.execute(
            "SELECT category, metric, value_num, value_text FROM report_metric "
            "WHERE snapshot_label = ?",
            (label_b,),
        )
    }
    out: list[MetricDelta] = []
    for key in sorted(set(a) | set(b)):
        an, atext = a.get(key, (None, None))
        bn, btext = b.get(key, (None, None))
        if atext == btext and an == bn:
            continue
        delta = (bn - an) if (an is not None and bn is not None) else None
        out.append(MetricDelta(key[0], key[1], atext, btext, delta))
    return out

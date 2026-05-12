"""Read-only navigation of the lexicon DB — wyrd-7oma (Phase 1a).

Grep-friendly markdown views of toponyms, etymons, decompositions, and
sources. Surfaces enough context to AUDIT a row before a curation pass
(lemma normalization, dead-rando prune, language retag) edits it via
JSONL events.

Data shape
==========

The module is split in two layers:

- ``fetch_*`` functions take a :class:`sqlite3.Connection` + the lookup
  key, return plain dicts/lists. Easy to unit-test against an in-memory
  fixture DB.
- ``format_*`` functions take those dicts and produce markdown strings.
  Tests check that key fields appear in the output without coupling to
  exact whitespace.

CLI commands in :mod:`wyrd.generators.kenning.cli` glue the two layers
together and ``click.echo`` the result.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

# ---------------------------------------------------------------------------
# Ref parsing helpers
# ---------------------------------------------------------------------------


def parse_etymon_ref(ref: str) -> tuple[str, str]:
    """Split ``"<language>:<canonical_form>"`` into ``(language, form)``.

    Both sides must be non-empty. The L2 round-trip kernel
    (:mod:`jsonl_log`) allows empty ``form`` to preserve junk rows for
    eventual cleanup, but a browser LOOKUP with empty form can't
    return a useful result — surface the bad input early."""
    if ":" not in ref:
        raise ValueError(f"etymon ref must be 'lang:form', got {ref!r}")
    lang, form = ref.split(":", 1)
    if not lang or not form:
        raise ValueError(f"etymon ref must be 'lang:form' with non-empty parts, got {ref!r}")
    return lang, form


def parse_toponym_ref(query: str) -> tuple[str, str | None]:
    """Split ``"<name>@<region>"``. When no ``@`` is present, region is
    ``None`` (caller treats as "any region"). Region of ``-`` or empty
    string means "null region" — the placeholder our JSONL dumper uses."""
    if "@" not in query:
        if not query:
            raise ValueError(f"toponym ref has empty name: {query!r}")
        return query, None
    name, region = query.split("@", 1)
    if not name:
        raise ValueError(f"toponym ref has empty name: {query!r}")
    if region in ("", "-"):
        return name, None
    return name, region


# ---------------------------------------------------------------------------
# Etymon fetchers
# ---------------------------------------------------------------------------


def fetch_etymon(conn: sqlite3.Connection, ref: str) -> dict[str, Any] | None:
    """Look up one etymon by ``"<language>:<canonical_form>"``.

    Returns a dict with the etymon's facts + joined glosses/tags +
    citations + descent edges + lemma family (inflected variants) +
    OCR-cluster siblings. ``None`` when no etymon matches.

    Resolves the lemma-parent and OCR-merge-target refs in the main
    query via two ``LEFT JOIN``s on the etymon table itself, sparing
    the caller two follow-up round trips.
    """
    language, form = parse_etymon_ref(ref)
    row = conn.execute(
        """
        SELECT e.id, e.canonical_form, e.language, e.modifier_type, e.position_pref, e.notes,
               e.lemma_id, e.inflection, e.lemma_method, e.merged_into_id, e.cognate_id,
               e.pronunciation_ipa, e.pronunciation_dialect, e.original_script,
               e.transliteration, e.english_shaped, e.stratum,
               le.language AS lemma_language, le.canonical_form AS lemma_form,
               me.language AS merged_into_language, me.canonical_form AS merged_into_form
          FROM etymon e
          LEFT JOIN etymon le ON le.id = e.lemma_id
          LEFT JOIN etymon me ON me.id = e.merged_into_id
         WHERE e.language = ? AND e.canonical_form = ?
        """,
        (language, form),
    ).fetchone()
    if row is None:
        return None

    eid = row["id"]
    glosses = [
        r["gloss"]
        for r in conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id=? ORDER BY gloss", (eid,)
        )
    ]
    tags = [
        r["tag"]
        for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id=? ORDER BY tag", (eid,))
    ]
    citations = [
        {
            "source_id": r["source_id"],
            "page": r["page"],
            "short_quote": r["short_quote"],
        }
        for r in conn.execute(
            """SELECT source_id, page, short_quote
                 FROM etymon_citation
                WHERE etymon_id = ?
                ORDER BY source_id, page""",
            (eid,),
        )
    ]
    descent_in = [
        {
            "parent_ref": f"{r['p_lang']}:{r['p_form']}",
            "edge_type": r["edge_type"],
            "confidence": r["confidence"],
            "source_id": r["source_id"],
        }
        for r in conn.execute(
            """SELECT pe.language AS p_lang, pe.canonical_form AS p_form,
                      d.edge_type, d.confidence, d.source_id
                 FROM etymon_descent d
                 JOIN etymon pe ON pe.id = d.parent_id
                WHERE d.child_id = ?
                ORDER BY d.id""",
            (eid,),
        )
    ]
    descent_out = [
        {
            "child_ref": f"{r['c_lang']}:{r['c_form']}",
            "edge_type": r["edge_type"],
            "confidence": r["confidence"],
            "source_id": r["source_id"],
        }
        for r in conn.execute(
            """SELECT ce.language AS c_lang, ce.canonical_form AS c_form,
                      d.edge_type, d.confidence, d.source_id
                 FROM etymon_descent d
                 JOIN etymon ce ON ce.id = d.child_id
                WHERE d.parent_id = ?
                ORDER BY d.id""",
            (eid,),
        )
    ]
    # Inflected variants — other etymons pointing lemma_id at this row.
    inflections = [
        {
            "ref": f"{r['language']}:{r['canonical_form']}",
            "inflection": r["inflection"],
        }
        for r in conn.execute(
            """SELECT language, canonical_form, inflection
                 FROM etymon
                WHERE lemma_id = ? AND id != ?
                ORDER BY canonical_form""",
            (eid, eid),
        )
    ]
    # OCR-cluster: rows merged INTO this one.
    merged_in = [
        f"{r['language']}:{r['canonical_form']}"
        for r in conn.execute(
            """SELECT language, canonical_form
                 FROM etymon
                WHERE merged_into_id = ?
                ORDER BY canonical_form""",
            (eid,),
        )
    ]
    # Lemma parent + OCR-merge target refs come from the LEFT JOINs
    # in the main query — no follow-up round trips.
    lemma_ref = f"{row['lemma_language']}:{row['lemma_form']}" if row["lemma_language"] else None
    merged_into_ref = (
        f"{row['merged_into_language']}:{row['merged_into_form']}"
        if row["merged_into_language"]
        else None
    )

    return {
        "ref": ref,
        "etymon_id": eid,
        "canonical_form": row["canonical_form"],
        "language": row["language"],
        "modifier_type": row["modifier_type"],
        "position_pref": row["position_pref"],
        "notes": row["notes"],
        "inflection": row["inflection"],
        "lemma_method": row["lemma_method"],
        "pronunciation_ipa": row["pronunciation_ipa"],
        "pronunciation_dialect": row["pronunciation_dialect"],
        "original_script": row["original_script"],
        "transliteration": row["transliteration"],
        "english_shaped": row["english_shaped"],
        "stratum": row["stratum"],
        "glosses": glosses,
        "tags": tags,
        "citations": citations,
        "descent_in": descent_in,
        "descent_out": descent_out,
        "inflections": inflections,
        "merged_in": merged_in,
        "lemma_ref": lemma_ref,
        "merged_into_ref": merged_into_ref,
    }


# ---------------------------------------------------------------------------
# Toponym fetchers
# ---------------------------------------------------------------------------


def fetch_toponyms_matching(
    conn: sqlite3.Connection, name: str, region: str | None
) -> list[dict[str, Any]]:
    """Return all toponyms matching ``name`` (case-sensitive). When
    ``region`` is None, returns every region the name appears in; when
    set, filters to that one region."""
    if region is None:
        rows = conn.execute(
            "SELECT id, modern_name, country, region FROM toponym WHERE modern_name = ?",
            (name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, modern_name, country, region FROM toponym WHERE modern_name = ? AND region = ?",
            (name, region),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_toponym_detail(conn: sqlite3.Connection, toponym_id: int) -> dict[str, Any]:
    """Full detail for one toponym row: attestations + scholar
    etymologies (with element lists) + matcher decompositions."""
    row = conn.execute(
        "SELECT id, modern_name, country, region FROM toponym WHERE id=?", (toponym_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no toponym row id={toponym_id}")

    attestations = [
        {
            "form": r["form"],
            "date_year": r["date_year"],
            "source_doc": r["source_doc"],
        }
        for r in conn.execute(
            """SELECT form, date_year, source_doc
                 FROM toponym_attestation
                WHERE toponym_id = ?
                ORDER BY COALESCE(date_year, 9999), id""",
            (toponym_id,),
        )
    ]

    # Fetch all elements for this toponym's etymologies in a single
    # join — avoids an N+1 pattern as scholar etymology count grows.
    elements_by_etymology: dict[int, list[dict[str, Any]]] = {}
    for el in conn.execute(
        """SELECT el.toponym_etymology_id AS te_id, el.ordinal, el.inflection,
                  el.surface_in_modern, e.language, e.canonical_form
             FROM toponym_etymology_element el
             JOIN etymon e ON e.id = el.etymon_id
             JOIN toponym_etymology te ON te.id = el.toponym_etymology_id
            WHERE te.toponym_id = ?
            ORDER BY el.toponym_etymology_id, el.ordinal""",
        (toponym_id,),
    ):
        elements_by_etymology.setdefault(el["te_id"], []).append(
            {
                "ordinal": el["ordinal"],
                "etymon_ref": f"{el['language']}:{el['canonical_form']}",
                "inflection": el["inflection"],
                "surface_in_modern": el["surface_in_modern"],
            }
        )

    etymologies: list[dict[str, Any]] = []
    for te in conn.execute(
        """SELECT id, source_id, page, historical_form, confidence, notes, attested_year
             FROM toponym_etymology
            WHERE toponym_id = ?
            ORDER BY source_id, id""",
        (toponym_id,),
    ):
        etymologies.append(
            {
                "etymology_id": te["id"],
                "source_id": te["source_id"],
                "page": te["page"],
                "historical_form": te["historical_form"],
                "confidence": te["confidence"],
                "notes": te["notes"],
                "attested_year": te["attested_year"],
                "elements": elements_by_etymology.get(te["id"], []),
            }
        )

    decompositions = fetch_decompositions(conn, toponym_id)

    return {
        "toponym_id": toponym_id,
        "modern_name": row["modern_name"],
        "country": row["country"],
        "region": row["region"],
        "attestations": attestations,
        "etymologies": etymologies,
        "decompositions": decompositions,
    }


def fetch_decompositions(conn: sqlite3.Connection, toponym_id: int) -> list[dict[str, Any]]:
    """All matcher decompositions for one toponym, canonical-pick first."""
    out: list[dict[str, Any]] = []
    for r in conn.execute(
        """SELECT id, decomposition_signature, morpheme_ids, unaccounted_fragments,
                  unaccounted_count, morpheme_count, is_canonical, canonical_source
             FROM toponym_decomposition
            WHERE toponym_id = ?
            ORDER BY is_canonical DESC, unaccounted_count, morpheme_count""",
        (toponym_id,),
    ):
        try:
            morphemes = json.loads(r["morpheme_ids"]) if r["morpheme_ids"] else []
        except json.JSONDecodeError:
            morphemes = []
        try:
            unaccounted = (
                json.loads(r["unaccounted_fragments"]) if r["unaccounted_fragments"] else []
            )
        except json.JSONDecodeError:
            unaccounted = []
        out.append(
            {
                "decomposition_id": r["id"],
                "signature": r["decomposition_signature"],
                "morphemes": morphemes,
                "unaccounted": unaccounted,
                "unaccounted_count": r["unaccounted_count"],
                "morpheme_count": r["morpheme_count"],
                "is_canonical": bool(r["is_canonical"]),
                "canonical_source": r["canonical_source"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Source fetcher
# ---------------------------------------------------------------------------


def fetch_source(conn: sqlite3.Connection, source_id: str) -> dict[str, Any] | None:
    """Source metadata + per-table contribution counts.

    Toponyms listed are the ones this source has a ``toponym_etymology``
    row about — i.e. the source's scholar-attributed coverage, not the
    full set of toponyms that mention the source in attestation
    free-text.
    """
    row = conn.execute(
        "SELECT id, author, title, year, region, language_focus, notes FROM source WHERE id=?",
        (source_id,),
    ).fetchone()
    if row is None:
        return None

    citation_count = conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE source_id = ?", (source_id,)
    ).fetchone()["n"]
    descent_count = conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_descent WHERE source_id = ?", (source_id,)
    ).fetchone()["n"]
    mining_run_count = conn.execute(
        "SELECT COUNT(*) AS n FROM mining_run WHERE source_id = ?", (source_id,)
    ).fetchone()["n"]
    etymology_count = conn.execute(
        "SELECT COUNT(*) AS n FROM toponym_etymology WHERE source_id = ?", (source_id,)
    ).fetchone()["n"]
    toponyms = [
        {"modern_name": r["modern_name"], "region": r["region"]}
        for r in conn.execute(
            """SELECT DISTINCT t.modern_name, t.region
                 FROM toponym t
                 JOIN toponym_etymology te ON te.toponym_id = t.id
                WHERE te.source_id = ?
                ORDER BY t.modern_name""",
            (source_id,),
        )
    ]
    return {
        "id": row["id"],
        "author": row["author"],
        "title": row["title"],
        "year": row["year"],
        "region": row["region"],
        "language_focus": row["language_focus"],
        "notes": row["notes"],
        "citation_count": citation_count,
        "descent_count": descent_count,
        "mining_run_count": mining_run_count,
        "etymology_count": etymology_count,
        "toponyms": toponyms,
    }


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------


def _kv_lines(pairs: list[tuple[str, Any]]) -> list[str]:
    """Render a list of (label, value) tuples as ``- label: value``
    lines. Skips entries whose value is None or empty string."""
    out: list[str] = []
    for label, value in pairs:
        if value is None or value == "":
            continue
        out.append(f"- {label}: {value}")
    return out


def _truncate_for_grep(value: str | None, max_len: int = 280) -> str | None:
    """Trim very long prose for terminal readability — full text lives
    in the JSONL file for follow-up. ``None`` passes through."""
    if value is None or len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def format_etymon(data: dict[str, Any]) -> str:
    """Render :func:`fetch_etymon` output as grep-friendly markdown."""
    lines: list[str] = [f"## etymon: {data['ref']}", ""]
    lines += _kv_lines(
        [
            ("Canonical form", f"`{data['canonical_form']}`"),
            ("Language", data["language"]),
            ("Modifier type", data["modifier_type"]),
            ("Position preference", data["position_pref"]),
            ("Stratum", data["stratum"]),
            ("IPA", data["pronunciation_ipa"]),
            ("IPA dialect", data["pronunciation_dialect"]),
            ("Original script", data["original_script"]),
            ("Transliteration", data["transliteration"]),
            ("English-shaped", data["english_shaped"]),
            ("Notes", _truncate_for_grep(data["notes"])),
        ]
    )

    if data["glosses"]:
        lines.append("")
        lines.append(f"### Glosses ({len(data['glosses'])})")
        lines += [f"- {g}" for g in data["glosses"]]

    if data["tags"]:
        lines.append("")
        lines.append(f"### Tags ({len(data['tags'])})")
        lines += [f"- {t}" for t in data["tags"]]

    # Lemma family
    if data["lemma_ref"] or data["inflection"] or data["inflections"]:
        lines.append("")
        lines.append("### Lemma family")
        if data["lemma_ref"]:
            infl = f" ({data['inflection']})" if data["inflection"] else ""
            lines.append(f"- This row is an inflection of `{data['lemma_ref']}`{infl}")
            if data["lemma_method"]:
                lines.append(f"- Linked by method: `{data['lemma_method']}`")
        if data["inflections"]:
            lines.append(f"- Inflected variants ({len(data['inflections'])}):")
            for inf in data["inflections"]:
                tag = f" ({inf['inflection']})" if inf["inflection"] else ""
                lines.append(f"  - `{inf['ref']}`{tag}")

    # OCR cluster
    if data["merged_into_ref"] or data["merged_in"]:
        lines.append("")
        lines.append("### OCR cluster")
        if data["merged_into_ref"]:
            lines.append(
                f"- This row is merged INTO `{data['merged_into_ref']}` (citations roll up)"
            )
        if data["merged_in"]:
            lines.append(f"- Other rows merged into this one ({len(data['merged_in'])}):")
            for ref in data["merged_in"]:
                lines.append(f"  - `{ref}`")

    if data["citations"]:
        lines.append("")
        lines.append(f"### Citations ({len(data['citations'])})")
        for c in data["citations"]:
            page = f" p.{c['page']}" if c["page"] else ""
            quote = f' — "{c["short_quote"]}"' if c["short_quote"] else ""
            lines.append(f"- `{c['source_id']}`{page}{quote}")

    if data["descent_in"]:
        lines.append("")
        lines.append(f"### Descent: ancestors ({len(data['descent_in'])})")
        for d in data["descent_in"]:
            conf = f", {d['confidence']}" if d["confidence"] else ""
            lines.append(
                f"- `{d['parent_ref']}` → this ({d['edge_type']}{conf}, source=`{d['source_id']}`)"
            )

    if data["descent_out"]:
        lines.append("")
        lines.append(f"### Descent: descendants ({len(data['descent_out'])})")
        for d in data["descent_out"]:
            conf = f", {d['confidence']}" if d["confidence"] else ""
            lines.append(
                f"- this → `{d['child_ref']}` ({d['edge_type']}{conf}, source=`{d['source_id']}`)"
            )

    return "\n".join(lines)


def format_toponym(data: dict[str, Any]) -> str:
    """Render :func:`fetch_toponym_detail` output as markdown."""
    ref_suffix = f"@{data['region']}" if data["region"] else "@-"
    lines: list[str] = [f"## toponym: {data['modern_name']}{ref_suffix}", ""]
    lines += _kv_lines(
        [
            ("Modern name", data["modern_name"]),
            ("Country", data["country"]),
            ("Region", data["region"]),
        ]
    )

    if data["attestations"]:
        lines.append("")
        lines.append(f"### Attestations ({len(data['attestations'])})")
        for a in data["attestations"]:
            year = f"{a['date_year']}: " if a["date_year"] else ""
            doc = f" ({a['source_doc']})" if a["source_doc"] else ""
            lines.append(f"- {year}{a['form']}{doc}")

    if data["etymologies"]:
        lines.append("")
        lines.append(f"### Scholar etymologies ({len(data['etymologies'])})")
        for te in data["etymologies"]:
            conf = f", {te['confidence']}" if te["confidence"] else ""
            page = f", p.{te['page']}" if te["page"] else ""
            year = f", {te['attested_year']}" if te["attested_year"] else ""
            lines.append(f"- `{te['source_id']}`{conf}{page}{year}")
            if te["historical_form"]:
                lines.append(f"  Historical form: `{te['historical_form']}`")
            if te["notes"]:
                lines.append(f"  Notes: {_truncate_for_grep(te['notes'])}")
            if te["elements"]:
                lines.append("  Elements:")
                for el in te["elements"]:
                    extras: list[str] = []
                    if el["inflection"]:
                        extras.append(f"inflection={el['inflection']}")
                    if el["surface_in_modern"]:
                        extras.append(f"surface={el['surface_in_modern']}")
                    extra = f" ({', '.join(extras)})" if extras else ""
                    lines.append(f"    {el['ordinal']}. `{el['etymon_ref']}`{extra}")

    if data["decompositions"]:
        lines.append("")
        lines.append(f"### Matcher decompositions ({len(data['decompositions'])})")
        lines += _format_decomposition_lines(data["decompositions"])

    return "\n".join(lines)


def _format_decomposition_lines(decompositions: list[dict[str, Any]]) -> list[str]:
    """One line per decomposition: ``- ★ morpheme-breakdown [source]``.

    The breakdown's ``[chars]`` brackets already show unaccounted
    fragment positions; we don't restate them as a trailing list.
    """
    out: list[str] = []
    for d in decompositions:
        star = "★ " if d["is_canonical"] else ""
        source = f" [{d['canonical_source']}]" if d["canonical_source"] else ""
        morphemes = _format_morphemes(d["morphemes"])
        out.append(f"- {star}{morphemes}{source}")
    return out


def _format_morphemes(morphemes: list[Any]) -> str:
    """Render the morpheme_ids list as a pipe-separated breakdown.

    Real-data shape is a list of 2-element ``[kind, value]`` lists
    where ``kind`` is ``"morpheme"`` (a matched canonical form like
    ``"Cot-"``) or ``"unaccounted"`` (a leftover surface fragment).
    Unaccounted fragments render as ``[chars]`` brackets so the
    visual difference is obvious in a grep.

    Legacy / alt shape (per lexicon.sql docstring): bare string for a
    matched morpheme, ``{"unaccounted": "<chars>"}`` dict for a leftover.
    Supported for backward-compat.
    """
    parts: list[str] = []
    for m in morphemes:
        if isinstance(m, list) and len(m) == 2:
            kind, value = m
            if kind == "morpheme":
                parts.append(str(value))
            elif kind == "unaccounted":
                parts.append(f"[{value}]")
            else:
                parts.append(str(m))
        elif isinstance(m, str):
            parts.append(m)
        elif isinstance(m, dict) and "unaccounted" in m:
            parts.append(f"[{m['unaccounted']}]")
        else:
            parts.append(str(m))
    return " | ".join(parts) if parts else "(empty)"


def format_toponym_list(toponyms: list[dict[str, Any]]) -> str:
    """Render multi-region match results — used when a bare ``name``
    matches multiple toponyms."""
    lines: list[str] = [f"## {len(toponyms)} matching toponyms", ""]
    for t in toponyms:
        region = t["region"] or "-"
        country = f" ({t['country']})" if t["country"] else ""
        lines.append(f"- `{t['modern_name']}@{region}`{country}")
    return "\n".join(lines)


def format_decompositions(
    modern_name: str, region: str | None, decompositions: list[dict[str, Any]]
) -> str:
    """Render decompositions for one toponym (no etymology / attestation
    context — for the ``browse decomposition`` subcommand)."""
    ref_suffix = f"@{region}" if region else "@-"
    lines: list[str] = [f"## Decompositions for {modern_name}{ref_suffix}", ""]
    if not decompositions:
        lines.append("(no decompositions — run `lexicon decompose --apply`)")
        return "\n".join(lines)

    canonical = [d for d in decompositions if d["is_canonical"]]
    if canonical:
        lines.append(f"### Canonical pick ({len(canonical)})")
        lines += _format_decomposition_lines(canonical)
        lines.append("")

    others = [d for d in decompositions if not d["is_canonical"]]
    if others:
        lines.append(f"### Alternatives ({len(others)})")
        lines += _format_decomposition_lines(others)

    return "\n".join(lines)


def format_source(data: dict[str, Any], *, list_toponyms: bool = False) -> str:
    """Render :func:`fetch_source` output as markdown. ``list_toponyms``
    expands the per-toponym list (default just shows the count)."""
    lines: list[str] = [f"## Source: {data['id']}", ""]
    lines += _kv_lines(
        [
            ("Title", data["title"]),
            ("Author", data["author"]),
            ("Year", data["year"]),
            ("Region", data["region"]),
            ("Language focus", data["language_focus"]),
            ("Notes", _truncate_for_grep(data["notes"])),
        ]
    )

    lines.append("")
    lines.append("### Contribution counts")
    lines += _kv_lines(
        [
            ("Etymon citations", data["citation_count"]),
            ("Descent edges", data["descent_count"]),
            ("Toponym etymologies", data["etymology_count"]),
            ("Mining-run audits", data["mining_run_count"]),
            ("Distinct toponyms covered", len(data["toponyms"])),
        ]
    )

    if list_toponyms and data["toponyms"]:
        lines.append("")
        lines.append(f"### Toponyms covered ({len(data['toponyms'])})")
        for t in data["toponyms"]:
            region = t["region"] or "-"
            lines.append(f"- `{t['modern_name']}@{region}`")

    return "\n".join(lines)

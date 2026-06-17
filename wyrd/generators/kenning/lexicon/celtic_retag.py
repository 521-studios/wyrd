"""Re-tag flat ``celtic`` L2 etymons to their branch — brittonic / goidelic
(wyrd-xdam.2, part of the wyrd-xdam branch-level Celtic model).

Branch is derived from the COUNTRIES of the toponyms that actually use each
``celtic`` etymon (via ``etymology_element`` → toponym country):

- used only in Goidelic nations (Ireland / Northern Ireland / Scotland / Isle of
  Man) → ``goidelic``
- used only in Brittonic territory (Wales, the English Brythonic substrate,
  Breton France) → ``brittonic``

An etymon used across BOTH branches, used in any unknown-country toponym
(e.g. Gaulish France with no region), or used by no toponym at all STAYS
``celtic`` — the umbrella wyrd-xdam.1 deliberately retained. Branch is never
guessed; only clean, single-branch evidence re-tags.

This operates ONLY on the curated scholarly-gazetteer L2 (Joyce, Watson, Morgan,
KEPN, …). The fine-grained wiktextract Celtic is L3-only and out of scope here.

Idempotent: re-running finds no ``celtic`` etymon left with clean single-branch
evidence (they've already moved), so it's a no-op. L2 is git-tracked, so the
re-tag is a reviewable diff, not data loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.regions import country_for_region

# country → branch. England is included as Brittonic: the pre-Anglo-Saxon Celtic
# substrate of England is Brythonic (wyrd-xdam.2 decision).
_GOIDELIC_COUNTRIES = frozenset({"Ireland", "Northern Ireland", "Scotland", "Isle of Man"})
_BRITTONIC_COUNTRIES = frozenset({"Wales", "England", "France"})

_CELTIC_PREFIX = "celtic:"


@dataclass
class RetagReport:
    distinct_celtic: int = 0  # distinct celtic etymon refs seen
    to_goidelic: int = 0  # distinct refs re-tagged → goidelic
    to_brittonic: int = 0  # distinct refs re-tagged → brittonic
    stayed_celtic: int = 0  # distinct refs left celtic (mixed/unknown/unattached)
    records_rewritten: int = 0  # JSONL records mutated (etymon + ref-bearing rows)


def _country_branch(country: str | None) -> str | None:
    if country in _GOIDELIC_COUNTRIES:
        return "goidelic"
    if country in _BRITTONIC_COUNTRIES:
        return "brittonic"
    return None  # unknown / out-of-scope (Gaulish France, None, …)


def _rebranch(ref: str, branch: str) -> str:
    """``celtic:carn`` + ``goidelic`` → ``goidelic:carn``."""
    _, _, form = ref.partition(":")
    return f"{branch}:{form}"


def _scan(
    all_rows: list[dict],
) -> tuple[dict[str, str | None], set[str], list[tuple[str, list[str]]]]:
    """First pass: collect toponym→country, the set of celtic etymon refs, and
    (toponym_ref, [celtic etymon refs]) links from etymology_element rows."""
    topo_country: dict[str, str | None] = {}
    celtic_refs: set[str] = set()
    elem_links: list[tuple[str, list[str]]] = []
    for r in all_rows:
        t = r.get("_type")
        if t == "toponym":
            ref = r.get("ref")
            if ref:
                topo_country[ref] = r.get("country")
        elif t == "etymon" and r.get("language") == "celtic":
            ref = r.get("ref")
            if ref:
                celtic_refs.add(ref)
        elif t == "etymology_element":
            tr = r.get("toponym_ref")
            cs = [
                e.get("etymon_ref")
                for e in (r.get("elements") or [])
                if str(e.get("etymon_ref", "")).startswith(_CELTIC_PREFIX)
            ]
            if tr and cs:
                elem_links.append((tr, cs))
    return topo_country, celtic_refs, elem_links


def build_branch_map(all_rows: list[dict]) -> dict[str, str]:
    """Map each cleanly-single-branch ``celtic:<form>`` ref → its branch.

    Takes the union of rows across ALL L2 files (an etymon + its uses span files),
    so the decision is global and consistent. Refs that are mixed-branch,
    unknown-country-tainted, or unattached are absent from the map (→ stay celtic).
    """
    topo_country, celtic_refs, elem_links = _scan(all_rows)

    branches: dict[str, set[str]] = {}  # ref → {branches from known countries}
    unknown_tainted: set[str] = set()  # ref used by ≥1 unknown-country toponym
    for tr, cs in elem_links:
        country = topo_country.get(tr)
        if country is None and "@" in tr:  # toponym not in L2 here; derive from region
            country = country_for_region(tr.split("@", 1)[1])
        branch = _country_branch(country)
        for c in cs:
            if branch is None:
                unknown_tainted.add(c)
            else:
                branches.setdefault(c, set()).add(branch)

    out: dict[str, str] = {}
    for ref in celtic_refs:
        if ref in unknown_tainted:
            continue  # any unknown-country usage → not clean
        brs = branches.get(ref)
        if brs and len(brs) == 1:  # single-branch evidence
            out[ref] = next(iter(brs))
    return out


def summarize(branch_map: dict[str, str], distinct_celtic: int, report: RetagReport) -> None:
    """Fill the distinct-ref counters on ``report`` from the built map."""
    report.distinct_celtic = distinct_celtic
    report.to_goidelic = sum(1 for b in branch_map.values() if b == "goidelic")
    report.to_brittonic = sum(1 for b in branch_map.values() if b == "brittonic")
    report.stayed_celtic = distinct_celtic - len(branch_map)


def retag_rows(rows: list[dict], branch_map: dict[str, str], report: RetagReport) -> bool:
    """Rewrite one file's rows IN PLACE: re-tag clean ``celtic`` etymon records to
    their branch and cascade every ``celtic:<form>`` ref (etymon ``ref``, element
    ``etymon_ref``, and any other ref-bearing record like tags/reflexes) to
    ``<branch>:<form>``. Returns True if anything in this file changed."""
    changed = False
    for r in rows:
        t = r.get("_type")
        if t == "etymology_element":
            for e in r.get("elements") or []:
                er = e.get("etymon_ref")
                if isinstance(er, str) and er.startswith(_CELTIC_PREFIX) and er in branch_map:
                    e["etymon_ref"] = _rebranch(er, branch_map[er])
                    report.records_rewritten += 1
                    changed = True
            continue
        # etymon records + any ref-bearing record (tags, reflexes, …)
        ref = r.get("ref")
        if isinstance(ref, str) and ref.startswith(_CELTIC_PREFIX) and ref in branch_map:
            branch = branch_map[ref]
            r["ref"] = _rebranch(ref, branch)
            if r.get("language") == "celtic":
                r["language"] = branch
            report.records_rewritten += 1
            changed = True
    return changed

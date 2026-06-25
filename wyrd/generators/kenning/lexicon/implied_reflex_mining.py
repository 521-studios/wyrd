"""Implied-reflex mining (wyrd-65jh) — residual attribution of a toponym's modern name.

A multi-element toponym breakdown maps a MODERN name to its etymon elements
(``Houghton`` → ``hōh`` + ``tūn``). Some of those elements already have a KNOWN
reflex surface (``tūn`` surfaces as ``ton``); the matcher's cognate-cluster index
(:func:`passthrough_mining._cluster_surfaces`) is exactly the map of which folded
surfaces each cluster can take. When every element BUT ONE is covered by such a
known reflex, the elements they anchor tile a prefix + suffix of the modern name
and leave **one contiguous residual span** in the middle/ends — and that residual
is an IMPLIED reflex of the remaining element (``Houghton`` − ``ton`` ⇒ ``hough``
is an implied reflex of ``hōh``).

This is the residual-attribution sibling of cross-scholar passthrough mining: it
reuses that module's anchor map (``_cluster_surfaces``) and its confidence tiers
(``_confidence_for``), and mirrors its position-aware tiling discipline (sorted
permutations, never the ``toponym_etymology_element.ordinal``, which mismatches
surface order ~40% of the time) in :func:`_walk_order` — the prefix/suffix split
this miner needs (a contiguous residual gap) isn't expressible through
``_segment_order``'s full-coverage tiling. No LLM; deterministic.

Output (two targets, D50 + projection):

* a D50 ``canonical-label@modern-english`` assertion on the residual element's
  ``canonical_morpheme`` node — the SOURCE OF TRUTH (which observed form is the
  modern-stratum label of that morpheme) — paired with a ``mint-canonical``
  declaration of that node so the label PROJECTS even for a SINGLETON residual
  morpheme (``project-canonical`` skips a label whose subject node is unminted, and
  the legacy identity fold only mints MULTI-member families); and
* (effectfully, via :func:`apply_implied_reflexes`) a ``reflex`` row + a
  ``reflex_etymon`` link so the surface feeds the era-grid / bundle, re-dumped to
  ``_reflexes.jsonl`` so it survives a rebuild (the ned5/br5o round-trip).

``surface_in_modern`` is NOT written here — its derivation is owned by wyrd-ujyo
(suffix-anchoring), merged separately; this miner stays scoped to the reflex +
canonical-label/mint-canonical targets.

DATA-QUALITY GUARDS (wyrd-65jh round-2 — drop the empirically-bad classes the
support≥3 band still admitted):

* MIN RESIDUAL ≥ :data:`_MIN_RESIDUAL` (3): a 2-char residual is almost always a
  spurious co-folded fragment (``eh``/``gh``/``nh``/``sh``←hamm, ``le``←hyll,
  ``ly``←lēah) rather than a real reflex.
* PLACEHOLDER / PERSONAL-NAME ETYMON: skip a residual attributed to a non-morpheme
  etymon — a bracket placeholder (``River-name``, ``Place-name``), a CAPITALIZED
  personal name (``Sótr``, ``Ecga``, ``Íri``), or an ``unknown``-language etymon.
  These are toponymic stand-ins, not morphemes with a reflex.
* CO-PRESENT KNOWN-REFLEX: skip a residual that is already the established
  (higher-cluster-evidence) reflex of a DIFFERENT, co-present breakdown element —
  i.e. the residual span is one element's canonical reflex mis-attributed to its
  neighbour (``ton``←dūn, since ``ton`` is the reflex of tūn; ``den``←dūn).

ALIGNMENT GUARDS (false alignment is the empirically top risk — a wrong reflex is
corruption, a missed one is harmless): position-order only; reject a residual
containing whitespace / apostrophe / digit; v1 single-token names only; residual
length ≥ :data:`_MIN_RESIDUAL` AND the anchor span ≥2; MAXIMAL-anchor-coverage
tiling — enumerate every valid tiling
and pick the one that maximizes anchor span (= the SHORTEST residual), so a long
anchor surface consumes the boundary char (``down`` beats ``dow``, ``brad`` beats
``bra``) instead of leaking it into the residual; emit only when that minimal
residual is unique across residual choices (ambiguous → skip, D46 leave-separate);
a SYMMETRIC known-reflex-leftover guard (belt-and-suspenders) — if the residual is a
boundary-misaligned variant of a known reflex of the residual element, reject: the
anchor UNDER-consumed (residual is ``<extra> + knownReflex`` — ``nton`` vs known
``ton`` of tūn; ``dford`` vs ``ford``) OR OVER-consumed (residual is
``knownReflex - <boundary>`` — ``on`` vs known ``ton`` of tūn, the āc anchor matched
``act`` in ``Acton`` and stole the ``t``); skip if the residual already equals a known reflex surface
of ANY cluster in the breakdown (already-known / homograph guard — also dodges the
same-surface dups of wyrd-c7iz). D45: a dash is never identity; the residual
``surface_form`` is the bare de-dashed (folded) span; position is DERIVED display
decoration (prefix→``pre`` / suffix→``post`` / middle→``inner``), never stored
identity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations

from wyrd.generators.kenning.canonicalization import Assertion, NodeRef, mint_canonical_id, validate
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.decomposition_grader import ClusterIndex, load_cluster_index
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface
from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface
from wyrd.generators.kenning.lexicon.passthrough_mining import (
    _cluster_surfaces,
    _confidence_for,
)

METHOD = "implied-reflex-residual-v1"

# The stratum a residual modern-name span labels: the present-day surface of the
# morpheme. The canonical-label value IS that observed modern form.
MODERN_STRATUM = "modern-english"

# Min ANCHOR span length (a 1-char anchor is too noisy to tile on — a coincidence
# rather than a reflex). Mirrors the conservative passthrough caps.
_MIN_SPAN = 2

# Min RESIDUAL length (wyrd-65jh round-2): a 2-char residual is overwhelmingly a
# spurious co-folded fragment (eh/gh/nh/sh←hamm, le←hyll, ly←lēah), not a real
# reflex — require ≥3. Strictly tighter than _MIN_SPAN, applied to the residual gap.
_MIN_RESIDUAL = 3

# Placeholder etymon canonical_forms (bracket toponymic stand-ins, not morphemes):
# a residual attributed to one of these is never a real reflex (wyrd-65jh round-2).
_PLACEHOLDER_FORMS = frozenset({"River-name", "Place-name", "Personal-name"})


def _is_non_morpheme_etymon(language: str, canonical_form: str) -> bool:
    """True when an etymon is a non-morpheme attribution target (wyrd-65jh round-2):
    a bracket PLACEHOLDER (``River-name``), an ``unknown``-language etymon, or a
    CAPITALIZED personal name (``Sótr``, ``Ecga``). Real morphemes are lower-case
    and carry a concrete language; these stand-ins have no reflex to attribute."""
    if not canonical_form:
        return True
    if canonical_form in _PLACEHOLDER_FORMS:
        return True
    if language == "unknown" or not language:
        return True
    # A personal name is title-cased; morphemes (tūn, halh, -ingas) are lower-case.
    return canonical_form[:1].isupper()


# Bound the permutation search the same way passthrough_mining does — a breakdown
# with more anchored elements than this is skipped rather than hung on N!.
_MAX_ANCHORS = 5

# A residual carrying any of these is not a clean morpheme span (multi-token,
# possessive, or a numeric artifact) — reject (folding already drops them, but the
# raw residual is checked BEFORE folding so the guard is explicit).
_NOISE_CHARS = frozenset(" \t'’0123456789")


def _cognate_reflex_expansion(db: LexiconDB, index: ClusterIndex) -> dict[str, set[str]]:
    """cluster key → the UNION of REFLEX surfaces across all cognate-EQUIVALENT
    clusters (the cross-cluster knowledge the OVER-consume guard needs).

    The leak this fixes (wyrd-65jh): an impoverished SINGLETON ASCII cluster (e.g.
    old-english ``Tun`` → ``e4630917``, whose only known surface is its own folded
    canonical_form ``tun``) is a separate cluster from the rich macron ``tūn``
    cognate cluster ``c346202`` (which knows the reflex ``ton``). The element-local
    over-consume guard can't see ``ton``, so ``on``←tun over-consume goes undetected.
    Uniting the clusters' REFLEX surfaces lets the ASCII ``tun`` element inherit
    ``ton`` → ``on`` (a proper suffix of ``ton``) is rejected as over-consume.

    COGNATE EQUIVALENCE SIGNAL (the soundest available, chosen deliberately):
    two clusters are equivalent when they contain etymons sharing the SAME
    ``(language, fold(canonical_form))`` key. This is tight — it unions only
    clusters whose etymons spell the same morpheme in the same language modulo
    macron / case / ligature folding (``Tun`` ≡ ``tūn`` ≡ ``tun`` ≡ ``-tūn`` in
    old-english) — so it directly heals the macron/ASCII split the task names
    WITHOUT the over-broad unioning that would wrongly reject novel residuals.
    Cognate-id grouping is already captured within a ``c<cognate_id>`` cluster;
    the missing piece is exactly this singleton↔cluster fold bridge, which the
    fold key supplies (one hop: every cluster spelling the same morpheme lands in
    the same key bucket).

    REFLEX SURFACES ONLY (not etymon canonical_forms): the over-consume guard asks
    "is the residual a SHORT fragment of a LONGER true reflex of this morpheme?".
    Unioning canonical_forms too pulls in toponym-level junk and proto-forms
    (``melton``, ``keswick``, ``barugaz``, ``presbyter``) that are not reflexes in
    the residual's stratum, and those long strings wrongly reject legitimate
    fragments (``mill``, ``hey``, ``bar``). Restricting the union to actual reflex
    rows (``index.surface_clusters``, the ``reflex`` table) keeps the guard to real
    reflex evidence — ``ton`` is a reflex of tūn; ``melton`` is not.

    Folding within a language is the principled fallback the task sanctions: it is
    strictly NARROWER than co-clustering (same morpheme, same language, spelling
    variation only), so it cannot leak unrelated morphemes' surfaces across.
    """
    # cluster -> its own reflex surfaces (from the reflex table, all positions).
    reflex_of: dict[str, set[str]] = defaultdict(set)
    for surface, clusters in index.surface_clusters.items():
        for cluster in clusters:
            reflex_of[cluster].add(surface)

    # cluster -> the (language, folded_form) keys its etymons carry; and the
    # inverse bucket so cognate-equivalent clusters can be gathered in one hop.
    cluster_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    key_clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in db.conn.execute("SELECT id, cognate_id, language, canonical_form FROM etymon"):
        cid = row["cognate_id"]
        cluster = f"c{cid}" if cid is not None else f"e{row['id']}"
        folded = fold_surface(row["canonical_form"])
        if not folded:
            continue
        key = (row["language"] or "", folded)
        cluster_keys[cluster].add(key)
        key_clusters[key].add(cluster)

    expanded: dict[str, set[str]] = {}
    for cluster in index.cluster_of.values():
        equiv: set[str] = {cluster}
        for key in cluster_keys.get(cluster, ()):
            equiv |= key_clusters.get(key, set())
        surfaces: set[str] = set()
        for c in equiv:
            surfaces |= reflex_of.get(c, set())
        expanded[cluster] = surfaces
    return expanded


@dataclass(frozen=True)
class _Element:
    """One breakdown element: its cluster (anchor key), the etymon it cites, and
    that etymon's stable natural-key parts (language, canonical_form)."""

    etymon_id: int
    cluster: str
    language: str
    canonical_form: str


@dataclass(frozen=True)
class _Breakdown:
    """One scholarly breakdown of a toponym: the (folded) modern name, the owning
    ``toponym_etymology`` row, the toponym's display name, and its ordered (by
    db ordinal — NOT used for tiling) elements."""

    etymology_id: int
    toponym_id: int
    modern_name: str
    folded_name: str
    elements: tuple[_Element, ...]


@dataclass(frozen=True)
class ImpliedReflex:
    """A mined implied reflex: a residual modern-name span attributed to one
    element's etymon.

    Identity is the etymon's stable natural key (``language``, ``canonical_form``)
    + the bare folded ``residual`` + the DERIVED ``position`` — never an etymon
    row-id (rebuild-unstable) and never a dash (D45). ``support`` = distinct
    attesting toponyms; ``sample_toponyms`` are display names for the rationale."""

    language: str
    canonical_form: str
    residual: str
    position: str  # pre | inner | post (derived from the tiling location)
    support: int
    sample_toponyms: tuple[str, ...]


# One cleanly-aligned breakdown: the residual it attributes + the per-element matched
# spans (etymon_id -> folded span). ``spans`` is retained for diagnostics/tiling
# bookkeeping only; surface_in_modern is no longer written here (wyrd-ujyo owns it).
@dataclass(frozen=True)
class _Alignment:
    breakdown: _Breakdown
    residual_element: _Element
    residual: str
    position: str
    spans: dict[int, str]  # etymon_id -> matched modern-name span (diagnostic)


def _position_for(prefix_len: int, suffix_start: int, name_len: int) -> str:
    """Derive the DISPLAY position of the residual from where it tiles the name
    (D45 — position is an output, never stored identity). A residual that starts
    at the head is ``pre``; one that ends at the tail is ``post``; one bounded by
    anchors on both sides is ``inner``."""
    at_head = prefix_len == 0
    at_tail = suffix_start == name_len
    if at_head and not at_tail:
        return "pre"
    if at_tail and not at_head:
        return "post"
    # Bounded both sides (inner), or the degenerate whole-name case (no anchors —
    # excluded upstream, but default to pre for a head-anchored single morpheme).
    return "inner" if not at_head else "pre"


def _align_breakdown(
    bd: _Breakdown,
    cluster_surfaces: dict[str, set[str]],
    expanded_surfaces: dict[str, set[str]],
) -> _Alignment | None:
    """Find the UNIQUE residual element + its span for one breakdown, or ``None``.

    For each candidate residual element, the OTHER elements are anchors that must
    tile a prefix + suffix of the folded name leaving one contiguous gap (the
    residual). Requires exactly one residual element to yield a valid, gap-clean,
    not-already-known tiling (ambiguous across residual choices → skip, D46)."""
    name = bd.folded_name
    if not name or len(bd.elements) < 2 or len(bd.elements) - 1 > _MAX_ANCHORS:
        return None

    valid: list[_Alignment] = []
    for residual_element in bd.elements:
        # PLACEHOLDER / PERSONAL-NAME guard (wyrd-65jh round-2): a residual element
        # that is a non-morpheme stand-in (bracket placeholder, personal name,
        # unknown language) has no reflex to attribute — skip it as a residual
        # candidate.
        if _is_non_morpheme_etymon(residual_element.language, residual_element.canonical_form):
            continue
        anchors = [e for e in bd.elements if e is not residual_element]
        # Every anchor must actually carry at least one folded surface, else it
        # can't tile (an unclustered/formless element makes the residual span
        # ambiguous — leave separate).
        if any(not cluster_surfaces.get(a.cluster) for a in anchors):
            continue
        aln = _residual_from_anchors(bd, name, residual_element, anchors, cluster_surfaces)
        if aln is None:
            continue
        # already-known / CO-PRESENT-known-reflex guard: any folded surface ANY
        # breakdown cluster can take, broadened across COGNATE-EQUIVALENT clusters
        # (expanded_surfaces) so the macron/ASCII split is healed. This catches the
        # co-present contamination where the residual span is the canonical reflex of
        # a DIFFERENT co-present element whose own (impoverished) cluster doesn't list
        # it but its cognate twin does (ton←dūn: 'ton' is the reflex of the
        # co-present tūn; den←dūn: 'den' belongs to denn/denu). Homograph / already
        # handled — don't re-attribute (wyrd-65jh round-2).
        known_surfaces = {
            sf
            for e in bd.elements
            for src in (cluster_surfaces, expanded_surfaces)
            for sf in src.get(e.cluster, ())
            if sf
        }
        if aln.residual in known_surfaces:
            continue
        if _known_reflex_leftover(residual_element, aln, cluster_surfaces, expanded_surfaces):
            # Boundary-leak guard (belt-and-suspenders): the residual element ALREADY
            # has a known reflex that is a prefix/suffix of the residual with a
            # non-empty leftover (e.g. residual 'nton' vs known 'ton' for tūn → the
            # anchor under-consumed the boundary 'n'). Reject (D46 leave-separate —
            # better to miss than mint 'nton'←tūn).
            continue
        valid.append(aln)
    if len(valid) != 1:
        return None  # zero (no clean residual) or ambiguous (>1) → skip (D46)
    return valid[0]


def _known_reflex_leftover(
    residual_element: _Element,
    aln: _Alignment,
    cluster_surfaces: dict[str, set[str]],
    expanded_surfaces: dict[str, set[str]],
) -> bool:
    """True when the computed residual is a BOUNDARY-MISALIGNED variant of a known
    reflex surface of the residual element — the telltale of an adjacent anchor that
    consumed the wrong number of boundary chars. Two symmetric cases (both reject,
    D46 leave-separate — better to miss than mint a misaligned span):

    UNDER-consume (residual is ``<extra> + knownReflex``): a known reflex is a
    proper prefix (``pre``) / suffix (``post``) of the LONGER residual with a
    non-empty leftover — the anchor under-consumed and leaked a char INTO the
    residual (``nton`` vs known ``ton`` of tūn; ``dford`` vs ``ford``). Checked
    against the element's OWN cluster surfaces only (``own``).

    OVER-consume (residual is ``knownReflex - <boundary>``): the residual is a
    proper prefix (``pre``) / suffix (``post``) of a LONGER known reflex — the
    anchor over-consumed a boundary char that belonged to the residual element's
    true reflex (``on`` vs known ``ton`` of tūn — the āc anchor matched ``act`` in
    ``Acton`` and stole the ``t``). Checked against the CROSS-CLUSTER ``expanded``
    set (see below).

    CROSS-CLUSTER over-consume (wyrd-65jh): the OVER-consume direction reads the
    residual element's known reflexes from ``expanded_surfaces`` — the UNION across
    all COGNATE-EQUIVALENT clusters (same morpheme, same language, modulo
    macron/case folding), NOT just the element's own cluster. An impoverished
    singleton ASCII ``tun`` cluster (knowing only ``tun``) thereby inherits ``ton``
    from its macron ``tūn`` cognate cluster, so over-consume fires on ``on``←tun
    (residual ``on`` is a proper suffix of the inherited ``ton``). See
    :func:`_cognate_reflex_expansion`.

    Why ONLY over-consume gets the broadened set: over-consume asks "is the residual
    a SHORT fragment of a longer TRUE reflex?" — exactly the ``on``/``ton`` shape,
    and a residual that is itself a real reflex is short-circuited below, so the
    broadened set is safe here. Under-consume asks the opposite ("is the residual
    ``<extra>+knownReflex``?"); broadening it would reject every LEGITIMATE longer
    residual that merely ends with some short noisy cognate fragment (``worth`` =
    ``w`` + the co-folded fragment ``orth``; ``mill``/``worth``/``pres`` all died
    that way) — so under-consume stays scoped to the element's own cluster.

    NOT-A-LEAK SHORT-CIRCUIT: if the residual is ITSELF an exact member of the
    (expanded) known-surface set, it is a legitimate reflex of this element and is
    NEVER a leak — even though it may also be a proper affix of some LONGER cognate
    surface (the residual ``ton`` is a real reflex of tūn AND a substring of the
    cognate ``eton``/``town``). Without it the broadened over-consume set wrongly
    rejects every true reflex that is a prefix/suffix of a longer cognate form
    (``ton``←tun, ``new``←nīwe, ``ey``←ēg).

    Only the matching end is checked per direction: a ``post`` residual (tail) leaks
    from the head-side anchor, so its known reflex aligns at the SUFFIX; a ``pre``
    residual (head) leaks from the tail-side anchor, so the PREFIX. An ``inner``
    residual is bounded both sides — check both ends. Exact equality
    (``residual == known``) is NOT a leak — both cases require a strict length
    mismatch."""
    residual = aln.residual
    own = {s for s in cluster_surfaces.get(residual_element.cluster, ()) if s}
    expanded = {s for s in expanded_surfaces.get(residual_element.cluster, ()) if s}
    # The over-consume "broadened" set is own ∪ expanded, NOT expanded alone:
    # ``expanded`` (the cross-cluster cognate union, _cognate_reflex_expansion) carries
    # reflex-TABLE surfaces only, while a cluster's longer true reflexes can enter via
    # the etymon canonical_form path — those live in ``own`` (_cluster_surfaces) and are
    # absent from ``expanded``. Reading expanded alone made the over-consume guard blind
    # to them, so a fragment stealing a boundary char off a canonical-form-only reflex
    # leaked through (e.g. ``anger``←hangra, the anchor stole the leading ``h`` of the
    # true reflex ``hanger``; ``bush``←biscop of ``bushop``). The under-consume branch
    # stays own-only (broadening it rejects legitimate longer residuals — see docstring).
    over_consume_known = own | expanded
    if residual in over_consume_known:
        return False  # residual is itself an attested reflex → legitimate, not a leak
    check_post = aln.position in ("post", "inner")
    check_pre = aln.position in ("pre", "inner")
    # UNDER-consume: a known reflex (element's OWN surfaces only — broadening leaks
    # false positives) is a proper affix of the LONGER residual; the anchor leaked
    # a char INTO the residual.
    if any(
        _is_proper_affix(known, residual, check_pre=check_pre, check_post=check_post)
        for known in own
    ):
        return True
    # OVER-consume: the residual is a proper affix of a LONGER known reflex from the
    # element's own surfaces ∪ the CROSS-CLUSTER cognate union (so an impoverished
    # singleton inherits the longer true reflex its residual is a fragment of, AND a
    # canonical-form-only reflex is still consulted); the anchor stole a boundary char.
    return any(
        _is_proper_affix(residual, known, check_pre=check_pre, check_post=check_post)
        for known in over_consume_known
    )


def _is_proper_affix(short: str, long: str, *, check_pre: bool, check_post: bool) -> bool:
    """True if ``short`` is a PROPER prefix (``check_pre``) or PROPER suffix
    (``check_post``) of ``long`` — strictly shorter, so exact equality is excluded.

    The two boundary-misalignment cases in :func:`_known_reflex_leftover` are this
    same relation with the arguments swapped: under-consume asks whether a known
    reflex is a proper affix of the residual; over-consume whether the residual is a
    proper affix of a known reflex. ``check_pre``/``check_post`` gate which ends are
    admissible for the residual's tiling position (a ``post`` tail residual leaks at
    the suffix, a ``pre`` head residual at the prefix, an ``inner`` residual both).
    """
    if len(short) >= len(long):
        return False
    return (check_post and long.endswith(short)) or (check_pre and long.startswith(short))


def _residual_from_anchors(
    bd: _Breakdown,
    name: str,
    residual_element: _Element,
    anchors: list[_Element],
    cluster_surfaces: dict[str, set[str]],
) -> _Alignment | None:
    """Tile ``anchors`` over ``name`` and return the MAXIMAL-anchor-coverage
    alignment attributing the residual gap to ``residual_element``, or ``None``.

    The anchors must tile a PREFIX (``name[0:p]``) and a SUFFIX (``name[s:]``) so
    the gap ``name[p:s]`` is contiguous. We enumerate the prefix/suffix split by
    trying every partition of the anchors into a (prefix-set, suffix-set), tiling
    each side independently (the permutation + per-surface-choice enumeration lives
    in :func:`_tile_lengths` / :func:`_walk_order`), then collect every clean
    candidate residual across all splits.

    FIX (boundary-leak): a short anchor surface (``dow`` of ``down``) leaks the
    boundary char into the residual (``nton`` instead of ``ton``). To stop that we
    prefer the alignment that MAXIMIZES total anchor span — equivalently, the
    SHORTEST residual (total name length is fixed). We only emit when that minimal
    residual is unique; ties at the minimal length are ambiguous → skip (D46)."""
    n_anchors = len(anchors)
    candidates: list[_Alignment] = []
    # Each anchor is assigned to either the prefix block or the suffix block.
    for mask in range(2**n_anchors):
        prefix_anchors = [a for i, a in enumerate(anchors) if not (mask >> i) & 1]
        suffix_anchors = [a for i, a in enumerate(anchors) if (mask >> i) & 1]
        candidates.extend(
            _candidate_splits(
                bd, name, residual_element, prefix_anchors, suffix_anchors, cluster_surfaces
            )
        )
    if not candidates:
        return None
    # Maximal anchor coverage == shortest residual. Pick it; require uniqueness of
    # the residual SURFACE at the minimal length (anchors must consume the boundary
    # char), else the breakdown is ambiguous → skip.
    min_len = min(len(c.residual) for c in candidates)
    minimal = [c for c in candidates if len(c.residual) == min_len]
    distinct = {c.residual for c in minimal}
    if len(distinct) != 1:
        return None
    return minimal[0]


def _candidate_splits(
    bd: _Breakdown,
    name: str,
    residual_element: _Element,
    prefix_anchors: list[_Element],
    suffix_anchors: list[_Element],
    cluster_surfaces: dict[str, set[str]],
) -> list[_Alignment]:
    """Every clean residual alignment from tiling ``prefix_anchors`` at the head and
    ``suffix_anchors`` at the tail. Each anchor side enumerates ALL boundaries it can
    reach (every permutation × per-cluster surface choice), so a longer anchor
    surface that consumes a boundary char is among the candidates the caller ranks
    by maximal coverage. A residual must be a clean span (length ≥
    :data:`_MIN_RESIDUAL`, no noise chars)."""
    prefix_lens = _tile_lengths(name, prefix_anchors, cluster_surfaces, from_head=True)
    suffix_starts = _tile_lengths(name, suffix_anchors, cluster_surfaces, from_head=False)
    matches: list[_Alignment] = []
    for p, pspans in prefix_lens.items():
        for s, sspans in suffix_starts.items():
            if p > s:
                continue
            residual = name[p:s]
            if len(residual) < _MIN_RESIDUAL or any(c in _NOISE_CHARS for c in residual):
                continue
            position = _position_for(p, s, len(name))
            spans = {**pspans, **sspans, residual_element.etymon_id: residual}
            matches.append(
                _Alignment(
                    breakdown=bd,
                    residual_element=residual_element,
                    residual=residual,
                    position=position,
                    spans=spans,
                )
            )
    return matches


def _tile_lengths(
    name: str,
    anchors: list[_Element],
    cluster_surfaces: dict[str, set[str]],
    *,
    from_head: bool,
) -> dict[int, dict[int, str]]:
    """Enumerate every position at which ``anchors`` can tile a contiguous block
    of ``name`` anchored at the head (``from_head=True``) or the tail.

    Returns ``{boundary -> {etymon_id: span}}``: for the head, ``boundary`` is the
    prefix length ``p`` (block is ``name[0:p]``); for the tail, ``boundary`` is the
    suffix start ``s`` (block is ``name[s:]``). An empty anchor list tiles the
    empty block at the boundary (``p=0`` / ``s=len``)."""
    if not anchors:
        return {0: {}} if from_head else {len(name): {}}
    out: dict[int, dict[int, str]] = {}
    clusters = [a.cluster for a in anchors]
    by_cluster: dict[str, list[_Element]] = defaultdict(list)
    for a in anchors:
        by_cluster[a.cluster].append(a)
    for order in sorted(permutations(clusters)):
        # Enumerate ALL surface choices per cluster (not just the first matching),
        # so a longer anchor surface that consumes the boundary char is reachable.
        for spans in _walk_order(name, list(order), by_cluster, cluster_surfaces, from_head):
            boundary = sum(len(v) for v in spans.values())
            key = boundary if from_head else len(name) - boundary
            # Keep one tiling per boundary; distinct permutations / surface picks
            # reaching the same boundary are surface-equivalent for the residual gap
            # (same total span). First wins, deterministic.
            out.setdefault(key, spans)
    return out


def _walk_order(
    name: str,
    order: list[str],
    by_cluster: dict[str, list[_Element]],
    cluster_surfaces: dict[str, set[str]],
    from_head: bool,
):
    """Yield every ``{etymon_id: span}`` tiling of ``order`` (a cluster sequence) as
    a contiguous block from the head or tail of ``name``. Each span must be ≥
    :data:`_MIN_SPAN`. ALL surface choices per cluster are enumerated (so a longer
    anchor surface that consumes a boundary char is among the yielded tilings).
    Anchors of a duplicated cluster are consumed in db order (equivalent — same
    cluster)."""
    # Pre-assign each position in ``seq`` to a concrete etymon_id. Anchors of a
    # duplicated cluster are interchangeable for tiling, so we hand them out in db
    # order: for the head walk left→right, for the tail walk the reversed seq (which
    # consumes the order right-to-left) — matching the original discipline.
    seq = order if from_head else list(reversed(order))
    pool: dict[str, list[int]] = {c: [e.etymon_id for e in es] for c, es in by_cluster.items()}
    take = dict.fromkeys(pool, 0)
    eids: list[int] = []
    for cluster in seq:
        i = take[cluster]
        eids.append(pool[cluster][i])
        take[cluster] += 1

    def step(idx: int, pos: int, end: int, spans: dict[int, str]):
        if idx == len(seq):
            yield dict(spans)
            return
        cluster = seq[idx]
        eid = eids[idx]
        surfaces = sorted(s for s in cluster_surfaces.get(cluster, ()) if s and len(s) >= _MIN_SPAN)
        for sf in surfaces:
            if from_head:
                if not name.startswith(sf, pos):
                    continue
                spans[eid] = sf
                yield from step(idx + 1, pos + len(sf), end, spans)
                del spans[eid]
            else:
                if not name.endswith(sf, 0, end):
                    continue
                spans[eid] = sf
                yield from step(idx + 1, pos, end - len(sf), spans)
                del spans[eid]

    yield from step(0, 0, len(name), {})


def _load_breakdowns(db: LexiconDB, index: ClusterIndex) -> list[_Breakdown]:
    """Every multi-element breakdown with a single-token modern name, carrying its
    ``toponym_etymology`` id, the toponym id + display name, and its elements with
    their stable natural-key parts.

    v1 single-token guard: a modern name with whitespace/apostrophe is skipped here
    (the residual guards would reject it anyway; skipping early bounds the search)."""
    rows = db.conn.execute(
        """
        SELECT te.id AS teid, te.toponym_id AS tid, t.modern_name AS nm,
               tee.etymon_id AS eid, e.language AS lang, e.canonical_form AS cf
        FROM toponym_etymology te
        JOIN toponym t ON t.id = te.toponym_id
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON e.id = tee.etymon_id
        ORDER BY te.id, tee.ordinal
        """
    ).fetchall()
    by_te: dict[int, list] = defaultdict(list)
    meta: dict[int, tuple[int, str]] = {}
    for row in rows:
        by_te[row["teid"]].append(row)
        meta[row["teid"]] = (row["tid"], row["nm"])

    out: list[_Breakdown] = []
    for teid, erows in by_te.items():
        tid, nm = meta[teid]
        if not nm or any(c in _NOISE_CHARS for c in nm):
            continue  # v1: single-token only (raw, pre-fold)
        folded = fold_surface(nm)
        elements = tuple(
            _Element(
                etymon_id=r["eid"],
                cluster=index.cluster_of[r["eid"]],
                language=r["lang"] or "",
                canonical_form=r["cf"],
            )
            for r in erows
        )
        out.append(
            _Breakdown(
                etymology_id=teid,
                toponym_id=tid,
                modern_name=nm,
                folded_name=folded,
                elements=elements,
            )
        )
    return out


def _mine_alignments(db: LexiconDB, index: ClusterIndex) -> list[_Alignment]:
    """All cleanly-aligned breakdowns (one residual each). Deterministic order:
    by etymology id (the input scan order)."""
    cluster_surfaces = _cluster_surfaces(db, index)
    expanded_surfaces = _cognate_reflex_expansion(db, index)
    alignments: list[_Alignment] = []
    for bd in _load_breakdowns(db, index):
        aln = _align_breakdown(bd, cluster_surfaces, expanded_surfaces)
        if aln is not None:
            alignments.append(aln)
    alignments.sort(key=lambda a: a.breakdown.etymology_id)
    return alignments


def extract_implied_reflexes(
    db: LexiconDB,
    *,
    index: ClusterIndex | None = None,
    min_support: int = 1,
) -> list[ImpliedReflex]:
    """Mine implied reflexes by residual attribution over multi-element breakdowns.

    For each breakdown whose anchored elements (those with a known reflex surface)
    tile a prefix + suffix of the folded modern name leaving exactly one contiguous
    residual gap, the gap is attributed to the remaining element as an implied
    reflex. Aggregated per ``(language, canonical_form, residual, position)``;
    ``support`` = distinct attesting toponyms. Records with support ≥
    ``min_support`` are returned, sorted ``(-support, language, canonical_form,
    residual, position)`` (content-determined → rebuild-stable, D36.9).
    """
    if index is None:
        index = load_cluster_index(db)
    alignments = _mine_alignments(db, index)

    Key = tuple[str, str, str, str]
    toponyms: dict[Key, set[int]] = defaultdict(set)
    names: dict[Key, set[str]] = defaultdict(set)
    for aln in alignments:
        e = aln.residual_element
        key = (e.language, e.canonical_form, aln.residual, aln.position)
        toponyms[key].add(aln.breakdown.toponym_id)
        names[key].add(aln.breakdown.modern_name)

    out: list[ImpliedReflex] = []
    for key, tids in toponyms.items():
        if len(tids) < min_support:
            continue
        language, canonical_form, residual, position = key
        out.append(
            ImpliedReflex(
                language=language,
                canonical_form=canonical_form,
                residual=residual,
                position=position,
                support=len(tids),
                sample_toponyms=tuple(sorted(names[key])[:5]),
            )
        )
    out.sort(key=lambda r: (-r.support, r.language, r.canonical_form, r.residual, r.position))
    return out


def implied_reflex_assertions(
    reflexes: list[ImpliedReflex], *, source: str, actor: str = ""
) -> list[Assertion]:
    """Author the D50 canonicalization assertions per mined implied reflex:

    * a ``canonical-label@modern-english`` (the SOURCE OF TRUTH for the residual
      modern label), subject = the residual element's ``canonical_morpheme`` node
      (minted from its rebuild-stable ``(language, canonical_form)``, never an etymon
      row-id, D50.1); the ``value`` qualifier is the residual surface; and
    * a ``mint-canonical`` declaration of that SAME node (wyrd-65jh round-2): without
      it ``project-canonical`` SKIPS the label whose subject node is unminted, and
      the legacy identity fold only mints MULTI-member families — so a SINGLETON
      residual morpheme's committed canonical-label would project to NOTHING. One
      mint per DISTINCT node (a morpheme with two residual labels mints once).

    Deterministic ids (no timestamp) → re-authoring is idempotent (D36.9). The mints
    sort first by id but ``append_assertion`` routes them to ``_canonical_nodes.jsonl``
    while labels land in ``_assert_canonical_label.jsonl`` (per-predicate streams)."""
    out: list[Assertion] = []
    minted_nodes: set[str] = set()
    for r in reflexes:
        node_id = mint_canonical_id("canonical_morpheme", r.language, r.canonical_form)
        if node_id not in minted_nodes:
            minted_nodes.add(node_id)
            mint = Assertion(
                predicate="mint-canonical",
                subject=NodeRef("canonical_morpheme", node_id),
                confidence="high",
                method=METHOD,
                source=source,
                actor=actor,
                rationale=(
                    f"hub for {r.language}:{r.canonical_form} (implied-reflex residual morpheme)"
                ),
            ).with_id()
            validate(mint)
            out.append(mint)
        rationale = (
            f"implied reflex: residual modern-name span '{r.residual}' ({r.position}) "
            f"of {r.language}:{r.canonical_form} (support {r.support}; "
            f"e.g. {', '.join(r.sample_toponyms)})"
        )
        assertion = Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_morpheme", node_id),
            qualifiers={"stratum": MODERN_STRATUM, "value": r.residual},
            confidence=_confidence_for(r.support),
            method=METHOD,
            source=source,
            actor=actor,
            rationale=rationale,
        ).with_id()
        validate(assertion)
        out.append(assertion)
    return out


def apply_implied_reflexes(
    db: LexiconDB,
    *,
    index: ClusterIndex | None = None,
    min_support: int = 2,
) -> dict[str, int]:
    """Effectful projection of mined implied reflexes onto the live DB (the f233 /
    bundle feed). Idempotent: upserts + ``INSERT OR IGNORE`` links.

    ``min_support`` defaults to **2** here (vs 1 for :func:`extract_implied_reflexes`):
    support=1 is weak evidence for the EFFECTFUL reflex/reflex_etymon + canonical-label
    projection (a single attesting toponym is a coincidence-prone singleton; the
    real-data measurement found ~3656 of 4009 are support=1). Extract still returns
    ALL records for vetting; only the WRITE path is gated at ≥2 so production gets the
    clean high-support reflexes, not the noise. Override with the CLI ``--min-support``.

    Per mined implied reflex: upsert a ``reflex(surface_form=residual, position)``
    and link ``reflex_etymon`` to the residual element's etymon (resolved by the
    natural key — productivity is left 0, recomputed at the rebuild tail). The caller
    is responsible for re-dumping ``_reflexes.jsonl`` (``dump_reflexes_to_file``) so
    the reflex round-trips on rebuild.

    NOTE (wyrd-65jh round-2): this miner does NOT write
    ``toponym_etymology_element.surface_in_modern`` — its derivation is owned by
    wyrd-ujyo (suffix-anchoring on ``main``), and the per-source re-dump path it
    required rewrote all L2 (a 7327-row data-loss bug). The reflex + canonical-label /
    mint-canonical are the only effects.

    Returns a counts dict."""
    if index is None:
        index = load_cluster_index(db)
    reflexes = extract_implied_reflexes(db, index=index, min_support=min_support)

    counts = {"reflexes": 0, "reflex_links": 0}

    # Reflex layer: one (surface, position) per mined record, linked to the etymon.
    for r in reflexes:
        eid = _resolve_etymon(db, r.language, r.canonical_form)
        if eid is None:
            continue
        reflex_id = db.upsert_reflex(r.residual, r.position)
        counts["reflexes"] += 1
        db.link_reflex_etymon(reflex_id, eid)
        counts["reflex_links"] += 1

    db.commit()
    return counts


def _resolve_etymon(db: LexiconDB, language: str, canonical_form: str) -> int | None:
    """Resolve the residual element's etymon by its natural key (the live,
    unmerged row). ``None`` if it has been merged away or removed."""
    # De-dash the form (wyrd-aicu.8, D45) so a dashed residual/ref form resolves
    # to the bare-stored etymon.
    canonical_form = normalize_morpheme_surface(canonical_form) or canonical_form
    row = db.conn.execute(
        "SELECT id FROM etymon "
        "WHERE language = ? AND canonical_form = ? AND merged_into_id IS NULL",
        (language, canonical_form),
    ).fetchone()
    return row["id"] if row else None

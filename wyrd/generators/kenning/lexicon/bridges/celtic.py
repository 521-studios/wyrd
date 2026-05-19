"""Anglicized Celtic-substrate form bridging.

Map Anglicized Celtic-substrate place-name forms (``Kil-``, ``Bally-``,
``Glen-``, ``Tully-``, ``Cluain-``, ...) to their canonical
Welsh / Irish / Scottish-Gaelic morphemes. Without this pass, the
Anglicized spellings stand alone in the lexicon — the morpheme is
correctly identified by mining but the lemma rollup doesn't see the
Celtic canonical form, so consensus counts split.

Differs from ``bridge_generic_language``:
* Maps a per-form lookup table (``_CELTIC_FORM_BRIDGES``) rather
  than relying on exact same-form matches.
* Iterates ALL celtic etymons (including pre-existing
  ``merged_into_id`` tombstones) so a stale stub-bridge to an
  unclustered target can be re-routed to a clustered alternative.
* Among multiple matching candidate languages, prefers the one
  whose target is in a cognate cluster, falling back to the caller's
  priority order in ``candidate_langs``.

See ``bridge_celtic_forms``'s own docstring for the full three-way
contrast.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.db import LexiconDB

_CELTIC_FORM_BRIDGES: dict[str, str] = {
    # Goidelic genitive of common landscape nouns
    "choill": "coill",  # gen of coill 'wood'
    "coille": "coill",
    "coillte": "coill",
    "cluana": "cluain",  # gen of cluain 'meadow'
    "chluana": "cluain",  # lenited gen
    "achaidh": "achadh",  # gen of achadh 'field'
    "fearna": "fearn",  # gen of fearn 'alder'
    "easa": "eas",  # gen of eas 'waterfall'
    "gart": "gort",  # gen of gort 'tilled field'
    "inse": "inis",  # gen of inis 'island'
    "innis": "inis",
    "luachra": "luachair",  # gen of luachair 'rushes'
    "tairbh": "tarbh",  # gen of tarbh 'bull'
    "thairbh": "tarbh",  # lenited
    "tulaigh": "tulach",  # gen/dat of tulach 'hill'
    "cnuic": "cnoc",  # gen of cnoc 'hill'
    "capaill": "capall",  # gen of capall 'horse'
    "caorach": "caora",  # gen pl of caora 'sheep'
    "cille": "cill",  # gen of cill 'church'
    "cloiche": "cloch",  # gen of cloch 'stone'
    "coraidh": "cora",  # gen of cora 'weir'
    "croiche": "croch",  # gen of croch 'cross/gallows'
    "croise": "cros",  # gen of cros 'cross'
    "curra": "currach",  # gen of currach 'marsh'
    "daingin": "daingean",  # gen of daingean 'fortress'
    "draoighin": "draighean",  # blackthorn (gen)
    "brighe": "brí",  # gen of brí 'hill'
    "craoibhe": "craobh",  # gen of craobh 'branch/tree'
    "cairn": "carn",  # gen pl of carn 'cairn'
    "cinn": "ceann",  # gen of ceann 'head'
    "aodha": "aodh",  # gen of Aodh (personal name root)
    # Eclipsis (n-/m-/g-/d-/b- prefixed by negation/genitive triggers)
    "gcorr": "corr",  # eclipsed corr 'crane'
    "mban": "bean",  # eclipsed gen pl of bean 'woman'
    "mhic": "mac",  # lenited gen of mac 'son'
    "dhoire": "doire",  # lenited doire 'oak grove'
    # Anglicized place-name spellings
    "drum": "druim",  # Anglicized of druim 'ridge'
    "kin": "ceann",  # Anglicized of cinn (gen of ceann)
    "lough": "loch",  # Anglicized of loch
    "more": "mór",  # Anglicized of mór 'great'
    "boher": "bóthar",  # Anglicized of bóthar 'road'
    "caher": "cathair",  # Anglicized of cathair 'fort'
    "cashel": "caiseal",  # Anglicized of caiseal 'stone fort'
    "carrig": "carraig",  # variant of carraig 'rock'
    "craig": "carraig",
    "creag": "carraig",
    "derry": "doire",  # Anglicized of doire
    "baun": "bán",  # Anglicized of bán 'white'
    "buidhe": "buí",  # older spelling of buí 'yellow'
    "ruadh": "rua",  # older spelling of rua 'red'
    "magh": "mag",  # plain — older spelling
    "maigh": "mag",  # gen/dat of magh
    "din": "dún",  # variant of dún 'fort'
    "clon": "cluain",  # Anglicized of cluain
    "cloon": "cluain",
    "cnocan": "cnocán",  # diminutive of cnoc
    # Other landscape forms (lemma matches a different Wiktionary spelling)
    "ath": "áth",  # ford
    "beal": "béal",  # mouth/opening
    "beinn": "beann",  # peak/mountain
    "clar": "clár",  # plain/board
    "cu": "cú",  # hound
    "eudan": "éadan",  # face/forehead
    "leamhan": "leamhán",  # elm
    "suidhe": "suí",  # seat
    "airgeat": "airgead",  # silver
    "aluin": "álainn",  # beautiful
    "brean": "bréan",  # foul-smelling
}


def bridge_celtic_forms(
    db: LexiconDB,
    *,
    apply: bool = False,
    table: dict[str, str] | None = None,
    candidate_langs: tuple[str, ...] = (
        "irish",
        "scottish-gaelic",
        "manx",
        "old-irish",
        "middle-irish",
        "welsh",
        "old-welsh",
        "middle-welsh",
        "breton",
        "old-breton",
        "middle-breton",
        "cornish",
        "proto-celtic",
    ),
) -> dict:
    """Bridge celtic place-name etymons to Wiktionary lemmas via a
    hand-curated form→lemma table.

    Differs from `bridge_generic_language` in three ways:
      1. Uses a curated form→lemma table (so Goidelic inflections,
         eclipsis/lenition, and Anglicized spellings can be mapped to
         their Wiktionary headwords).
      2. Iterates ALL celtic rows (including pre-existing tombstones)
         so the chain-flatten can re-route an existing stub-bridge
         (celtic→old-irish that has no cognate cluster) to a clustered alternative
         (celtic→irish with a cognate cluster).
      3. Among multiple matching candidate languages, prefers the one
         whose target is in a cognate cluster (cognate_id IS NOT NULL),
         falling back to the priority order in `candidate_langs`.

    Default `candidate_langs` is biased toward MODERN reflexes (Irish,
    Scottish-Gaelic, Welsh) ahead of Old-* / Proto- because Wiktionary's
    Celtic etymology coverage is denser at the modern end — those entries
    are the ones with descent edges that cluster-cognates can walk.

    Returns:
      - examined: total celtic etymons examined (canonical + tombstones)
      - bridged: count that found a (preferred) target
      - unmatched: count with form not in the bridge table
      - missing_target: count where the table named a lemma that doesn't
        exist as a canonical etymon in any candidate language
      - rows_written: actual UPDATE row count when apply=True
    """
    table = _CELTIC_FORM_BRIDGES if table is None else table

    # Build (lower(canonical_form), language) → (live_canonical_id, cognate_id)
    # over canonical candidate-language etymons. We resolve through any
    # merged_into_id chain so a target form named in the table that has
    # itself been OCR-merged into a canonical still routes correctly.
    placeholders = ", ".join(["?"] * len(candidate_langs))
    candidate_index: dict[tuple[str, str], tuple[int, int | None]] = {}
    for row in db.conn.execute(
        f"""
        SELECT id, canonical_form, language, merged_into_id, cognate_id
        FROM etymon
        WHERE language IN ({placeholders})
        ORDER BY id
        """,
        candidate_langs,
    ).fetchall():
        # Resolve through redirect (one hop is enough for our data shape;
        # OCR-cluster passes don't produce multi-step chains within a
        # single language).
        live_id = row["id"]
        live_synset = row["cognate_id"]
        if row["merged_into_id"] is not None:
            live = db.conn.execute(
                "SELECT id, cognate_id FROM etymon WHERE id = ?",
                (row["merged_into_id"],),
            ).fetchone()
            if live is not None:
                live_id = live["id"]
                live_synset = live["cognate_id"]
        key = (row["canonical_form"].lower(), row["language"])
        # First-seen wins per (form, lang). The ORDER BY id keeps it
        # deterministic (older entry wins).
        if key not in candidate_index:
            candidate_index[key] = (live_id, live_synset)

    # Walk ALL celtic rows (canonical + tombstones). Tombstones whose
    # current target is unclustered get re-routed by the chain-flatten
    # OR-clause to the clustered candidate this pass picks.
    celtic_rows = db.conn.execute(
        "SELECT id, canonical_form, merged_into_id "
        "FROM etymon WHERE language = 'celtic' ORDER BY id"
    ).fetchall()

    bridges: list[tuple[int, int]] = []  # (celtic_id, target_id)
    missing_target = 0
    unmatched = 0
    for row in celtic_rows:
        form = row["canonical_form"].lower()
        lemma = table.get(form)
        if lemma is None:
            unmatched += 1
            continue

        # Find the best candidate: prefer clustered targets (cognate_id IS
        # NOT NULL) in priority order, fall back to first-found unclustered.
        best_clustered: int | None = None
        best_unclustered: int | None = None
        for cand_lang in candidate_langs:
            entry = candidate_index.get((lemma.lower(), cand_lang))
            if entry is None:
                continue
            tid, syn = entry
            if syn is not None and best_clustered is None:
                best_clustered = tid
                break  # first clustered wins
            if best_unclustered is None:
                best_unclustered = tid
        target_id = best_clustered if best_clustered is not None else best_unclustered

        if target_id is None:
            missing_target += 1
            continue
        if target_id == row["id"]:
            continue  # bridging to self is a no-op
        bridges.append((row["id"], target_id))

    rows_written = 0
    if apply and bridges:
        # Chain-flatten + lemma-reparent in batch. The OR-clause re-routes
        # any pre-existing redirect that pointed AT this celtic etymon
        # (e.g. an existing stub-bridge) onto the freshly-picked
        # clustered target, preventing a 2-deep chain that would split
        # witnesses in the single-level COALESCE rollup.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, sid, sid, tid) for sid, tid in bridges],
        )
        rows_written = cur.rowcount
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, sid) for sid, tid in bridges],
        )
        db.commit()

    return {
        "examined": len(celtic_rows),
        "bridged": len(bridges),
        "unmatched": unmatched,
        "missing_target": missing_target,
        "rows_written": rows_written,
        "applied": apply,
    }

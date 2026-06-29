"""``wyrd kenning lexicon audit-semantic-coherence`` — flag cluster
pollution + undetected homonyms in the bundle via embedding-based
semantic-similarity comparisons.

User-asked 2026-05-22 after wyrd-hitl shipped: how do we
systematically catch other cluster-pollution cases like 'Hill→cyln'
(kiln) without relying on surface similarity (which sound-changes
break)?

The audit produces TWO CSVs:

1. **cross-sibling-suspects.csv** — for each modern_usage bucket
   with 2+ subjects, the subjects whose meaning glosses are
   semantically distant from their bucket-mates. The 'cyln'-in-
   '-hill' case: a Kiln-meaning subject sitting in a bucket
   otherwise full of Hill/Forest-meaning subjects.

2. **intra-entry-suspects.csv** — for each subject with 2+ glosses,
   the entries whose own gloss list is internally incoherent (e.g.
   'Bank (financial), Bank (river edge)' would be two homonyms
   bundled as one). The homonym-signal column distinguishes
   already-split polysemy from undetected homonyms — see the
   ticket wyrd-36ez for the layered handling of cases like 'bear'.

Both audits use Ollama embeddings (default: mxbai-embed-large at
localhost; override with --model or WYRD_OLLAMA_URL).

No bundle modification, no auto-split. Report-only; operator
reviews + makes split decisions.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import click

# wyrd-36ez round 2 (Gemini HIGH): default to localhost. The
# user's macbook Ollama at 10.5.2.31 is opt-in via the existing
# $WYRD_OLLAMA_URL env var convention (matches the LLM CLIs in
# this package, e.g. mine_llm.py's --ollama-url flag).
DEFAULT_OLLAMA_URL = os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")
# wyrd-a106 bug 2026-05-23: nomic-embed-text returns silently-degenerate
# vectors (norm ~0.05 instead of 1.0) on short single-word inputs even
# WITH the search_document: task prefix. Worse, it sometimes returns
# HTTP 500 ('failed to encode response: json: unsupported value: NaN')
# on certain batches. mxbai-embed-large is robust to short inputs out
# of the box (Finch/a finch = 0.93, Cold/Cool = 0.70, no prefix needed)
# and stays stable under sustained batched load. 1024-dim vs 768-dim;
# slightly heavier but the audit doesn't notice.
DEFAULT_MODEL = "mxbai-embed-large"

# Source-language priority for picking the 'primary' lemma of a
# subject when one subject has multiple source langs in its word.
# Mirrors _ROOT_CODES order in wyrd.generators.kenning.__init__ for
# British place-name etymology relevance.
_PRIMARY_LANG_PRIORITY = (
    "old_english",
    "old_scandinavian",
    "old_french",
    "celtic_mix",
    "latin",
    "germanic",
    "greek",
    "middle_english",
    "modern_english",
)


def _post_embed_request(
    base_url: str, url: str, model: str, texts: list[str], timeout: float
) -> dict:
    """POST the batch to ``url`` (``/api/embed``) and return the parsed payload.
    Wraps the three common transport failures (Ollama down → URLError; model not
    pulled → HTTPError 404; …) as operator-actionable ClickExceptions so the CLI
    exits 1 with a single-line message instead of a traceback (wyrd-36ez)."""
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
        # Ollama returns 404 with "model 'X' not found, try pulling it first"
        # when the model isn't installed.
        if e.code == 404 and "not found" in body_text.lower():
            raise click.ClickException(
                f"Ollama model {model!r} not available at {base_url}. "
                f"Pull it with: ollama pull {model}"
            ) from e
        raise click.ClickException(f"Ollama {url} returned HTTP {e.code}: {body_text[:200]}") from e
    except urllib.error.URLError as e:
        raise click.ClickException(
            f"Could not reach Ollama at {base_url}: {e.reason}. "
            f"Is the service running? Override with --ollama-url or $WYRD_OLLAMA_URL."
        ) from e


def _retry_degenerate_vectors(
    embeddings: list[list[float]], texts: list[str], url: str, model: str, timeout: float
) -> tuple[int, int]:
    """Re-embed degenerate vectors (norm ~0, wyrd-a106) ONE AT A TIME so
    concurrent batch load can't recur during the retry. Mutates ``embeddings``
    in place; returns ``(n_retried, n_retry_failed)``."""
    degen_threshold = 0.5
    n_retried = 0
    n_retry_failed = 0
    for i, v in enumerate(embeddings):
        norm_sq = sum(x * x for x in v)
        if norm_sq >= degen_threshold * degen_threshold:
            continue
        n_retried += 1
        # Re-embed up to 3 times; if still degenerate, accept it and let
        # cosine flagging surface it downstream.
        attempts = 0
        while attempts < 3:
            attempts += 1
            retry_body = json.dumps(
                {"model": model, "input": [texts[i]], "options": {"num_ctx": 8192}}
            ).encode("utf-8")
            retry_req = urllib.request.Request(
                url, data=retry_body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(retry_req, timeout=timeout) as retry_resp:
                retry_payload = json.loads(retry_resp.read().decode("utf-8"))
            retry_vecs = retry_payload.get("embeddings") or []
            if retry_vecs:
                new_v = retry_vecs[0]
                if sum(x * x for x in new_v) >= degen_threshold * degen_threshold:
                    embeddings[i] = new_v
                    break
        else:
            n_retry_failed += 1  # all 3 retries failed
    return n_retried, n_retry_failed


def _ollama_embed(
    base_url: str, model: str, texts: list[str], timeout: float = 60.0
) -> list[list[float]]:
    """POST /api/embed with a batch of input strings. Returns a list
    of dense vectors (one per input, same order).

    Ollama's /api/embed accepts a single string or a list. We always
    send the list shape for consistent response parsing.

    wyrd-36ez round 2 (Gemini MED): wraps transport errors with
    operator-actionable messages. The three common failure modes:
    Ollama not running (URLError), model not pulled (HTTPError 404
    with 'model not found' body), unexpected response shape
    (RuntimeError below). Each raises a ClickException so the CLI
    exits with code 1 + a clear single-line message instead of a
    Python traceback."""
    url = f"{base_url.rstrip('/')}/api/embed"
    payload = _post_embed_request(base_url, url, model, texts, timeout)
    # Response shape: {"model": ..., "embeddings": [[...], [...], ...]}
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise click.ClickException(
            f"Ollama embed returned unexpected shape: "
            f"expected {len(texts)} vectors, got "
            f"{len(embeddings) if isinstance(embeddings, list) else type(embeddings)}"
        )
    # wyrd-a106 bug 2026-05-23: mxbai-embed-large under batched load
    # intermittently returns degenerate vectors (norm ~0.05 instead of 1.0) for
    # some inputs while others in the same batch come back clean — a transient
    # model-state issue, not malformed input. Detect + retry one at a time.
    n_retried, n_retry_failed = _retry_degenerate_vectors(embeddings, texts, url, model, timeout)
    if n_retried:
        import sys as _sys

        print(
            f"  [embed batch] retried {n_retried} degenerate vectors "
            f"({n_retry_failed} still degenerate after 3 attempts)",
            file=_sys.stderr,
        )
    return embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns
    0.0 on degenerate (zero-norm) inputs to avoid div-by-zero."""
    # strict=True: vectors should always be same dim; an embedding-
    # model-side mismatch would be a real bug worth raising on.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _primary_source_lemma(word: dict) -> tuple[str, str] | None:
    """Pick the (source_lang, first_lemma) for a word according to
    _PRIMARY_LANG_PRIORITY. Returns None when no priority lang has
    a non-empty form list."""
    for lang in _PRIMARY_LANG_PRIORITY:
        forms = word.get(lang)
        if isinstance(forms, list) and forms:
            first = forms[0]
            if isinstance(first, str) and first.strip():
                return lang, first
    return None


def _load_audit_allowlist(path: Path | None, event_type: str) -> frozenset[tuple[str, str]]:
    """Load an operator-maintained allowlist of (source_lang, source_lemma)
    pairs filtered by ``_type=<event_type>`` from the shared audit
    allowlist file. Two event types ride in the same file today:

    - ``known_polysemy`` — confirmed real polysemy (one word with
      multiple related senses) that the cosine audit can't tell apart
      from undetected homonyms; filtered out of intra-entry-suspects.
    - ``known_cluster_acceptable`` — confirmed legitimate cross-language
      homographs that legitimately share a bundle surface form (e.g.
      OE ``ost`` 'knot in a tree' and OF ``ost`` 'army' both contribute
      to ``-ost-`` toponyms); filtered out of cross-sibling-suspects.

    Returns an empty frozenset when ``path`` is None, an empty string,
    a directory, or a nonexistent file (operator bypass shapes — empty
    ``--known-…-file ""`` is the natural "turn this off" gesture).
    Raises ``click.ClickException`` on malformed JSON or
    missing-required-field rows — those are operator typos, not
    bypass intent.

    File format: JSONL. One synthetic ``_type=source`` row at the top
    (operator note) plus one ``_type=<event_type>`` row per entry with
    ``source_lang`` and ``source_lemma`` fields keyed in the BUNDLE
    form (underscore-joined, e.g. ``old_english`` not the lexicon's
    ``old-english``). An optional ``reason`` field rides along for
    audit-log readability; only ``source_lang`` + ``source_lemma``
    participate in the filter. Both keys are ``.strip()`` ed on load
    to guard against trailing whitespace from hand-edited entries.
    Rows of any other ``_type`` are silently skipped so a single file
    can carry multiple event types without cross-pollination.
    """
    # Operator-bypass shapes: None / empty / "." / nonexistent / dir.
    # Click parses ``--known-…-file ""`` into ``Path('.')``,
    # which exists as a directory; without this guard the .open() call
    # below would raise IsADirectoryError mid-audit.
    if path is None or str(path) in ("", ".") or not path.exists() or path.is_dir():
        return frozenset()
    entries: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise click.ClickException(
                    f"{path}:{line_no}: malformed JSON in allowlist: {exc}"
                ) from exc
            if row.get("_type") != event_type:
                continue
            lang = row.get("source_lang")
            lemma = row.get("source_lemma")
            if not lang or not lemma:
                raise click.ClickException(
                    f"{path}:{line_no}: {event_type} row missing source_lang and/or source_lemma"
                )
            entries.add((lang.strip(), lemma.strip()))
    return frozenset(entries)


def _load_known_polysemy(path: Path | None) -> frozenset[tuple[str, str]]:
    """Back-compat wrapper around :func:`_load_audit_allowlist` for
    ``known_polysemy`` rows (intra-entry-suspects filter)."""
    return _load_audit_allowlist(path, "known_polysemy")


def _load_known_cluster_acceptable(path: Path | None) -> frozenset[tuple[str, str]]:
    """Load the ``known_cluster_acceptable`` allowlist (cross-sibling-
    suspects filter). See :func:`_load_audit_allowlist`."""
    return _load_audit_allowlist(path, "known_cluster_acceptable")


def _apply_allowlist_filter(
    rows: list[dict],
    allowlist: frozenset[tuple[str, str]],
) -> tuple[list[dict], int]:
    """Drop rows whose (source_lang, source_lemma) is in the allowlist.

    Returns ``(kept_rows, dropped_count)``. Both inputs use the BUNDLE
    underscore-joined form (see :func:`_load_audit_allowlist`).
    ``.strip()`` on both sides defends against hand-edited whitespace
    drift in the allowlist file or bundle.
    """
    if not allowlist:
        return rows, 0
    kept = [
        r for r in rows if (r["source_lang"].strip(), r["source_lemma"].strip()) not in allowlist
    ]
    return kept, len(rows) - len(kept)


# Back-compat alias retained so external test imports from PR #344
# don't break. New call sites should use the unprefixed
# ``_apply_allowlist_filter`` since the shape is event-type-agnostic.
_apply_polysemy_filter = _apply_allowlist_filter


def _build_audit_entities(subjects: list, limit: int | None) -> list[dict]:
    """One audit entity per (subject, word) pair — most subjects have one word,
    but a words list can contribute several modern_usages. Skips subjects with
    no glosses and words with no usage / no primary source lemma."""
    entities: list[dict] = []
    for i, subject in enumerate(subjects):
        if limit is not None and i >= limit:
            break
        glosses = subject.get("meaning") or []
        if not glosses:
            continue
        for w_idx, word in enumerate(subject.get("words") or []):
            usage = word.get("modern_usage")
            if not usage:
                continue
            primary = _primary_source_lemma(word)
            if primary is None:
                continue
            entities.append(
                {
                    "subject_idx": i,
                    "word_idx": w_idx,
                    "modern_usage": usage,
                    "source_lang": primary[0],
                    "source_lemma": primary[1],
                    "glosses": glosses,
                    "joined": " | ".join(glosses),
                }
            )
    return entities


def _embed_entities(entities: list[dict], ollama_url: str, model: str, batch_size: int) -> None:
    """Pass 1: embed each entity's joined gloss list (batched), storing the
    dense vector on ``e['vector']``. Echoes throughput progress."""
    click.echo(
        f"Embedding {len(entities)} entity gloss lists via {model} @ {ollama_url}...",
        err=True,
    )
    t0 = time.time()
    for batch_start in range(0, len(entities), batch_size):
        batch = entities[batch_start : batch_start + batch_size]
        texts = [e["joined"] for e in batch]
        vectors = _ollama_embed(ollama_url, model, texts)
        for e, v in zip(batch, vectors, strict=True):
            e["vector"] = v
        if batch_start % (batch_size * 10) == 0:
            done = min(batch_start + batch_size, len(entities))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            click.echo(
                f"  [{done}/{len(entities)}]  {rate:.1f} entities/s",
                err=True,
            )
    click.echo(
        f"  done in {time.time() - t0:.1f}s ({len(entities) / max(time.time() - t0, 0.001):.1f} entities/s)",
        err=True,
    )


def _bucket_by_usage(
    entities: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Pass 2: bucket entities by modern_usage. Returns ``(buckets,
    multi_buckets)`` — multi_buckets keeps only the ≥2-sibling buckets."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        buckets[e["modern_usage"]].append(e)
    multi_buckets = {k: v for k, v in buckets.items() if len(v) >= 2}
    return buckets, multi_buckets


def _cross_row(usage: str, s: dict, sibs: list[dict]) -> dict:
    """One cross-sibling suspect row: ``s``'s avg/min/max cosine to its
    bucket-mates + the distinct other-source lemmas in the bucket."""
    others = [o for o in sibs if o is not s]
    sims = [_cosine(s["vector"], o["vector"]) for o in others]
    avg = sum(sims) / len(sims) if sims else 0.0
    other_lemmas = sorted(
        {(o["source_lang"], o["source_lemma"]) for o in others}
        - {(s["source_lang"], s["source_lemma"])}
    )
    return {
        "modern_usage": usage,
        "subject_idx": s["subject_idx"],
        "source_lang": s["source_lang"],
        "source_lemma": s["source_lemma"],
        "glosses": s["joined"],
        "n_bucket_mates": len(others),
        "avg_cosine_to_mates": round(avg, 4),
        "min_cosine_to_mate": round(min(sims), 4) if sims else 0.0,
        "max_cosine_to_mate": round(max(sims), 4) if sims else 0.0,
        "other_bucket_lemmas": "; ".join(f"{lang}:{lemma}" for lang, lemma in other_lemmas),
    }


def _compute_cross_rows(multi_buckets: dict[str, list[dict]]) -> list[dict]:
    """Pass 3: cross-sibling avg-similarity rows, sorted ascending by avg cosine
    (lowest = most suspect cluster pollution)."""
    cross_rows: list[dict] = []
    for usage, sibs in multi_buckets.items():
        for s in sibs:
            cross_rows.append(_cross_row(usage, s, sibs))
    cross_rows.sort(key=lambda r: r["avg_cosine_to_mates"])
    return cross_rows


def _embed_per_gloss(
    multi_gloss: list[dict], ollama_url: str, model: str, batch_size: int
) -> tuple[list[list[float]], list[tuple[int, int]]]:
    """Pass 4: flatten the multi-gloss entities' glosses, embed them in batches,
    and return ``(flat_vectors, gloss_offsets)`` where each offset is the
    ``(start_in_flat, length)`` slice for that entity."""
    click.echo(
        f"Embedding per-gloss vectors for {len(multi_gloss)} multi-gloss entities...",
        err=True,
    )
    t0 = time.time()
    flat_glosses: list[str] = []
    gloss_offsets: list[tuple[int, int]] = []  # (start_in_flat, length) per entity
    for e in multi_gloss:
        gloss_offsets.append((len(flat_glosses), len(e["glosses"])))
        flat_glosses.extend(e["glosses"])
    flat_vectors: list[list[float]] = []
    for batch_start in range(0, len(flat_glosses), batch_size):
        batch_texts = flat_glosses[batch_start : batch_start + batch_size]
        flat_vectors.extend(_ollama_embed(ollama_url, model, batch_texts))
        if batch_start % (batch_size * 10) == 0:
            done = min(batch_start + batch_size, len(flat_glosses))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            click.echo(
                f"  [{done}/{len(flat_glosses)}]  {rate:.1f} glosses/s",
                err=True,
            )
    click.echo(
        f"  done in {time.time() - t0:.1f}s",
        err=True,
    )
    return flat_vectors, gloss_offsets


def _intra_row(
    e: dict, vecs: list[list[float]], length: int, buckets: dict[str, list[dict]]
) -> dict:
    """One intra-entry suspect row: pairwise-cosine stats among ``e``'s gloss
    vectors + the homonym signal (other source lemmas sharing this surface)."""
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(_cosine(vecs[i], vecs[j]))
    avg = sum(sims) / len(sims) if sims else 0.0
    bucket_mates = [o for o in buckets[e["modern_usage"]] if o is not e]
    same_surface_other_lemmas = sorted(
        {(o["source_lang"], o["source_lemma"]) for o in bucket_mates}
        - {(e["source_lang"], e["source_lemma"])}
    )
    return {
        "modern_usage": e["modern_usage"],
        "subject_idx": e["subject_idx"],
        "source_lang": e["source_lang"],
        "source_lemma": e["source_lemma"],
        "glosses": e["joined"],
        "n_glosses": length,
        "avg_intra_pairwise_cosine": round(avg, 4),
        "min_intra_pairwise_cosine": round(min(sims), 4) if sims else 0.0,
        "same_surface_other_lemmas": "; ".join(
            f"{lang}:{lemma}" for lang, lemma in same_surface_other_lemmas
        ),
        "has_other_lemmas": bool(same_surface_other_lemmas),
    }


def _compute_intra_rows(
    multi_gloss: list[dict],
    flat_vectors: list[list[float]],
    gloss_offsets: list[tuple[int, int]],
    buckets: dict[str, list[dict]],
) -> list[dict]:
    """Pass 5: intra-entry coherence rows, sorted ascending by avg pairwise
    cosine (lowest = most likely undetected homonym)."""
    intra_rows: list[dict] = []
    for e, (start, length) in zip(multi_gloss, gloss_offsets, strict=True):
        vecs = flat_vectors[start : start + length]
        intra_rows.append(_intra_row(e, vecs, length, buckets))
    intra_rows.sort(key=lambda r: r["avg_intra_pairwise_cosine"])
    return intra_rows


@click.command("audit-semantic-coherence")
@click.option(
    "--meanings",
    "meanings_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("audit"),
    show_default=True,
    help="Directory for the two output CSVs.",
)
@click.option(
    "--ollama-url",
    default=DEFAULT_OLLAMA_URL,
    show_default=True,
    help=(
        "Ollama base URL. Default is localhost; "
        "set $WYRD_OLLAMA_URL or pass --ollama-url to point elsewhere."
    ),
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Embedding model (Ollama). mxbai-embed-large is robust on short single-word glosses.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Strings per /api/embed call. Bigger = fewer round trips but more memory + risk of timeout.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after embedding this many subjects (for quick iteration).",
)
@click.option(
    "--top",
    type=int,
    default=200,
    show_default=True,
    help="Rows per output CSV (sorted ascending by similarity — lowest = most suspect).",
)
@click.option(
    "--known-polysemy-file",
    "known_polysemy_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/audit/known_polysemy.jsonl"),
    show_default=True,
    help=(
        "Allowlist of (source_lang, source_lemma) pairs that are confirmed real "
        "polysemy — one word with multiple related senses — that the cosine "
        "audit cannot tell apart from undetected homonyms. Filtered out of the "
        "intra-entry-suspects CSV. Operator-maintained JSONL. Pass an empty "
        "string or a nonexistent path to skip filtering."
    ),
)
@click.option(
    "--known-cluster-acceptable-file",
    "known_cluster_acceptable_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/audit/known_polysemy.jsonl"),
    show_default=True,
    help=(
        "Allowlist of (source_lang, source_lemma) pairs that are confirmed "
        "legitimate cross-language homographs — two different etymons that "
        "rightfully share a bundle surface form (e.g. OE ``ost`` 'knot in a "
        "tree' and OF ``ost`` 'army' both contribute to ``-ost-`` toponyms). "
        "Filtered out of the cross-sibling-suspects CSV. Defaults to the same "
        "file as --known-polysemy-file; rows are filtered by ``_type`` so a "
        "single file can carry both event types. Pass an empty string or a "
        "nonexistent path to skip filtering."
    ),
)
def lexicon_audit_semantic_coherence(
    meanings_path: Path | None,
    output_dir: Path,
    ollama_url: str,
    model: str,
    batch_size: int,
    limit: int | None,
    top: int,
    known_polysemy_path: Path,
    known_cluster_acceptable_path: Path,
) -> None:
    """Embedding-based audit for bundle cluster pollution + undetected
    homonymy. Produces two suspect CSVs for human review.

    See module docstring for design rationale. Report-only — never
    modifies the bundle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    known_polysemy = _load_known_polysemy(known_polysemy_path)
    if known_polysemy:
        click.echo(
            f"Loaded {len(known_polysemy)} known-polysemy entries from "
            f"{known_polysemy_path} (filtered out of intra-entry-suspects)",
            err=True,
        )
    known_cluster_acceptable = _load_known_cluster_acceptable(known_cluster_acceptable_path)
    if known_cluster_acceptable:
        click.echo(
            f"Loaded {len(known_cluster_acceptable)} known-cluster-acceptable "
            f"entries from {known_cluster_acceptable_path} "
            "(filtered out of cross-sibling-suspects)",
            err=True,
        )

    if meanings_path is None:
        click.echo("Loading bundle from L4 runtime DB...", err=True)
        from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
        from wyrd.generators.kenning.runtime.runtime_db_adapter import (
            bundle_dict_from_runtime_db,
        )

        bundle = bundle_dict_from_runtime_db(get_runtime_db())
    else:
        click.echo(f"Loading {meanings_path}...", err=True)
        with meanings_path.open(encoding="utf-8") as f:
            bundle = json.load(f)
    subjects = bundle.get("subjects", [])
    click.echo(f"  {len(subjects)} subjects", err=True)

    entities = _build_audit_entities(subjects, limit)
    click.echo(f"  {len(entities)} audit entities (subject × word)", err=True)

    # === Pass 1: embed each entity's joined gloss list ============
    _embed_entities(entities, ollama_url, model, batch_size)

    # === Pass 2: bucket entities by modern_usage ==================
    buckets, multi_buckets = _bucket_by_usage(entities)
    click.echo(
        f"{len(multi_buckets)} multi-sibling buckets (out of {len(buckets)} total)",
        err=True,
    )

    # === Pass 3: cross-sibling avg similarity =====================
    cross_rows = _compute_cross_rows(multi_buckets)
    cross_rows, cross_filtered_out = _apply_allowlist_filter(cross_rows, known_cluster_acceptable)
    cross_path = output_dir / "cross-sibling-suspects.csv"
    _write_csv(cross_path, cross_rows[:top])
    cross_msg = (
        f"wrote {len(cross_rows[:top])} suspects to {cross_path} "
        f"(of {len(cross_rows)} total across {len(multi_buckets)} multi-sibling buckets"
    )
    if cross_filtered_out:
        cross_msg += f"; {cross_filtered_out} filtered as known cluster-acceptable"
    cross_msg += ")"
    click.echo(cross_msg, err=True)

    # === Pass 4: intra-entry coherence (per-gloss embed needed) ==
    multi_gloss = [e for e in entities if len(e["glosses"]) >= 2]
    flat_vectors, gloss_offsets = _embed_per_gloss(multi_gloss, ollama_url, model, batch_size)

    # === Pass 5: intra-entry pairwise-cosine rows ================
    intra_rows = _compute_intra_rows(multi_gloss, flat_vectors, gloss_offsets, buckets)
    intra_rows, intra_filtered_out = _apply_allowlist_filter(intra_rows, known_polysemy)
    intra_path = output_dir / "intra-entry-suspects.csv"
    _write_csv(intra_path, intra_rows[:top])
    msg = (
        f"wrote {len(intra_rows[:top])} suspects to {intra_path} "
        f"(of {len(intra_rows)} multi-gloss entries"
    )
    if intra_filtered_out:
        msg += f"; {intra_filtered_out} filtered as known polysemy"
    msg += ")"
    click.echo(msg, err=True)

    click.echo(
        "\nReview decision matrix:\n"
        "  intra low + has_other_lemmas=True   → polysemy within already-split homonym; LEAVE\n"
        "  intra low + has_other_lemmas=False  → likely missing homonym; SPLIT\n"
        "  cross low + has_other_lemmas any    → likely cluster pollution; REVIEW",
        err=True,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def add_to(parent: click.Group) -> None:
    """Register ``audit-semantic-coherence`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_semantic_coherence)

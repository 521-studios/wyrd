"""``wyrd kenning lexicon adjudicate-element-glosses`` — weak-tail LLM pick (wyrd-u9k6).

For each sub-threshold ('weak') surface in ``_element_glosses.jsonl`` that still
has consensus candidates (real etymons from the toponyms' actual etymologies),
a fast LLM PICKS the true element among them — never invents. Accepted
high/medium picks land in ``_element_gloss_adjudications.jsonl``, which
``run_full_enrichment`` replays alongside the consensus glosses. Resumable.

Progress line per the mining-progress convention (CLAUDE.md): echoes every ~25
surfaces with an s/each rate so operators can extrapolate ETA.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import click


@click.command("adjudicate-element-glosses")
@click.option(
    "--consensus",
    "consensus_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/mining/_element_glosses.jsonl"),
    show_default=True,
    help="The consensus jsonl (mine-element-glosses output) — supplies the weak rows.",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Lexicon DB supplying the toponym corpus for grounding context.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_element_gloss_adjudications.jsonl"),
    show_default=True,
)
@click.option("--ollama-url", default=None, help="Override $WYRD_OLLAMA_URL.")
@click.option("--model", default="qwen2.5:7b", show_default=True)
@click.option("--limit", type=int, default=None, help="Process only the first N (smoke test).")
@click.option("--timeout", type=float, default=60.0, show_default=True)
def lexicon_adjudicate_element_glosses(
    consensus_path: Path,
    db_path: Path,
    output_path: Path,
    ollama_url: str | None,
    model: str,
    limit: int | None,
    timeout: float,
) -> None:
    """LLM-adjudicate the weak tail (PICK among grounded candidates)."""
    import os

    import requests

    from wyrd.generators.kenning.lexicon.element_gloss_backfill import (
        ADJUDICATION_SYS,
        accept_adjudication,
        build_adjudication_user,
        load_toponym_lowers,
        toponym_context,
    )

    url = (ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")) + "/api/chat"
    toponym_lowers = load_toponym_lowers(str(db_path))

    with open(consensus_path, encoding="utf-8") as fh:
        weak = [
            r
            for r in (json.loads(line) for line in fh)
            if not r.get("is_element") and r.get("candidates")
        ]
    weak.sort(key=lambda r: -r.get("weight", 0))
    if limit:
        weak = weak[:limit]

    done: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                with contextlib.suppress(json.JSONDecodeError, KeyError):
                    done.add(json.loads(line)["surface"])
    todo = [r for r in weak if r["surface"] not in done]

    def ask(system: str, user: str) -> dict:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        return json.loads(resp.json()["message"]["content"])

    t0 = time.time()
    accepted = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for n, r in enumerate(todo, 1):
            ctx = toponym_context(r["surface"], r["position"], toponym_lowers)
            rec = {
                "surface": r["surface"],
                "weight": r.get("weight", 0),
                "position": r["position"],
                "model": model,
            }
            try:
                response = ask(
                    ADJUDICATION_SYS,
                    build_adjudication_user(r["surface"], r["position"], ctx, r["candidates"]),
                )
                cand = accept_adjudication(r, response)
                rec.update(
                    choice=response.get("choice"),
                    confidence=response.get("confidence"),
                    why=response.get("why", ""),
                )
                rec["accepted"] = cand is not None
                if cand is not None:
                    rec["candidate"] = cand
                    accepted += 1
            except Exception as exc:  # noqa: BLE001 — record + continue, don't abort the batch
                rec["error"] = str(exc)[:120]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0:
                rate = (time.time() - t0) / n
                click.echo(
                    f"  [{n}/{len(todo)}]  accepted={accepted} ({rate:.1f}s/each, "
                    f"ETA {rate * (len(todo) - n) / 60:.0f}m)",
                    err=True,
                )
    click.echo(
        f"[{len(todo)}/{len(todo)}]  accepted={accepted} -> {output_path} "
        f"({(time.time() - t0) / 60:.0f}m). Replayed by run_full_enrichment.",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``adjudicate-element-glosses`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_adjudicate_element_glosses)

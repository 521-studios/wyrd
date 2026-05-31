"""``wyrd kenning lexicon mine-pronunciation-llm`` — LLM IPA for no-table langs.

PAID/LLM step (Ollama). Appends to ``data/mining/_pronunciation.jsonl``, the L2
record replayed deterministically at enrichment. Resumable. Emits progress per
the mining-progress convention.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click
import requests

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon.pronunciation_mining import (
    LLM_LANGUAGES,
    build_messages,
    parse_response,
    record,
    select_targets,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_DEFAULT_JSONL = "data/mining/_pronunciation.jsonl"


def _load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # A row counts as done only if it captured a real outcome (IPA or an
        # explicit no-ipa skip). Rows carrying an "error" key were transient
        # failures (timeout, Ollama down) — leave them out so a resumed run
        # retries them rather than treating the blip as permanent.
        if rec.get("language") and rec.get("form") and "error" not in rec:
            done.add((rec["language"], rec["form"]))
    return done


@click.command("mine-pronunciation-llm")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path(_DEFAULT_JSONL))
@click.option("--model", default=os.environ.get("WYRD_PRON_MODEL", "qwen2.5:7b"))
@click.option("--ollama-url", default=os.environ.get("WYRD_OLLAMA_URL", "http://10.5.2.31:11434"))
@click.option(
    "--language",
    "languages",
    multiple=True,
    help="Restrict to these languages (default: all no-table generation langs).",
)
def lexicon_mine_pronunciation_llm(
    db_path: Path, out_path: Path, model: str, ollama_url: str, languages: tuple[str, ...]
) -> None:
    """Mine broad IPA for the no-G2P-table generation languages (wyrd-vm8t)."""
    langs = tuple(languages) if languages else tuple(LLM_LANGUAGES)
    targets = select_targets(str(db_path), langs)
    done = _load_done(out_path)
    todo = [(lang, form) for lang, form in targets if (lang, form) not in done]
    click.echo(f"targets={len(targets)} done={len(done)} todo={len(todo)}", err=True)

    chat_url = ollama_url.rstrip("/") + "/api/chat"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ipa_n = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for n, (lang, form) in enumerate(todo, 1):
            try:
                resp = requests.post(
                    chat_url,
                    json={
                        "model": model,
                        "messages": build_messages(lang, form),
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.1},
                    },
                    timeout=60,
                )
                resp.raise_for_status()  # surface HTTP status (e.g. model not loaded)
                parsed = parse_response(resp.json()["message"]["content"])
                rec = record(lang, form, parsed, model)
                if parsed is not None:
                    ipa_n += 1
            except Exception as exc:  # noqa: BLE001 — record the failure, keep going
                rec = {**record(lang, form, None, model), "error": str(exc)[:80]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            # Progress on both paths (per the mining convention; final line always fires).
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                eta = rate * (len(todo) - n) / 60
                click.echo(
                    f"  [{n}/{len(todo)}] ipa={ipa_n} ({rate:.1f}s/entry, ETA {eta:.0f}m)", err=True
                )


def add_to(parent: click.Group) -> None:
    """Register ``mine-pronunciation-llm`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_pronunciation_llm)

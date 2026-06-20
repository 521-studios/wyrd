# Ingesting a new etymology source

> **New here?** Start with [`OVERVIEW.md`](OVERVIEW.md) — it frames
> what this whole sub-system is doing and why. This doc assumes you
> already know the answers to "what's the authoring layer?", "why
> are we mining?", and "what's the three-tier provider model?" If
> any of those are unfamiliar, OVERVIEW first.

This is the recipe for taking a previously-unknown public-domain (or
otherwise authorized) etymology text and wiring its data into the
authoring lexicon. The **architectural decisions** that motivate each
step live in `DECISIONS.md`; this doc is procedural — what to do, in
what order, and what to look at when something goes wrong.

The pipeline produces rows in the SQLite lexicon at
`~/.wyrd/lexicon.db` (override with `WYRD_LEXICON_DB`; run `wyrd
kenning lexicon path` to confirm). The runtime generator still reads
`data/meanings.json`, which is exported from the lexicon by a separate
`export-meanings` step that you only run when you're ready to ship a
new bundle.

---

## TL;DR — happy-path checklist

```bash
# 1. Acquire the OCR text
curl -sL "https://archive.org/.../book_djvu.txt" -o sources/<id>.txt

# 2. Document the addition
$EDITOR sources/MANIFEST.md      # add bibliographic entry

# 3. Smoke-test the parser yield (no LLM call yet)
.venv/bin/python -c "from wyrd.generators.kenning.cli import _select_parser_and_run as p; \
    print(len(p(open('sources/<id>.txt').read(), 'auto')))"

# 4. Pick the tier (see § Tier choice) and mine
.venv/bin/wyrd kenning lexicon mine-llm sources/<id>.txt \
    --provider ollama         # English (default)
    # or --provider anthropic # Celtic / Irish / Welsh / Manx

# 4b. Tier 2 review — Gemini Flash on Qwen low/medium-confidence rows.
#     Standard step for English; skip for Celtic (Haiku is already Tier 1).
.venv/bin/wyrd kenning lexicon review sources/ --provider gemini --apply

# 5. Post-mining cleanup (LLM-free; cheap)
/tmp/post-mining-pipeline.sh    # link-lemmas → normalize-ocr → reverse-search → fuzzy-search → report
```

---

## Stage 0 — does this source belong in the corpus?

Before downloading anything, check the sensitivity heuristic in
`DECISIONS.md` D7. PD status is a legal property, not an ethical one.

- **Mine and bundle**: Mayan, Nahuatl, Egyptian, Sumerian, Greek,
  Roman, all European corpora. PD or licensed.
- **Framework only** (refuse to bundle): Native American, Aboriginal
  Australian, Maori, Hawaiian, Khoisan. We provide the language-pack
  format; community members supply the data.
- **Edge cases** (ask first): Hebrew biblical strata = bundle; Hebrew
  modern political = caution. Bantu varies by community size and
  politics.

If unsure, default to framework-only and surface the question.

---

## Stage 1 — acquire the OCR text

Source: usually Internet Archive's `_djvu.txt` (the OCR full-text
artifact). Save under `sources/<authorlast>_<year>_<region>.txt`.

Naming convention examples already in the corpus:

- `mawer_1920_northumberland_durham.txt`
- `joyce_1875_irish_names_vol1.txt`
- `mawer_stenton_1924_introduction_to_survey.txt`

**The whole `sources/` tree is gitignored** (~1.8G of OCR books +
OS OpenNames CSVs). Only `sources/MANIFEST.md` is checked in.

If the publication date is post-1928 (US-PD cutoff at the time of
writing this doc), check the country of publication and the user's
position on whether to include — different jurisdictions have
different PD lengths, and we have a Mawer/Stenton 1930 EPNS volume
in the corpus that the user added explicitly.

---

## Stage 2 — document the addition

Edit `sources/MANIFEST.md` and add a one-line entry under the right
regional section:

```markdown
- `<author>_<year>_<region>.txt` — <Author>, *<Title>* (<year>).
  Brief one-liner about what's in it.
```

This is the file that lets a fresh checkout know what should be
in `sources/` even though the OCR text isn't committed.

---

## Stage 3 — smoke-test the parser yield

Before spending any LLM tokens, sanity-check that the regex-based
parser actually finds entries in the book. The parser auto-picks
between `parse_skeat_text` (suffix-section format) and
`parse_alphabetical_text` (Mawer/Ekwall/Johnston flat layout).

```bash
.venv/bin/python -c "
from wyrd.generators.kenning.cli import _select_parser_and_run as p
from pathlib import Path
text = Path('sources/<id>.txt').read_text()
entries = p(text, 'auto')
print(f'parser found {len(entries)} entries')
for e in entries[:3]:
    print(f'  {e.toponym!r}: {e.body_text[:80]!r}')
"
```

**Yield diagnostics:**

| Yield | Likely cause | What to do |
| --- | --- | --- |
| 0 entries | Parser doesn't recognize headword shape, or `find_body_bounds` clipped to a tiny region | Check `find_body_bounds(lines)` output; if body span is < 5% of total lines, the safety net should kick in (PR #4). If still 0, the book is structurally non-dictionary (Joyce 1875 vol 1/2 are thematic chapters; LLM will still process the prose but most entries will be sentences) |
| 1–10 entries | Body-bounds heuristic is mis-clipping, or the book has a non-standard structure | Inspect `_BODY_START_PATTERNS` matches in the file — the largest-span vote picks the best, but if all candidates are tiny the safety net falls back to the whole doc |
| Hundreds, but mostly false positives | Parser is finding sentence starts, not real headwords | The `_entry_body_is_real` filter should drop most; per `wyrd-3qq` the `_ENTRY_BODY_SIGNALS` regex covers English + Celtic + Norse + Manx. Add new markers if the book uses idiosyncratic conventions |
| Reasonable count (50–500) | Normal | Proceed |

**Failure modes seen in this corpus:**

- **Joyce 1875 vol 1/vol 2** are thematic chapter books, not dictionaries.
  Parser yields ~1100 candidate entries each — most are chapter-section
  prose openings, not headwords. The LLM correctly declines them. The
  legitimate index entries at the tail still extract cleanly. The
  empirical signature for this book class is **~67% decline, ~14%
  reject** (combined: 240 accepted / 754 declined / 155 rejected
  across vol1+vol2). Match this shape early on a new candidate book
  to flag "thematic chapters, not dictionary" before sinking serious
  token budget.
- **Mawer 1922 / Mawer 1924 / Mawer-Stenton 1924** are methodology
  works ("Place-Names and History", "Chief Elements", "Introduction
  to Survey") — dictionaries about toponymy, not place-name dictionaries.
  Expect very low accept rate. Mining is still worth running because
  the bibliographic / element-discussion sections sometimes name
  specific places with attestations.
- **Skeat 1904 Huntingdonshire** had 100% decline — the book is
  effectively a list of names without etymology bodies.
- **Skeat 1904 Hertfordshire** (companion volume same year) had
  ~26% rejection — well above the 5–15% normal band. Format
  anomaly worth investigating but didn't block mining; the
  accepted rows were fine. Per-book OCR/format checks before full
  mining can save tokens on the worst offenders.

---

## Stage 4 — pick the tier

`DECISIONS.md` D2 lays out the three-tier provider model; D13 says
the tier choice is **per-book**, not per-language-family.

Empirically:

| Content shape | Tier 1 (bulk) | Approximate cost |
| --- | --- | --- |
| English dictionary (Mawer, Skeat 1906+, Ekwall, Wyld-Hirst, EPNS) | **Ollama / Qwen 3.5** at `http://10.5.2.31:11434` | Free |
| Celtic, Irish, Welsh, Manx, Pictish (Joyce, Moore, Johnston Scotland, Gillies, Macbain, Watson, Morgan) | **Anthropic Haiku 4.5** | ~$0.60 per 200-entry book |
| Old Norse content (Lindkvist) | Try Ollama first; if Qwen yield < 30% switch to Haiku | Variable |
| Anything Sonnet-tier | **Don't.** Per `DECISIONS.md` D19, Sonnet doesn't lift over Gemini Flash for this task and reserve API budget for runtime user features (explainer, register-shift, MCP) | — |

**Watson 1904 is a low-yield book regardless of provider** (~7% accept
on Qwen, ~3% on Haiku, both around 380 parsed entries each). It looks
like an English-prose book but the toponyms are Scottish-Gaelic
substrate, and neither tier handles that well. The book is a
philological essay, not a dictionary; mining costs more than it
returns. By contrast, **Watson 1926 (Celtic Place-Names of Scotland)
is the case where Haiku earns its keep**: 52% accept rate (200/387)
on the same provider that scored 3% on Watson 1904. The shape of the
book matters more than the shape of the toponym substrate. Smoke-test
30 entries before committing to a full provider pass on any book whose
content layout you don't recognize.

**Quilgars 1906 (Loire-Inférieure) is the third instance of this
class** — 30-entry Haiku smoke yielded 1 acceptance (3%). It's a
*topographical* dictionary (alphabetical headwords + property records:
"Kerorven, manoir en Guérande, 1476") rather than a *toponymic
etymology* dictionary. The headwords are dense in Breton morphemes
(Plou-/Ker-/Tre-/Lan- show up by the hundreds) but the bodies are
deed citations, not glossed etymological breakdowns. Counting
morphemes in raw OCR overstates yield — agents that probe via keyword
counting will flag this kind of book as a strong candidate; verify by
running a 5–30 entry mining smoke before committing API budget (see
`bd memories trust-audit-table-over-agent-summaries`).

**Joret 1881 (patois Normand du Bessin) is a fourth class** — title's
*"suivi d'un dictionnaire étymologique"* suggests toponymic etymology
but the dictionary section is **Norman dialect WORDS + Latin etymons**
(`Etande` ← *extendere*, `Foche` ← *focacia*, `FoRTEUNE` ← *fortuna*),
not place-name etymology. Pipeline-mismatched with the existing
toponym mine-llm path; same shape as Troude 1876 / Le Gonidec /
Châlons-Loth Vannetais — language dictionaries that fit the
wyrd-4rt Wiktionary-style ingestion path, not the toponym extractor.
Probe before mining: read 6-10 sample entries to confirm shape
("Foochat, s. m.: focal" vs. "Foochat, ... from O.E. focal-tūn").

**Mawer 1920 Northumberland-Durham via Gemini Flash 2.5 as Tier-1**
(2026-05-03, wyrd-m70) — successful "different model on already-mined
book" pattern. Gemini Tier-1 hit 62% accept on a book that Qwen had
done at 47% and Haiku at 60%; the Gemini-tagged 501 acceptances added
an independent 3rd-model witness layer + 121 new unique lemmas at
1+ witness. Cost ~\$2.50 over ~95 minutes. Free-tier daily quota was
sufficient for one such pass; concurrent Gemini Tier-1 work hits
the rate limit fast. Note: works cleanly when the prior mining used
Qwen (Tier-2 review path filters Ollama-tagged rows and slips to
Tier-1 cleanly), but Haiku-mined books historically had a unique-
(toponym_id, source_id) collision that blocked 3rd-model attribution
until the candidate query was extended (PR #66 / wyrd-eca, 2026-05-05).

**Gemini Tier-2 OVER Haiku Tier-1 lifts on non-Celtic books**
(2026-05-05, wyrd-eca + 5-book empirical pass). Pattern: a non-Celtic
book mined with Haiku Tier-1 (because the content shape was unfamiliar
or the Welsh-Marches NF / Norman-influenced layer made Qwen weak)
still benefits from Gemini Tier-2 as a second-witness layer. The
`lexicon review --include-haiku` flag widens the candidate query
(was limited to Ollama/Qwen-tagged rows). Empirical accept rates on
the low/medium-confidence subset of each book's Haiku rows:

| Book | Haiku rows | Reviewed (low+med) | Gemini written | Accept |
| --- | ---: | ---: | ---: | ---: |
| moorman_1910_west_riding_yorkshire | 102 | 32 | 15 | 47% |
| bannister_1916_herefordshire | 66 | 22 | 10 | 45% |
| arbois_1890_recherches_propriete_fonciere | 235 | 66 | 23 | 35% |
| duignan_1902_staffordshire | 78 | 34 | 7 | 21% |
| goodall_1914_sw_yorkshire | 76 | 12 | 2 | 17% |

Total +57 Gemini-tagged rows. Roughly: clean English county / Norman-
adjacent content sits at 45-50% accept, generic English at 17-21%,
Celtic+Roman at 35%. The candidate query only picks up low/medium
confidence Haiku rows, so the high-confidence majority (Haiku already
got it right) is left alone — the lift is concentrated where Haiku
hedged.

Workflow: after Haiku Tier-1 completes on a non-Celtic book, run
`lexicon review --book <id> --provider gemini --include-haiku --apply`.
Idempotent against the same `provider_tag` so re-running is safe.

**Gemini Tier-2 OVER Haiku is also productive on pure Celtic content**
(2026-05-05, after the Joyce 1898 smoke). The original "skip Tier-2
for Celtic" rule was justified by Qwen's failure as Tier-1, NOT by
testing Gemini-on-Haiku. The empirical data inverts the rule:

| Celtic book | Reviewed (low+med) | Gemini written | Accept |
| --- | ---: | ---: | ---: |
| joyce_1898_irish_names_vol3 (smoke) | 30 | 24 | **80%** |
| joyce_1898_irish_names_vol3 (full) | +24 | +12 | 50% |
| joyce_1913_irish_names_of_places | 32 | 17 | 53% |
| watson_1926_celtic_place_names_of_scotland | 66 | 14 | 21% |
| johnston_1904_stirlingshire | 51 | 18 | 35% |
| morgan_1912_wales | 39 | 21 | 54% |
| macbain_1922_highlands_and_islands | 37 | 16 | 43% |
| watson_1904_ross_and_cromarty (haiku id) | 8 | 3 | 38% |

Total +125 Gemini-tagged rows on top of Haiku Tier-1 across 7 Celtic
books. The smoke rate (joyce_1898 at 80%) overstates full-pass yield
in the same way as the Watson-1904 calibration warning above, but the
floor across the seven books is 21% — well above the 15% threshold
where Tier-2 stops paying for itself. Updated rule:

- Run Gemini Tier-2 on Haiku-Tier-1 Celtic books with `lexicon review
  --book <id> --provider gemini --include-haiku --apply`. Idempotent.

The 5 Celtic Sonnet-Tier-1 books were swept post-PR #69 (`--include-sonnet`)
on 2026-05-05 and yielded **+150 Gemini Tier-2 rows on n=252 reviewed
(60% weighted, 52–83% per-book)** — confirming Gemini lifts over Sonnet
on Celtic toponyms at roughly the same rate it lifts over Haiku:

| Book | Reviewed | Written | Accept |
| --- | ---: | ---: | ---: |
| joyce_1875_irish_names_vol1 | 105 | 59 | 56% |
| joyce_1875_irish_names_vol2 | 86 | 45 | 52% |
| moore_1890_isle_of_man | 29 | 24 | 83% |
| johnston_1892_place_names_of_scotland | 20 | 12 | 60% |
| morgan_1887_wales_monmouthshire | 12 | 10 | 83% |

**Tier 2 review (Gemini Flash) is the standard second pass on English
mining**, not a contingency. After `mine-llm --provider ollama`
completes, run `lexicon review --provider gemini --apply` — see
Stage 5 below. Celtic mining via Haiku does NOT go through this
two-pass recipe; Haiku is already a strong-enough Tier 1 for that
content. Optional Tier 3: `--provider anthropic` for whatever stays
uncertain after Gemini. Per D19 don't reach for Sonnet here.

**The Ollama instance only runs one inference at a time.** If you queue
two parallel mining batches against `10.5.2.31`, total throughput
doesn't change — the books just complete in interleaved order. Haiku
runs entirely independently of Ollama, so a Qwen batch and a Haiku
batch can run in parallel and double the throughput.

**Required env (Ollama):**

```bash
WYRD_OLLAMA_URL=http://10.5.2.31:11434  # operator's macbook (Ollama host)
```

**Required env (Haiku):**

```bash
ANTHROPIC_API_KEY=...
```

Qwen requires `"think": false` in the chat payload — already wired
into `OllamaClient` in `extractors/llm.py`. The thinking variant
consumes all tokens in the reasoning channel and emits empty content
otherwise.

---

## Stage 5 — mine

For a single book:

```bash
WYRD_OLLAMA_URL=http://10.5.2.31:11434 \
  .venv/bin/wyrd kenning lexicon mine-llm sources/<id>.txt
# or
.venv/bin/wyrd kenning lexicon mine-llm sources/<id>.txt --provider anthropic
```

For a batch, write a tiny shell loop and run it under
`Bash run_in_background` plus a `Monitor` filter. See
`/tmp/mine-english-tier-2.sh` for the canonical pattern. Key shape:

```bash
for book in "${BOOKS[@]}"; do
  echo "BOOK START: $book" >&2          # Monitor filter picks up these markers
  .venv/bin/wyrd kenning lexicon mine-llm "sources/${book}.txt" 2>&1 \
    | tail -3 \
    | sed "s/^/  [$book] /"
  echo "BOOK COMPLETE: $book" >&2
done
echo "BATCH COMPLETE" >&2
```

Monitor pattern:

```text
grep -E "BOOK (START|COMPLETE)|BATCH COMPLETE|accepted=|Traceback|Error|RuntimeError"
```

### Reading the mining output

Each book ends with three lines:

```
[<book>] Done in <s>s (<rate>s/entry). accepted=<A> declined=<D> rejected=<R>
[<book>] Rejection reasons: {'form_not_in_body': N, ...}
[<book>] Ingested: {'toponyms': X, 'etymologies': Y, 'elements': Z, 'etymons_touched': W}
```

| Counter | Meaning |
| --- | --- |
| `accepted` | Model returned a structured etymology AND every claimed form passed the form-in-body validator (`DECISIONS.md` D3). Becomes a `toponym_etymology` row. |
| `declined` | Model returned `found: false` ("no etymology in this body"). Common on chapter prose, methodology books, and entries that are just cross-references. NOT a rejection — the row is recorded with `confidence='low'` and an empty elements list (the `accepted=true` path with no elements). |
| `rejected` | Model returned an etymology but the form-in-body validator caught a hallucinated or OCR-garbled form. The row is dropped entirely. Healthy rate is 5–15%; >25% means investigate (probably bad OCR or a content-format mismatch). |

Acceptance rate by book type:

| Book type | Typical accept rate |
| --- | ---: |
| Skeat county dictionary | 80–90% |
| Mawer EPNS / Ekwall county | 50–80% |
| Comparative philology (Taylor, Mcclure) | 3–30% |
| Methodology volumes (Mawer 1922/1924/Mawer-Stenton 1924) | 5–25% |
| Celtic dictionary via Haiku | 40–60% |
| Celtic prose chapters via Haiku | 5–15% |
| Gemini Tier-2 OVER Haiku on non-Celtic Haiku-mined book | 17–47% (low/med subset) |
| Gemini Tier-2 OVER Haiku on Celtic+Roman content | 35% (Arbois 1890, n=66) |
| Gemini Tier-2 OVER Haiku on pure Celtic | 21–53% across 7 books, n=287 (2026-05-05 confirmed) |
| Gemini Tier-2 OVER Sonnet on Celtic | 52–83% across 5 books, n=252 (2026-05-05 confirmed) |

Low accept is not always a bug — it's often the right answer for the
book type. Consult the table before assuming the pipeline is broken.

**Smoke-yield bias caveat (2026-05-03):** the first N entries are not
representative of full-pass yield on treatise-class books. d'Arbois
1890 hit 40% accept on smoke-30 but 10% on the full 960 pass — the
early Roman/Gaulish discussion chapters were dense; later ones got
abstract. **Treat the smoke rate as an upper bound for treatise /
chrestomathy / methodology books** (the alphabetical-county-dict class
is more uniform). For yield projections, sample two slices: the
default first-30 plus an explicit `--limit` skip-then-take that
samples deeper into the body.

**Rejection-reason diagnostic:** a single rejection reason >30% of
total rejections is a signal that the validator's design context may
not fit the source class — investigate before assuming the book is
low-yield. See `DECISIONS.md` D5 (wyrd-z56) for the canonical
worked example (d'Arbois 1890 / 800-year floor / PR #42 fix).

### Tier 2 review pass (English only — Gemini Flash)

**Smoke-test the prompt first.** Before kicking off a full review run
on a 200-entry book, point the same code at 5–10 candidate rows to
catch prompt regressions or schema mismatches before you've burned
API tokens at scale:

```bash
.venv/bin/wyrd kenning lexicon review sources/ --provider gemini \
    --book mawer_1920_northumberland_durham --limit 10
# (without --apply this is a dry run)
```

If the breakdowns and confidence flags look sane, drop `--limit` and
add `--apply` for the real pass. Cheap insurance — the SYSTEM_PROMPT
gets edited often enough that mass-rerunning a stale prompt across
3000 entries is a real risk.

After Tier 1 mining completes, run the layered review:

```bash
.venv/bin/wyrd kenning lexicon review sources/ --provider gemini --apply
```

What it does:

- Picks `toponym_etymology` rows whose notes indicate Ollama (Qwen)
  origin AND whose `confidence` is in `('low', 'medium')`.
- Re-parses the source text from `sources/`, runs the entry through
  Gemini Flash, and **inserts a NEW `toponym_etymology` row** tagged
  `extracted_by:gemini:<model>` in `source_quote`. The original Qwen
  row stays untouched.
- The consensus view shows where the two providers agree or disagree;
  promotion-eligibility (≥3 witnesses) rolls up across both.
- **Idempotent**: skips toponyms that already have a row from the
  chosen provider, so you can safely interrupt and resume.

Required env: `WYRD_GEMINI_API_KEY=...` (model defaults to
`gemini-2.5-flash`; override via `WYRD_GEMINI_MODEL`).

Optional Tier 3 — Anthropic on whatever's still uncertain after
Gemini:

```bash
.venv/bin/wyrd kenning lexicon review sources/ --provider anthropic --apply
```

Per `DECISIONS.md` D19, the Sonnet uplift over Gemini Flash for
mining was empirically ~zero. Reserve API budget for runtime user
features (explainer, register-shift, MCP).

**Tier-2 on Celtic / Welsh / Irish / Manx IS productive over Haiku**
(confirmed 2026-05-05). The original "skip Tier-2 for Celtic" rule
came from Qwen's failure as Tier-1 on Celtic, NOT from testing
Gemini-on-Haiku. With wyrd-eca's `--include-haiku` flag (PR #66) the
candidate query reaches Haiku-tagged rows, and the empirical floor
across 7 books is 21% accept (Watson 1926) with peaks at 50-53%
(Joyce 1898, Joyce 1913, Morgan 1912). See the case study above for
the full per-book table.

For non-Celtic Haiku-mined books (Bannister, Harrison, Moorman,
Arbois, etc.) and Celtic Haiku-mined books alike, Gemini Tier-2 is
now a standard pass:

```bash
.venv/bin/wyrd kenning lexicon review sources/ \
    --book <id> --provider gemini --include-haiku --apply
```

Idempotent against re-runs. For Sonnet-mined books, use
`--include-sonnet` (PR #69, wyrd-0rsk).

### Resuming / re-mining

- The mining writer uses `INSERT OR IGNORE` on dedupe keys, so
  re-running the same book is **idempotent**. Safe to re-run after a
  parser fix to pick up the additional entries.
- Per-row provider tagging (`extracted_by:ollama:`, `extracted_by:gemini:`,
  `extracted_by:anthropic:` in `source_quote`) lets a later review pass
  see who said what.

### Decline recovery (wyrd-bgr)

Tier-1 declines are entries the model refused to extract from
("no etymology in this body"). Tier-2 review never sees them because
review starts from existing toponym_etymology rows. Across dual-mined
books, a stronger Tier-1 model recovers a substantial fraction of
those declines:

| Book | Qwen extracted | Haiku-also extracted | Haiku-only (Qwen missed) |
|---|---:|---:|---:|
| mawer_1920_northumberland_durham | 366 | 426 | 177 (39% of decline pool) |
| lindkvist_1912_scandinavian_place_names | 39 | 119 | 91 (3× lift over Qwen) |
| skeat_1901_cambridgeshire | 47 | 51 | 10 |
| skeat_1913_suffolk | 42 | 42 | 7 |
| skeat_1906_bedfordshire | 20 | 24 | 6 |

To cheaply re-mine just the gaps (without re-paying for entries the
first model already got), use `--declines-only`:

```bash
# After Qwen Tier-1 leaves declines on Ekwall 1922 Lancashire,
# point Haiku at the residual:
wyrd kenning lexicon mine-llm sources/ekwall_1922_lancashire.txt \
    --provider anthropic \
    --model claude-haiku-4-5-20251001 \
    --declines-only
```

The flag pre-filters parsed entries to those that have NO existing
toponym_etymology row for the source — regardless of which Tier-1
provider produced the existing rows. Filter applies before `--limit`,
so `--limit 50 --declines-only` smokes the first 50 of the residual,
not the first 50 of the parsed list.

#### First production batch (2026-05-05)

Ran Haiku across 10 Qwen-only English books with `--declines-only`.
1169 residual entries → 396 accepted → **314 new etymologies
ingested** (27% recovery at the etymology level — the gap between
accepted and ingested is mostly low-confidence rows the writer drops
per `lexicon.py:1394`):

| Book | Residual | Accepted | Ingested | Accept % |
| --- | ---: | ---: | ---: | ---: |
| roberts_1914_sussex | 356 | 134 | 113 | 38% |
| ekwall_1928_river_names | 222 | 72 | 45 | 32% |
| smith_1928_north_riding_yorkshire | 127 | 43 | 39 | 34% |
| mawer_stenton_1927_worcestershire | 122 | 43 | 37 | 35% |
| ekwall_1922_lancashire | 109 | 44 | 31 | 40% |
| wyld_hirst_1911_lancashire | 84 | 14 | 10 | 17% |
| skeat_1904_hertfordshire | 64 | 16 | 16 | 25% |
| mawer_stenton_1930_sussex | 43 | 15 | 11 | 35% |
| skeat_1911_berkshire | 23 | 8 | 7 | 35% |
| mawer_stenton_1925_buckinghamshire | 19 | 7 | 5 | 37% |
| **Total** | **1169** | **396** | **314** | **34%** |

Notes:
- Reject rates were high on Ekwall 1928 river names (31%) and Skeat
  1911 Berkshire (61%) — both consistent with the
  short-entry / OCR-noise pattern that trips the
  `form_not_in_body` validator. Not a recovery problem; the
  writer correctly drops them.
- Wyld & Hirst 1911 Lancashire underperforms (17% accept) — its
  entries are short and Qwen had already extracted the easy ones.
- The 34% average tracks the smoke projection from the dual-mined
  comparison (Mawer 39%, plus 5 smaller books at 17–30%).

---

## Stage 6 — post-mining cleanup pipeline

Once the LLM passes are done, run the LLM-free chain to consolidate
the data:

```bash
/tmp/post-mining-pipeline.sh
```

That script chains:

1. **`link-lemmas --apply`**: matches inflected forms (`cotan`,
   `denu`, `byrhtas`) to their existing canonical lemma rows
   (`cot`, `den`, `byrht`) and sets `lemma_id` + `inflection`.
   Conservative — never invents a lemma. Per `DECISIONS.md` D8,
   this is what makes the consensus rollup count "cot" + "cotan"
   + "cotum" as one morpheme with witnesses unioned.

2. **`normalize-ocr --apply`**: clusters OCR-mangled variants
   (`Hædan` / `Hcsdan` / `Haedan`) into a single etymon row with
   citations / glosses / tags / reflex-links / `etymon_text_match`
   repointed to the surviving row, and inflected children
   re-parented through the winner's lemma. Per D9, this runs
   *upstream* of lemma-linking in spirit — the linker sees a
   clean form set rather than 4 spurious variants. Per D21, the
   merge sums `etymon_text_match.match_count` on collisions
   rather than dropping the loser's count, so mining evidence
   is preserved across the merge.

3. **`reverse-search ./sources --apply`**: every etymon's canonical
   form is regex-scanned against every source body. Hits land in
   `etymon_text_match` (the parallel evidence table per D12).
   Looser evidence than extraction citations — does NOT count
   toward the consensus view.

4. **`fuzzy-search ./sources --apply`**: gloss-anchored Levenshtein-1
   match for variant spellings. Per D15, an edit-distance hit
   only counts if a gloss string from `etymon_gloss` appears within
   ~100 characters of the matched form in the source body —
   prevents `bere` / `bera` (barley vs bear) coincidences.

5. **`report`**: prints corpus state + per-source citation counts +
   consensus distribution. Read this to see how rando-port coverage
   moved.

The chain is **idempotent**. Re-running it after every mining batch
is the right cadence.

### Iterating on enrichment heuristics

Per D22, the enrichment stages are reversible. To swap in a new
heuristic without re-mining:

```bash
# Reset just the stage you're iterating on
wyrd kenning lexicon clear-enrichment --stage=ocr --apply
# (other choices: --stage=lemmas, --stage=text-match, --stage=all-derived)

# Re-run that stage with the new logic
wyrd kenning lexicon normalize-ocr --apply
```

`--stage=all-derived` is the "fully reset" path — clears OCR merges,
lemma links (and `lemma_method` stamps), and every
`etymon_text_match` row. Then re-running the post-mining chain
rebuilds enrichment from scratch on the unchanged mining evidence.

Each derived row carries a method/version stamp so a future v2
heuristic can selectively rebuild only its predecessor's work:
- `etymon.lemma_method` — which `link-lemmas` version produced this link
- `etymon_text_match.method` — which search heuristic produced this row
  (`reverse-search-v1`, `fuzzy-search-v1`, `llm-disambiguator-v1`, ...)

### Why these run after mining, not during

Mining writes raw inflected forms as it sees them. Lemma linking
needs the lemma to already exist — running the linker mid-mining
would miss inflections of lemmas that haven't been ingested yet.
Running it once at the end gives the linker the full picture.

OCR clustering similarly benefits from the full set of variants;
clustering early would miss later-arriving variants.

---

## Stage 7 — verify

```bash
.venv/bin/wyrd kenning lexicon report --top 25
.venv/bin/wyrd kenning lexicon stats
```

Things to look at:

- **`rando-port` coverage**: how many seed lemmas now have ≥1
  scholarly extraction witness? `report` breaks this down. Target
  is to push more rando-port lemmas into the ≥3-witness "promotion-
  eligible" bucket.
- **New lemmas** (not in rando-port): mining sometimes discovers
  morphemes the Wikipedia-derived seed missed. Sample query in
  `INGESTION.md` § Useful queries below.
- **Per-source row counts**: a book that mined hundreds of entries
  but contributed few `toponym_etymology` rows is producing a lot
  of duplicates of already-known toponyms — check whether the
  toponyms are being deduplicated correctly via the
  `(modern_name, country, region)` unique index.
- **Anomalies in rejection reasons**: if a book has a high rate of
  `bad_language` or `element_count_out_of_range`, the prompt or
  extractor schema may need tuning.

---

## Stage 8 — when ready to ship

Regenerate `data/meanings.json` from the lexicon:

```bash
.venv/bin/wyrd kenning lexicon export-meanings > wyrd/generators/kenning/data/meanings.json
```

(Confirm the command flags from `--help` — there may be filter options
like `--min-witnesses 3` to limit the export to promotion-eligible rows.)

> **Memory: give `export-meanings` headroom (~1.5 GB+).** The export holds the
> corpus, the word DB, and **all ~67K promoted families simultaneously** —
> grouping families into subjects by `(modifier_type, glosses, tags)` signature
> is a full-corpus operation, so every family dict must be live at once (~1 GB
> peak RSS). Run it **with memory headroom and NOT concurrently** with other
> heavy passes (a second export, bulk ingest, enrichment); it has been
> OOM-killed under concurrent load (0-byte output, no file written — wyrd-v3ow.1).
> If a host is memory-constrained, run the export in its own subprocess with an
> RSS ceiling rather than alongside the rest of a deploy. Capping the *peak*
> itself (a two-pass group-keys-first rewrite) is deferred — it trades the
> wyrd-4zyb bulk-attested-years wall-time win and risks the u6fn.4 fidelity
> gate, so it's only worth it if a hard memory ceiling forces it (see wyrd-9dwv).

Then regenerate the per-culture **proportions** — the empirical corpus
distribution (slot structures + tag co-occurrence) that the **vector**
generator samples its skeleton from and draws its cohesion signal from. This
is a *data feed* for vector scoring, NOT a scoring mode of its own: vector is
the only scorer (D36); "proportions" here just names the bundled corpus stats.

```bash
for culture in english scottish welsh irish; do
  .venv/bin/wyrd kenning rebuild-proportions $culture \
    data/seed/${culture}_place_names.json \
    > data/seed/${culture}_proportions.json
done
```

Run the test suite (`pytest tests/`) to catch regressions before
committing the regenerated bundle.

---

## Wiktionary ingestion (wyrd-4rt)

Wiktionary mining is a separate sub-pipeline from the LLM-driven
place-name dictionary mining above. There's no Tier 1 / 2 / 3 model
dispatch and no form-in-body validation — wiktextract has already
parsed the wikitext into structured JSON, so the "extraction" was
done by human Wiktionary editors. Our job is to load the graph.

### Acquire the dump

[Kaikki.org](https://kaikki.org/dictionary/rawdata.html) publishes
wiktextract output as JSONL, in either the full corpus or per-language
splits. The full corpus is multi-GB; per-language slices are 100s of
MB. URLs of the form:

```
https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz       # full
https://kaikki.org/dictionary/<Language>/kaikki.org-dictionary-<Language>.jsonl
```

For place-name etymology coverage, the most valuable slices are
**Old English**, **Old Norse**, **Proto-Germanic**, **Proto-Celtic**,
**Latin**, and **Proto-Indo-European** — each gives a rich Etymology +
Descendants tree connecting modern reflexes back to common roots.

Save under `sources/` as `wiktextract_<slice>.jsonl[.gz]` (the file
isn't checked in — `sources/` is gitignored).

### Ingest

```bash
.venv/bin/wyrd kenning lexicon ingest-wiktionary sources/wiktextract_proto_germanic.jsonl.gz --apply
```

Flags:
- `--limit N` — stop after N entries (smoke test before a full run)
- `--since-line N` — skip the first N lines (resume after interruption)
- `--db PATH` — alternate DB path

The ingester reads one JSON record per line, walks each entry's
Etymology + Descendants sections, upserts the referenced etymons,
and inserts `etymon_descent` edges per the D27 taxonomy. Edge_type
semantics + bridging rules are in the [Etymological descent graph
(D27)](#etymological-descent-graph-d27) subsection below; the
wiktextract-specific bit is just the template-name → edge_type
mapping:

| Source | Direction | edge_type |
|---|---|---|
| `{{inh}}` / `{{inherited}}` template | UP | `inheritance` |
| `{{bor}}` / `{{borrowed}}` template | UP | `borrowing` |
| `{{der}}` / `{{derived}}` template | UP | `derivation` |
| `{{cal}}` / `{{calque}}` template | UP | `calque` |
| descendants tree node (default) | DOWN | `inheritance` |
| descendants node with `tags=['calque']` | DOWN | `calque` |
| descendants node with `tags=['borrowed']` | DOWN | `borrowing` |
| descendants node with `tags=['derived']` | DOWN | `derivation` |
| `{{cog}}` / `{{m}}` / `{{l}}` / qualifier templates | n/a | skipped |
| anything else | n/a | counted as `unsupported_template` |

Etymology section uses `{{template}}` markers; descendants section is
a recursive nested tree of `{lang_code, word, tags?, descendants?}`
nodes (NOT a flat list with `depth` markers — that was a v1 misread,
fixed in wyrd-c9t).

Source attribution is the synthetic `'wiktionary'` source row (per
D27).

### Cluster cognates afterward

The ingest only writes descent edges. To populate `etymon.cognate_id`
so cross-language cognate queries become a single JOIN, run the
cluster pass next:

```bash
.venv/bin/wyrd kenning lexicon cluster-cognates --apply
```

This walks the descent graph from each root and assigns `cognate_id`
to every reachable etymon. Reversible via
`clear-enrichment --stage=cognates --apply`.

### Idempotency + resumability

The ingest is idempotent on `(canonical_form, language)` for
etymons and on `(parent_id, child_id, edge_type, source_id)` for
descent edges (UNIQUE constraints from data/seed/lexicon.sql). Re-running
the same JSONL is safe — duplicate inserts are silently skipped.

If interrupted, note the last `lines_read` from the CLI output and
resume with `--since-line N`.

### What if a template kind isn't recognized?

The CLI reports `unsupported_templates = N` when wiktextract entries
contain etymology templates we haven't mapped. Supported set covers
inh / bor / der / cal plus long-name variants. Extend
`_UPWARD_TEMPLATE_TO_EDGE` or `_SKIPPED_TEMPLATE_NAMES` in
`wyrd/generators/kenning/lexicon/wiktextract_ingester.py` to add new kinds.
Descendants nodes never produce unsupported_templates (they're read
as data, not via templates).

### Anti-patterns

- **Don't run mine-llm against wiktextract dumps**. The LLM extractor
  is for unstructured OCR text (place-name dictionaries). Wiktextract
  is already structured; running an LLM over it would burn tokens
  for zero benefit.
- **Don't add wiktextract entries to `sources/MANIFEST.md`**. The
  manifest is for OCR'd scholarly books that drive the place-name
  pipeline. Wiktextract is a single synthetic source.
- **Don't run `link-lemmas` against wiktextract data**. That stage
  links inflected forms to lemmas; wiktextract entries are already
  lemma-form (or marked as such), and linkage rules tuned for OE
  / ON inflection on toponym morphemes don't apply.

---

## Useful queries

### How is rando-port faring overall?

```sql
WITH rando_lemmas AS (
  SELECT DISTINCT COALESCE(e.lemma_id, e.id) AS lemma_id
  FROM etymon e
  JOIN etymon_citation c ON c.etymon_id = e.id
  WHERE c.source_id = 'rando-port'
),
witness AS (
  SELECT
    rl.lemma_id,
    COUNT(DISTINCT CASE WHEN c.source_id != 'rando-port' THEN c.source_id END) AS scholar_witnesses
  FROM rando_lemmas rl
  LEFT JOIN etymon e ON COALESCE(e.lemma_id, e.id) = rl.lemma_id
  LEFT JOIN etymon_citation c ON c.etymon_id = e.id
  GROUP BY rl.lemma_id
)
SELECT
  CASE
    WHEN scholar_witnesses = 0 THEN '0  (rando-only)'
    WHEN scholar_witnesses = 1 THEN '1  witness'
    WHEN scholar_witnesses = 2 THEN '2  witnesses'
    ELSE                              '3+ witnesses (promotion-eligible)'
  END AS bucket,
  COUNT(*) AS lemmas
FROM witness
GROUP BY 1
ORDER BY MIN(scholar_witnesses);
```

### What lemmas have we discovered that aren't in rando?

```sql
WITH lemma_sources AS (
  SELECT
    COALESCE(e.lemma_id, e.id) AS lemma_id,
    GROUP_CONCAT(DISTINCT c.source_id) AS sources
  FROM etymon e
  JOIN etymon_citation c ON c.etymon_id = e.id
  GROUP BY lemma_id
)
SELECT COUNT(*)
FROM lemma_sources
WHERE sources NOT LIKE '%rando-port%';
```

Run AFTER `link-lemmas --apply` — otherwise inflected forms of
existing rando lemmas show up as spuriously new.

### Per-source toponym + etymology row count

```sql
SELECT
  s.id,
  COUNT(DISTINCT te.toponym_id) AS toponyms,
  COUNT(DISTINCT te.id) AS etymologies,
  COUNT(DISTINCT tee.id) AS elements
FROM source s
LEFT JOIN toponym_etymology te ON te.source_id = s.id
LEFT JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
GROUP BY s.id
ORDER BY toponyms DESC;
```

### Per-source × provider mining-run audit (D23)

The `mining_run` table records every Tier 1 / Tier 2 run with
accept / decline / reject counts (the numbers the mining flow used
to lose to stdout). Per-source × per-provider summary:

```sql
SELECT
  source_id,
  provider || ' / ' || model AS tier,
  mode,
  SUM(parsed_count) AS parsed,
  SUM(accepted) AS accepted,
  SUM(declined) AS declined,
  SUM(rejected) AS rejected,
  COUNT(*) AS runs
FROM mining_run
GROUP BY source_id, provider, model, mode
ORDER BY source_id, mode, provider;
```

For a quick "what's the worst-yield book on Qwen?" probe:

```sql
SELECT source_id,
       SUM(accepted) * 1.0 / NULLIF(SUM(parsed_count), 0) AS accept_rate,
       SUM(parsed_count) AS parsed
FROM mining_run
WHERE provider = 'ollama' AND mode = 'mine'
GROUP BY source_id
HAVING parsed >= 50
ORDER BY accept_rate ASC
LIMIT 10;
```

A row in this output below ~30% accept rate is a candidate for the
Haiku Tier 1 swap (see Stage 4 — the Watson 1904 empirical exception).

### Back-filling historical mining runs

Mining runs that pre-date D23 (i.e. before the writer existed) can
be folded in via `wyrd kenning lexicon import-mining-log <path>`,
which takes a JSONL file with one record per run. Recovery typically
comes from session transcripts (see the `wyrd-ej4` ticket and PR
description for the agent prompt that extracts them). The CLI is
idempotent on `(source_id, provider, model, mode, completed_at)` so
re-running is safe.

### Etymological descent graph (D27)

Once Wiktionary mining (wyrd-4rt) populates `etymon_descent`, these
queries answer the cross-language reflex / cognate questions:

```sql
-- Modern descendants of a Proto-Germanic root (walks DOWN)
WITH RECURSIVE descendants(id) AS (
  SELECT child_id FROM etymon_descent
    WHERE parent_id = (
      SELECT id FROM etymon WHERE canonical_form = '*tūnaz' AND language = 'proto-germanic'
    )
    AND edge_type IN ('inheritance', 'borrowing')
  UNION
  SELECT d.child_id FROM etymon_descent d
    JOIN descendants ON d.parent_id = descendants.id
    WHERE d.edge_type IN ('inheritance', 'borrowing')
)
SELECT e.canonical_form, e.language
FROM descendants
JOIN etymon e ON e.id = descendants.id
ORDER BY e.language, e.canonical_form;
```

```sql
-- Cross-language cognates of a modern word (after wyrd-81n
-- cluster-cognates populates cognate_id)
SELECT e.canonical_form, e.language
FROM etymon e
WHERE e.cognate_id = (
        SELECT cognate_id FROM etymon WHERE canonical_form = 'town' AND language = 'modern-english'
      )
  AND e.cognate_id IS NOT NULL;
```

```sql
-- All etymons known to descend from at least one chain (have parents)
SELECT DISTINCT e.canonical_form, e.language
FROM etymon e
JOIN etymon_descent d ON d.child_id = e.id
ORDER BY e.language, e.canonical_form;
```

Edge types and their semantics:
- `inheritance` (`{{inh}}`) — direct lineage, bridges cognate clustering.
- `borrowing` (`{{bor}}`) — borrowed across languages, also bridges.
- `cognate` (`{{cog}}`) — peer relation; does NOT bridge (would over-unify).
- `derivation` (`{{der}}`), `calque` (`{{cal}}`), `compound` (`{{compound}}` / `{{affix}}`) — context-specific.
- `unknown` — fallback for free-text "compare with X" assertions.

---

## Anti-patterns

- **Don't commit `sources/`** — it's 1.8G+. Gitignored except for
  MANIFEST.md.
- **Don't put a lexicon DB inside the repo.** The default lives at
  `~/.wyrd/lexicon.db`; the legacy in-repo path is auto-migrated on
  first CLI invocation. Use `WYRD_LEXICON_DB` for per-run overrides.
- **Don't write `SESSION_HANDOFF.md` style transient docs into the
  repo** — use `bd remember` or `/tmp` for session continuity.
  Long-term architectural rationale goes in `DECISIONS.md`.
- **Don't promote Tier 1 to Sonnet for mining** (`DECISIONS.md` D19).
- **Don't loosen the form-in-body validator** without explicit thought
  (`DECISIONS.md` D3). It's the load-bearing safety guard.
- **Don't skip `link-lemmas`** between mining and search — the
  consensus rollup will under-count, and "new lemmas" queries will
  return spuriously high numbers.
- **Don't run `Skill: pr-review-loop` as a background Task** — see
  the skill manifest. It must run in the main conversation so it
  can spawn agent reviewer Tasks.

---

## When the parser doesn't yield

If a book yields 0 or near-0 entries, the diagnostic order is:

1. Run `find_body_bounds` directly:
   ```python
   from wyrd.generators.kenning.parsers.dictionary import find_body_bounds
   lines = open('sources/<id>.txt').read().split('\n')
   s, e = find_body_bounds(lines)
   print(f'body bounds: {s}..{e}  ({e - s} lines, {100*(e-s)//len(lines)}%)')
   ```
   If this is < 5% on a > 1000-line doc, the safety net should
   already be kicking in. If it's NOT and yields are still 0, the
   book is genuinely structurally different (Joyce 1875).

2. Check `_ENTRY_BODY_SIGNALS` against the body text:
   ```python
   from wyrd.generators.kenning.parsers.dictionary import _ENTRY_BODY_SIGNALS
   import re
   text = open('sources/<id>.txt').read()
   sample = re.split(r'\n\s*\n', text)[100:120]   # mid-document paragraphs
   for p in sample:
       print('signal:', bool(_ENTRY_BODY_SIGNALS.search(p)), '|', p[:80])
   ```

3. Check headword detection at paragraph starts:
   ```python
   from wyrd.generators.kenning.parsers.dictionary import _ENTRY_HEADWORD, _is_real_headword
   import re
   for p in re.split(r'\n\s*\n', text)[100:120]:
       m = _ENTRY_HEADWORD.match(p)
       if m and _is_real_headword(m.group('name')):
           print('HW:', m.group('name'), '|', p[:80])
   ```

If body-signals and headwords are firing but the parser still emits
nothing, look at `parse_alphabetical_text`'s flush logic — likely the
`_entry_body_is_real` filter is rejecting all bodies.

If a fix is needed and it's substantial, file a beads ticket
referencing the affected books and the suspected cause. Examples:
`wyrd-3qq` (Celtic body-signal regex) and the body-bounds bug were
both surfaced by this exact diagnostic loop.

---

## When fuzzy-search produces sketchy hits

`fuzzy-search --apply` requires a gloss anchor (D15) to count an
edit-distance match — `bere` ↔ `bera` (barley vs bear) doesn't fire
without a gloss within ~100 characters. That gates a lot, but **not
all** generic gloss words filter cleanly. Symptoms:

- A canonical etymon's `matched_form` rows in `etymon_text_match`
  are dominated by a string that is **itself a different canonical
  etymon**. Example seen in this corpus: ON `herað` ("district /
  county") was being claimed as the canonical for body-text
  occurrences of `heath` (OE `hǣþ`, "untilled land"). They are
  unrelated etymons; `heath` already exists as its own canonical
  row. Root cause: D15's gloss anchor accepted generic gloss words
  ("land", "area", "district") that co-occur with many topographic
  words.

- Order-of-magnitude: this shape produced ~47 false positives across
  5 sources before the audit caught it.

Diagnostic:

```sql
SELECT m.etymon_id, e.canonical_form, m.matched_form, m.source_id,
       m.match_count
FROM etymon_text_match m
JOIN etymon e ON e.id = m.etymon_id
JOIN etymon e2 ON LOWER(e2.canonical_form) = LOWER(m.matched_form)
WHERE e2.id != e.id
ORDER BY m.match_count DESC
LIMIT 50;
```

A row in this output is one where the matched_form **is itself a
canonical etymon** for a different row — almost always a false
positive that should belong to the other etymon.

Cleanup:

```sql
DELETE FROM etymon_text_match
WHERE id IN (
  SELECT m.id
  FROM etymon_text_match m
  JOIN etymon e2 ON LOWER(e2.canonical_form) = LOWER(m.matched_form)
  WHERE m.etymon_id != e2.id
);
```

Long-term fix tracked in `wyrd-c3x` (gate-tightening: refuse the
match if any *other* canonical etymon is closer to the matched form
than the claimed one). For ambiguous cases, `wyrd-6z7` proposes
escalating to a Tier 2 LLM disambiguator with the surrounding
snippet and candidate set.

---

## Mining the wyrd-ami fantasy-name corpus

The wyrd-ami pipeline (see OVERVIEW.md "Sibling pipeline") researches
fantasy / gaming creature names against the etymon corpus, producing
`fantasy_morpheme` rows with `usable=1` + `etymon_id` for resolvable
inputs and `usable=0` + `bar_reason` otherwise. The canonical input is
the Pathfinder 2 SRD bestiary stored at `~/521Studios/pfsrd2-data`.

### Happy-path checklist

```bash
# 1. Extract the pfsrd2 corpus to JSONL (no LLM, no DB write).
.venv/bin/wyrd kenning lexicon extract-pfsrd2-monsters \
    /home/$USER/521Studios/pfsrd2-data \
    --output /tmp/pfsrd2-monsters.jsonl

# 2. Mine. --skip-resolved skips inputs that already have a row for
#    the current approach_version (each row costs one Gemini call).
.venv/bin/wyrd kenning lexicon mine-fantasy-name \
    --batch /tmp/pfsrd2-monsters.jsonl \
    --skip-resolved \
    --apply

# 3. Inspect the bar-reason breakdown.
sqlite3 ~/.wyrd/lexicon.db "
  SELECT bar_reason, COUNT(*) FROM fantasy_morpheme
  WHERE approach_version = 'fantasy-v1'
  GROUP BY bar_reason ORDER BY COUNT(*) DESC;"
```

### What the extractor emits

For each monster file:

- **Family root** (single-word, deduped across all family-mates) is
  always emitted when the family field is present and single-word.
  "Bugbear", "Genie", "Demon", "Devil" all come out once apiece even
  though pfsrd2 has dozens of variants of each.
- **The monster's own name** is emitted as a separate record when
  it's single-word AND differs from the family root. "Djinni",
  "Efreeti", "Marid", "Shaitan", "Janni" all under family "Genie" each
  emit their own record — they're etymologically distinct Arabic
  morphemes that the family root alone wouldn't capture.
- Multi-word names ("Bugbear Thug", "Ancient Black Dragon") are
  dropped — no clean single-token morpheme.

Empirical 2026-05-06: 3,670 monster files → 1,308 distinct morpheme
records (the family-dedupe collapsing being the main reason 1,308 is
much smaller than 3,670).

### Bar reasons (decoder)

| Bar reason | Meaning | Fixable how |
|---|---|---|
| `modern_coinage` | Game-designer invented (Vrock, Werebear, Picture-in-Cloud); no historical etymon exists | Not fixable — these names have no real history |
| `outside_language_family` | LLM identified a real etymon in a language NOT in `APPROVED_LANGUAGES` | Approve the language (carefully — `APPROVED_LANGUAGES` is the gate that keeps fantasy register coherent) |
| `attested_but_not_in_corpus` | LLM identified a real etymon in an approved language, but the etymon row doesn't exist in our `etymon` table | Backfill etymon corpus — see epic `wyrd-ialp` |
| `uncertain_attestation` | LLM returned low-confidence; conservative-default barred | Investigate manually or re-run with a stronger model |
| `no_etymology_found` | LLM said "I don't know" with high confidence | Probably a real modern coinage we can't classify |
| `homograph_collision` | LLM flagged that the input collides with multiple unrelated meanings | Manual disambiguation |
| `proper_noun_only` | LLM flagged this is a proper noun, not a common-noun morpheme | Edge case — generally correct to skip |

### Approved languages

`extractors.fantasy.APPROVED_LANGUAGES` is the gate that decides whether
a real etymon counts as "in family" for fantasy generation. The set
covers:

- Core Indo-European stack (proto-indo-european, proto-germanic,
  ancient-greek, latin, old-english, middle-english, modern-english,
  old-norse, old-french, etc.)
- Celtic family (welsh, scottish-gaelic, old-irish, middle-irish,
  breton, cornish, manx, irish)
- World-religion source material via ISO codes (`he` Hebrew, `ar`
  Arabic, `fa` Persian, `sa` Sanskrit, `akk` Akkadian, `egy` Egyptian,
  `arc` Aramaic, `pal` Pahlavi/Middle Persian) — wave-2 addition,
  2026-05-06.
- Wave-2 precursor / postcursor codes — user-approved 2026-05-06
  to enable full-history descent walks back from a modern reflex to
  its deepest ancestor without bailing at the language gate:
  - Semitic stack: `hbo` Biblical Hebrew, `sem-pro` Proto-Semitic,
    `sem-wes-pro` Proto-West-Semitic, `afa-pro` Proto-Afroasiatic,
    `sux` Sumerian (Akkadian substrate), `syc` Classical Syriac
    (Aramaic descendant).
  - Iranian stack: `peo` Old Persian, `fa-cls` Classical Persian,
    `xpr` Parthian, `ira-pro` Proto-Iranian.
  - Indo-Aryan stack: `iir-pro` Proto-Indo-Iranian (root of
    Iranian + Indo-Aryan branches), `inc-pro` Proto-Indo-Aryan,
    `pra` Prakrit, `pi` Pali.
  - Egyptian stack: `cop` Coptic (late-ancient Egyptian descendant).
  - Armenian: `axm` Old / Classical Armenian (intersects Iranian +
    Greek descent paths).

The `_LANGUAGE_ALIAS_MAP` normalizes LLM-returned descriptive names
(`sanskrit`, `arabic`, `ancient-egyptian`, `old-persian`, `coptic`,
`pali`, `proto-semitic`, etc.) to the ISO codes the etymon table
uses, so the LLM doesn't need to know our internal code conventions.

What's NOT approved (intentionally — these would require their own
mood/aesthetic register decisions): Japanese, Chinese, Korean,
Nahuatl, Finnish, Basque, modern Romance (French, Spanish, Italian,
Portuguese, Romanian — tickets filed for historical-chain mapping
first), modern Norse (Swedish, Danish, Norwegian, Faroese — same).

### Resuming a partial run

The mine commits per-resolution, so a crash partway through doesn't
lose already-paid-for results. To resume after a crash, just re-run
with `--skip-resolved` — already-processed inputs are skipped at the
filter (case-insensitive match against the COLLATE NOCASE column).

To force re-processing of a resolved row (e.g. after an LLM-prompt
change or new language approval), bump `extractors.fantasy.APPROACH_VERSION`
and re-run; the (input_name, approach_version) UNIQUE means rows
under the new version write fresh entries, and `--skip-resolved`
reads its skip-set scoped to the *current* version only.

### Empirical baseline (2026-05-06)

After PR #82 (timeout fix + case-normalize), PR #83 (wave-2 languages
+ alias map), PR #86 (extractor v2 family+variant emission), and
PR #87 (--skip-resolved flag), the v2 mine of the full 1,308-record
corpus yielded:

- **218 usable** (16.7%) — production-ready morphemes for town-name
  generation
- 639 modern_coinage (48.9%) — Pathfinder-invented; not fixable
- 210 attested_but_not_in_corpus (16.1%) — fixable via wyrd-ialp
  epic; expect this bucket to mostly flip to usable once
  Hebrew / Arabic / Egyptian / Aramaic / Akkadian / Pahlavi
  wiktextract slices are ingested
- 180 outside_language_family (13.8%) — Korean / Japanese / Tagalog
  etc., correctly barred
- 22 no_etymology_found, 21 homograph_collision, 17
  uncertain_attestation, 1 proper_noun_only

The Pathfinder bestiary's high modern_coinage rate (~half) is a
floor we can't reduce — game designers invented the names — but the
attested_but_not_in_corpus bucket is real upside for the next pass.

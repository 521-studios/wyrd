# docstring-grep-verify-reviewer

Module and function docstrings that name specific tables, columns,
function names, regex patterns, or output shapes must be
grep-verified against the code before merge. The wyrd-67fv slice D
review loop spent 5 fix rounds chasing 7 docstring inaccuracies
(round 2 introduced a NEW inaccuracy while fixing round 1's
fantasy_export claim — fixes can be wrong too) because NEW module
docstrings were written without checking each claim against the
function body. Examples included:

* `ingest.py` claimed "Idempotent per `(source_id, page)`" — code
  never references `page` and there's no UNIQUE constraint that
  would make it idempotent.
* `decomposition_export.py` claimed "walks every toponym +
  toponym_etymology pairing" — SQL actually joins
  `toponym_decomposition` and never touches `toponym_etymology`.
* `fantasy_export.py` claimed source was "etymon tagged
  fantasy-suitable" with output shape `{morpheme, language, gloss,
  source_book, source_page}` — actual source is `fantasy_morpheme`
  rows where `usable=1`, with a completely different output shape.
* `bundle/_emit.py` claimed `_LANG_CODE_TO_JSON_FIELD` rolls
  `welsh + old-welsh + middle-welsh → welsh` — actually they all
  collapse into `celtic_mix` (a single shared bucket).
* `bundle/_subject.py` `_emit_word_languages` and
  `_WordLanguageAccumulators` docstrings listed 5 sibling field
  families — the actual code emits 9 (wyrd-vsrn / wyrd-qhs0 /
  wyrd-lr4 grew the set without updating prose).

**FLAG when a NEW or MODIFIED docstring contains:**

1. A SQL table or column name that doesn't appear in the function's
   SQL string literal.
2. A function name referenced as called/used that doesn't appear in
   the function body (grep for the bare name).
3. A regex count (e.g. "Three regex patterns", "Five precision
   gates") that doesn't match `len(re.findall(r'^_[A-Z_]+_RE\b', ...))`
   or the equivalent obvious count.
4. An output-shape claim (`{key1, key2, ...}` or
   `dict[K, V]`) where the keys don't match the actual `SELECT`
   columns or dict construction in the function body.
5. A field count or list ("5 dicts", "the four renderings", "the
   nine sibling families") that doesn't match the actual dataclass
   field count or list construction.

**Acceptable** (don't flag):

* Stable conceptual descriptions that don't name specific symbols
  ("this module owns the per-language phonological-bridge pass").
* Examples that use placeholder names (`<lang>_variants`).
* Forward-looking docs that explicitly say "Phase 2 will add ...".

**Review approach:**

1. For each NEW module docstring (file added in the PR), open the
   first function below it and grep-verify every named symbol /
   table / regex count / output key.
2. For MODIFIED module docstrings, compare the diff's new lines
   against the function body line-by-line.
3. For dataclass-bundle docstrings (`_BucketAccumulator`,
   `_WordLanguageAccumulators`, etc.), count the actual
   `field(default_factory=...)` declarations and compare to any
   count words in the prose.
4. If a count or shape claim is wrong: flag it. These are
   factual errors that mislead future maintainers (and they
   accumulate — slice D's `_WordLanguageAccumulators` doc was
   wrong because three separate wyrd-ticket waves grew the set
   without anyone re-reading the class doc).


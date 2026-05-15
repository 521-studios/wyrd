# L2 JSONL `jq` recipes (wyrd-l2d2)

One-liners for common queries over `data/mining/*.jsonl`. Every recipe
assumes the canonical-state row shape emitted by `lexicon dump-jsonl`
(no `_op` field; one `source` row + N typed list/keyed rows).

The `_type` field is the primary dispatch key. See `wyrd/generators/kenning/jsonl_log.py`
(KEYED_TYPES + LIST_TYPES) for the full vocabulary.

## List every etymon cited by a source

```bash
jq -r 'select(._type == "etymon") | .ref' data/mining/mawer_1920_northumberland_durham.jsonl
```

## Find etymons whose glosses match a pattern

```bash
jq -c 'select(._type == "etymon" and (.glosses // [] | join(" ") | test("river|stream"; "i")))' data/mining/welsh_*.jsonl
```

`test()` uses PCRE-style regex with `"i"` for case-insensitive.

## Count rows per `_type` per source

```bash
for f in data/mining/*.jsonl; do
  printf "%-50s  " "$(basename "$f" .jsonl)"
  jq -r '._type' "$f" | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}'
  echo
done
```

## Diff two source files by ref-set

```bash
diff \
  <(jq -r '._type + ":" + (.ref // .etymon_ref // .toponym_ref // "_unknown_")' data/mining/source_a.jsonl | sort) \
  <(jq -r '._type + ":" + (.ref // .etymon_ref // .toponym_ref // "_unknown_")' data/mining/source_b.jsonl | sort)
```

The `_type:` prefix already distinguishes row types, so no need to
suffix the etymon_ref / toponym_ref differently from `.ref`. Falls
through to `.toponym_ref` for attestation and etymology_element rows
that don't carry a top-level `.ref`.

## Show every row touching a specific etymon ref (replay-style walk)

```bash
REF='old-english:ford'
jq -c --arg ref "$REF" '
  select(
    (._type == "etymon" and .ref == $ref) or
    (._type == "citation" and .etymon_ref == $ref) or
    (._type == "etymon_descent" and (.parent_ref == $ref or .child_ref == $ref)) or
    (._type == "etymology_element" and ((.elements // []) | any(.etymon_ref == $ref)))
  )
' data/mining/*.jsonl
```

## Count attestations per source

```bash
jq -r 'select(._type == "attestation") | input_filename' data/mining/*.jsonl \
  | sort | uniq -c | sort -rn | head -10
```

Single-process `jq` pass using `input_filename` — counts and groups
attestations in one walk rather than spawning a `jq + wc` pair per
file. On large dump dirs this is ~10× faster.

## Find attestations within a year range

```bash
jq -c 'select(._type == "attestation" and .date_year >= 1000 and .date_year <= 1100)' \
  data/mining/open_domesday_hull.jsonl | head
```

## Find toponyms (any source) that have an etymology in source X but no attestation

```bash
# A toponym with an etymology emit but no attestation under the same ref.
# Useful for finding scholar-attributed names that the period-form pipeline
# can't anchor in time (no Tier-3 forward-projection candidate).
SOURCE=mawer_1920_northumberland_durham
jq -r '
  select(._type == "toponym") | .ref
' data/mining/${SOURCE}.jsonl | sort -u > /tmp/${SOURCE}.toponyms.txt
jq -r '
  select(._type == "attestation") | .toponym_ref
' data/mining/${SOURCE}.jsonl | sort -u > /tmp/${SOURCE}.attested.txt
comm -23 /tmp/${SOURCE}.toponyms.txt /tmp/${SOURCE}.attested.txt | head
```

## Validate that every `etymon_ref` in citations resolves to an emitted etymon

```bash
SRC=data/mining/mawer_1920_northumberland_durham.jsonl
jq -r 'select(._type == "etymon") | .ref' "$SRC" | sort -u > /tmp/emitted.txt
jq -r 'select(._type == "citation") | .etymon_ref' "$SRC" | sort -u > /tmp/cited.txt
comm -13 /tmp/emitted.txt /tmp/cited.txt
# Output: refs cited but never emitted. Should be empty in a healthy dump.
```

## Tips

- Use `jq -c` (compact) for piping; `jq -r` for raw strings; default
  pretty-prints with whitespace.
- The dump output is one JSON object per line. Don't try to `jq .` an
  unwrapped file — use `jq -c .` or a `select(...)` filter.
- For large queries, `jq --stream` reads incrementally. Helpful on
  `open_domesday_hull.jsonl` (37K rows).
- Replay semantics (event sequencing, `_op: patch`, etc.) only apply
  to event-log files. Per-source dumps are canonical-state — no `_op`
  fields, so a simple `select()` walk is the right shape.

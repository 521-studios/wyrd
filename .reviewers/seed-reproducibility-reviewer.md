# seed-reproducibility-reviewer

Wyrd's contract is that the same `(generator, params, seed)` tuple always yields the same output. Any randomness must flow through `wyrd.seed.rng_for(seed)` so seeds are reproducible. **Default severity: P1** — this contract is foundational to the product; a generator that doesn't honor it silently produces different outputs on different machines / runs.

**Patterns to FLAG:**

1. **Direct use of the global `random` module:**

   ```python
   # BAD — not reproducible
   import random
   random.choice(words)
   random.random()
   random.randrange(2**63)
   ```

   These pull from a process-wide RNG with no relationship to the request's seed. Two calls to the same `(generator, params, seed)` tuple will produce different outputs.

   ```python
   # GOOD — every random choice is bound to the request seed
   from wyrd.seed import rng_for

   rng = rng_for(seed)
   choice = rng.choice(words)
   ```

2. **`secrets` / `os.urandom` outside `resolve_seed`:**

   ```python
   # BAD — non-deterministic, can't reproduce from a seed
   secrets.randbits(64)
   os.urandom(16)
   ```

   Acceptable **only** in `wyrd/seed.py:resolve_seed()` (the one place that's allowed to mint a seed) and tests that exercise `resolve_seed` itself. Anywhere else, even when "the seed isn't user-visible," the contract is broken — operators can't reproduce, debug, or share results.

3. **Multiple results without sub-seed derivation:**

   When a generator (or the dispatcher loop) produces multiple results, sub-seeds must be derived deterministically from the parent seed via a `random.Random` instance — see `wyrd/app.py:_dispatch()` for the canonical pattern.

   ```python
   # BAD — every result uses the same seed, so they're identical
   results = [generator.generate(params, seed) for _ in range(count)]

   # BAD — sub-seeds are non-deterministic
   results = [generator.generate(params, secrets.randbits(64)) for _ in range(count)]

   # GOOD — sub-seeds derived deterministically from the parent seed
   sub_rng = random.Random(seed)
   results = [
       generator.generate(params, sub_rng.randrange(2**63))
       for _ in range(count)
   ]
   ```

**Acceptable patterns:**

- `rng = rng_for(seed); rng.choice(...)` everywhere randomness is consumed.
- A `random.Random` instance threaded through the call stack as a parameter.
- `secrets.randbits` in `resolve_seed()` only.

**Do NOT flag:**

- Tests for `resolve_seed` itself that necessarily call `secrets` / `os.urandom`.
- Use of `random.Random(seed)` as the explicit deterministic source — this is the underlying primitive `rng_for` returns.

**Review approach:**

1. Grep for `import random`, `random.`, `secrets.`, `os.urandom` in `wyrd/`.
2. For each hit, confirm it routes through `rng_for(seed)` or is in the explicitly-allowed location (`wyrd/seed.py:resolve_seed`).
3. For loops that produce N results, confirm sub-seeds are derived from the parent seed deterministically.
4. For new `generate()` methods, cross-reference with `test-coverage-reviewer`'s deterministic-output regression test rule.


# Superseded kenning decisions (archive)

Historical decision bodies that have been **superseded** and moved out of the
active `DECISIONS.md` so they don't load into the read-all-kenning-docs working
set. See `DEPRECATED.md` (same folder) for the ledger of what moved, when, and
what replaced it. Each entry keeps its original `DECISIONS.md` heading so
cross-references still resolve by search.

These are **not current** — read the named replacement decision instead. They
are preserved only for archaeology (why a now-dead approach existed).

---

## D17. Bayesian mixture is the novelty knob, not just better fit.

> **SUPERSEDED 2026-05-18 by D36 (vector-driven generator architecture).** The
> proportions sampler this targeted is retired. Live residue: `--cohesion` is
> now a multiplicative boost inside the vector scorer; `--novelty` was removed
> and re-wired onto the vector path (wyrd-fcub). Read D36 for the current model.

Once the generator wires tag-level co-occurrence in, it samples from a
mixture: α·empirical-pair-frequency + β·tag-class-prior + γ·marginal.
The `--novelty` knob (or equivalent) shifts weight toward γ to allow
plausible-but-unattested combinations. The Braitham Gate regression
test pins this: the generator must still be capable of producing names
of that shape (descriptive+topography compound first-word) under
default settings.

Why: pure empirical conditioning produces a generator that can only
remix existing combinations. Pure marginal produces noise. The knob
gives the GM a continuum, with "do not regress on Braitham Gate" as the
fixed point on the novel side.

**Runtime status (wyrd-gfa, 2026-05-02; β-term wyrd-mj2, 2026-05-04):
shipped on the proportions sampler, then partly retired.** The original
implementation blended each bucket's empirical-frequency distribution
with a uniform marginal (`(1-novelty)·empirical + novelty·uniform`),
exposed as the `--novelty` CLI knob (0..1). The `β·tag-class-prior` term
landed via the `--cohesion` knob (wyrd-mj2): the structure walk threaded
`prior_tags`, computed a per-key class-conditional likelihood from the
tag co-occurrence model, and applied it as a multiplicative boost to the
empirical weights before the novelty blend (`weights = empirical *
boost; result = (1-novelty)·weights + novelty·uniform`) rather than the
strict additive `α·empirical + β·tag-class-prior + γ·marginal` of the
textbook formulation. wyrd-9gt closed as superseded by this realized
form. When proportions scoring was retired, the **`--novelty`** half was
removed with it (re-wired onto the vector path, wyrd-fcub); the
**`--cohesion`** half survives, now applied as a multiplicative boost
inside the vector scorer, so a GM can still dial attested-pair fidelity.

---

## D17 refinement: cohesion knob (wyrd-mj2, PR #59).

> **SUPERSEDED 2026-05-18 by D36 (vector-driven generator architecture).** The
> implementation below targeted the proportions sampler; the math/rationale
> carry over to the vector scorer's cohesion multiplier (read D36), but the
> named proportions-era helpers (`Generator.select`, `_cohesion_boost`,
> `_raw_class_score`) are gone.

> **Status:** the implementation below targeted the proportions
> sampler. Proportions scoring is now retired: `--novelty` was removed
> (re-wired onto the vector path, wyrd-fcub) and `--cohesion` was
> re-wired as a multiplicative boost inside the vector scorer. The math
> and rationale carry over; the named proportions-era helpers
> (`Generator.select`, `_cohesion_boost`, `_raw_class_score`) are gone.

D17 originally specified the novelty knob: blend each morpheme
bucket's empirical-frequency distribution with a uniform marginal,
allowing plausible-but-unattested combinations. wyrd-mj2 adds the
companion **cohesion** knob, which biases the OPPOSITE direction —
toward attested tag-class pairings.

As the structure walk fills slots, the union of previously-picked
usages' tags becomes the prior-context set. Each subsequent slot's
bucket gets a per-key multiplier:

```
raw_score(usage)  = Σ over (ta in prior, tb in usage.tags)
                       of tag_cooccurrence[ta|tb] / tag_marginal[ta]
multiplier(usage) = (1 - cohesion) + cohesion * (raw_score / mean_raw_in_bucket)
```

Mean-normalization preserves total mass — at cohesion=1 average-
likelihood candidates get ~1×, above-average >1×, below-average <1×.

In the proportions sampler it composed orthogonally with novelty
(uniform-marginal blend, applied LAST) and harshness (D6 phonological
re-weight). GMs could dial 'attested-pair fidelity' (cohesion) and
'novelty' independently. No-op when the bundle carries no
`tag_cooccurrence` data (such bundles rode the bit-stable path even at
cohesion=1).

Bit-stable at cohesion=0 — the boost short-circuited to None and the
sampler took its harshness=0/novelty=0/key_boost=None fast path.

Bit-stability across PYTHONHASHSEED: the raw-class-score computation
sorted `prior_tags` and `candidate_tags` before iteration. Without the
sort, set-iteration order varies across processes and float-
summation accumulates ULP-level different scores that could flip
weighted_choice outcomes at boundaries.

# Register-Effect Catalog: Scholarly Grounding

**Status:** v1 grounded. Filed under wyrd-2166 to convert the v1 catalog
from hand-tuned defaults (wyrd-kq7w.2 / PR #235) into a citation-bearing
artifact. Per-register weights below are now annotated with the
phonaesthetics / sound-symbolism literature that supports (or fails to
support) each weight.

**TL;DR for catalog readers:**

1. **`stop_vs_continuant` is the load-bearing dimension** across every
   serious sound-symbolism study reviewed. If you only tune one dim,
   tune this one. Plosives → unpleasant + active universally; sonorants
   + continuants → pleasant + passive universally.
2. **`vowel_backness` was under-used in v1.** Ohala's frequency code
   (1994) makes it the primary acoustic carrier of size/dominance
   inference; +back = large/dominant, −back = small/ethereal. v1 had
   weights of 0.0 on this dimension across `noble`, `mystical`,
   `sinister`; v1.1 adds them with citations.
3. **`palatalization` is cross-culturally *diminutive*, not magical.**
   Kochetov & Alderete (2011) document palatalization as the
   universal "small/cute/childish" marker across Japanese, Basque,
   Russian, etc. v1 placed it in `mystical` — defensible only via
   Ohala's frequency-code chain (palatal = high-frequency = small =
   ethereal). v1.1 keeps the weight but documents the chain.
4. **`cluster_density`, `final_fortition`, `final_cluster_rate` are
   Germanic-stereotype features**, not perceptual primitives. They
   carry the "harsh" feel for English-speaking operators (Mooshammer
   et al. 2024 on Orkish confirms English-speaker perception) but
   bouba-kiki literature gives them zero direct support. Tag as
   `IE-conventional`, not `universal`.
5. **`soft_consonants` is internally inconsistent.** Whissell (2000)
   shows /l, m, n/ are firmly Gentle but /r/ is firmly Harsh. Treating
   them as one bucket means /r/-heavy lemmas score high on
   "softness" when they should score the opposite. Filed for a
   schema-level refinement (see Follow-ups below).
6. **The `exotic` register is an identity-marker, not a sound-symbolic
   primitive.** The features it weights (palatalization, retroflexion,
   pharyngeal, back vowels) carry no documented cross-cultural sound-
   symbolic payload — they don't read "spiky" or "round" in the
   literature. They DO read "phonologically distinct from the
   operator's default English inventory," which is the register's
   intent: mark a phonological color outside the English default for
   the British-Isles-historical default operator. Keep the register;
   the v1 docstring is accurate as a functional description. The
   citation gap is that the weights aren't doing crossmodal-
   perception work — they're doing phonological-identity-marking work.
7. **The `ancient` register has no phonosymbolic literature backing.**
   The "polysyllabic + vowel-final + pharyngeal" weights are pure
   Latinate-prestige convention. Keep as IE-conventional.

---

## 1. The 14 phonological dimensions: grounding status

The 14 dimensions enumerated in `PhonologicalFeatureName` (vector_schemas.py)
fall into three groups by literature backing:

### 1a. Universal sound-symbolic primitives

These have cross-linguistic replication (Blasi et al. 2016, PNAS, 6,452
word lists across ~62% of world's languages, FDR-corrected) AND multi-
study experimental validation (Fort/Martin/Peperkamp 2015; Sidhu &
Pexman 2024; Ćwiek et al. 2022 across 25 languages, 89% canonical-
phoneme congruence).

| Dimension | Universal mapping | Citation |
|---|---|---|
| `stop_vs_continuant` (+stops) | Sharp/threatening/active universally | Fort et al. 2015; Sidhu & Pexman 2024; Blasi 2016 ("bite"→/k/, "horn"→/k,r/); Whissell 2000 |
| `vowel_height` (+high) | Small/light/ethereal (high vowels = high F2/F0 = small vocalizer) | Ohala 1994 p. 335; Sidhu & Pexman 2018 review |
| `vowel_backness` (+back) | Large/dominant (back vowels = low F2 = large vocalizer) | Ohala 1994 p. 335; Knoeferle et al. 2017 (F2 as primary acoustic correlate); Auracher et al. 2010 |
| `palatalization` | Small/diminutive/childish | Kochetov & Alderete 2011; Blasi 2016; Ohala 1994 p. 335 (palatal = sharp/small) |
| `sibilance` | Active/arousing (mixed pleasant/unpleasant per phoneme: /z/ pleasant, /ʃ/ unpleasant) | Whissell 2000; Aryani et al. 2018; Sidhu & Pexman 2018 |
| `aspirated_voiceless` (+1) | Active/sharp/explosive | Whissell 2000 ("minor explosion of breath"); Ohala 1994 voiceless > voiced for high-frequency code |

### 1b. IE-conventional features

These have NO direct cross-cultural sound-symbolic backing but do
correlate with English-speaker register perceptions. They are
operator-side associations — useful for British-Isles-historical naming
but should NOT be treated as perceptual primitives.

| Dimension | Convention | Source |
|---|---|---|
| `cluster_density` | "Harsh" in Germanic/Slavic genre tropes | Mooshammer et al. 2024 (Orkish); Crystal 1995 |
| `final_fortition` | "Harsh" in same tradition | Same |
| `final_cluster_rate` | "Harsh" in same tradition | Same |
| `vowel_final_bias` | "Soft" via open-syllable preference; supports CV-rhythm legality | Styles & Gawne 2017 (failures track phonotactic illegality, not absence of mapping) |
| `polysyllabic_bias` | "Grand/formal" via Latinate prestige | Whissell-independent; Latin/Norman-French prestige in English |

### 1c. Identity-marking features

These mark "foreign" or "non-English-typical" without carrying any
documented sound-symbolic payload. Treat as register-marking
identifiers, not perceptual primitives. Most heavily implicated in the
`exotic` register's orientalism problem (see register grounding below).

| Dimension | Notes |
|---|---|
| `retroflexion` | Unmarked in Indic languages; marked for English speakers |
| `pharyngeal` | Unmarked in Semitic languages; marked for English speakers; Whissell 2000 hints at disgust/anger via back-of-throat articulation |

### 1d. Internally inconsistent

| Dimension | Problem |
|---|---|
| `soft_consonants` | Bundles /l m n/ (Whissell Gentle) with /r/ (Whissell firmly Harsh — "the rigid-tongue posture", 2017 p. 3). A lemma whose phonological vector says `soft_consonants: +0.8` could be /r/-heavy and read as harsh, not soft. Filed for schema-level split. |

---

## 2. The frequency code (Ohala 1994): operational summary

The single most-cited theoretical framework in this domain. From Ohala
(1994), "The frequency code underlies the sound-symbolic use of voice
pitch," in Hinton/Nichols/Ohala eds., *Sound Symbolism*, Ch. 22, pp.
325–347; Table 22.3 p. 340; key list pp. 335–336:

> **High acoustic frequency** (high F0, high formants, voiceless > voiced,
> dental/alveolar/palatal/front-velar > labial/back-velar, high-front
> vowels > low-back vowels, palatalized/[acute]/[sharp] features) →
> **small vocalizer** → submission, non-threat, deference, politeness,
> appeasement, desire for goodwill.
>
> **Low acoustic frequency** (low F0, low formants, voiced > voiceless,
> labial/velar > coronal, low-back vowels > high-front, labialized/
> velarized/pharyngealized/[grave] features) → **large vocalizer** →
> threat, dominance, self-sufficiency, aggression, authority,
> confidence.

The motivation is physical (longer vocal tracts → lower frequencies →
size signaling is survival-relevant → cross-species code). Grawunder
& Winter (2021) meta-analysis (*Phil. Trans. R. Soc. B*) confirms low
acoustic frequency reliably signals dominance across 19 species
including humans.

**Operational use in this catalog:** ANY register whose intent is
"signal size" (large = `noble grand`, `sinister menacing`; small =
`mystical ethereal`) MUST use `vowel_height` + `vowel_backness` weights.
v1 omitted these for `noble` / `mystical` / `sinister`; v1.1 adds them.

---

## 3. Whissell's affective phoneme inventory (1999, 2000, 2017)

Whissell built phoneme ratings inductively from ~15,761 words rated on
the Dictionary of Affect's pleasantness × activation plane. She placed
each English phoneme along 8 radii: Pleasantness, Cheeriness,
Activation, Nastiness, Unpleasantness, Sadness, Passivity, Softness.

The two discriminating diagonals she found most loaded
(Whissell 1999 *Perceptual and Motor Skills* 89:19-48; restated in
Whissell 2017 p. 3):

**GENTLE (pleasant + passive)** — 10 phonemes: /l, v, iː, ɛ, ɔ, aɪ, θ,
ð, m, z/. Example: "lovely, bluebell, calm, leisure, myth."

**HARSH (unpleasant + active)** — 12 phonemes: /ʃ, uː, k, ɜr, r, d, g,
t, p, ŋ, ɪ, ɔɪ/. Example: "shook, murder, tramp, gnaw, dying."

Critical observations:
- The Gentle inventory contains **zero plosives**.
- The Harsh inventory contains **every voiceless stop and every voiced
  stop** (no plosive escapes Harsh).
- The /l/ vs /r/ contrast is the load-bearing sonorant split: /l/ is
  Gentle, /r/ is Harsh ("l as sweet, r as tough", 2017 p. 3).
- Vowels do NOT split cleanly by height or backness. /iː/ (high-front
  tense) is Gentle but /ɪ/ (high-front lax) is Harsh; /ɔ/ (low-back) is
  Gentle but /uː/ (high-back) is Harsh. **Tense/lax matters more than
  front/back.**
- Pleasantness and Passivity are strongly correlated; Activation and
  Unpleasantness are strongly correlated. The off-diagonals — Cheery
  (pleasant + active) and Sad (unpleasant + passive) — are sparser.
  Whissell associates Cheery with /aɪ iː/ (smile-vowels: "cheese") and
  Sad with /uː ɔ/ in low-arousal contexts.

---

## 4. Cross-cultural validation (Blasi 2016, Ćwiek 2022, Lockwood &
   Dingemanse 2015)

Blasi et al. 2016 (PNAS) is the gold standard: 6,452 word lists,
~62% of world's languages, FDR-corrected at 5%. Five mappings
survived as statistically robust universals:

1. **roundness/calmness** → /r/, sonorants (/l m n/)
2. **small/diminutive** → /i/ + palatals; -/a/, -/u/
3. **large/heavy** → voiced obstruents /b d g/ + back vowels /u a/; -/i/
4. **sharpness/threat** → voiceless stops /k t/; -sonorants
5. **nasal/breath** → /n/, /m/

**Bouba-kiki replication** (Ćwiek et al. 2022, *Phil. Trans. B*): 25
languages, 917 speakers, 6 writing systems → 72% congruence overall,
89% with canonical phonemes. Failures (Hunjara: Rogers & Ross 1975;
Syuba: Styles & Gawne 2017) track phonotactic illegality of the
stimuli, not absence of the crossmodal mapping. **Operational lesson:**
register-tagged names should conform to the implicit phonotactics of
the register's target "feel-language" — a `harsh` name like `Tkrxv-`
overshoots and the sound-symbolic effect dies.

**European-biased phonaesthemes** (NOT cross-cultural):
- /gl-/ "light/shine" (glimmer, glow, glisten) — Proto-Indo-European
  *ghel-, NOT universal.
- /fl-/ "flow", /sn-/ "nose/contempt", /sl-/ "slimy" — English-specific.
- Sapir's /i/-small vs /a/-big lexically — failed Blasi 2016's
  basic-vocabulary screen. The size mapping is real *perceptually* but
  not encoded in word inventories worldwide.

---

## 5. Per-register grounding

Each register below documents:
- **Verdict** (UNIVERSAL / MIXED / IE-CONVENTIONAL / NO LITERATURE BACKING /
  ORIENTALISM CONCERN)
- **Citations** supporting (or failing to support) each non-trivial weight
- **v1 weights** (current catalog state as of wyrd-kq7w.2)
- **v1.1 proposed adjustments** with rationale

### grim

**Verdict:** MIXED (tags-only register; phonology stays neutral by
design so it can compose with `--register grim,harsh`).

**Citations:**
- Semantic tags (death, military, monster, undead, magic) inherit
  intent from legacy MOODS dict — no phonological claims to ground.
- Empty phonological vector is intentional and well-motivated:
  composes with phonology-bearing registers without double-counting.

**v1 weights:** phonological={}, semantic_tags={death:0.7, military:0.3,
monster:0.5, undead:0.6, magic:0.4}, position_bias={}.

**v1.1:** No adjustments. The neutrality is the design.

### harsh

**Verdict:** MIXED — universal sound-symbolic core (`stop_vs_continuant`,
voiceless stops) wrapped in IE-conventional cluster features.

**Citations:**
- `cluster_density: 0.6` — IE-conventional. Cluster-as-harsh is
  operator-side perception (Mooshammer et al. 2024 on Orkish confirms
  English-speaker-coded). Bouba-kiki literature gives zero direct
  support; Köhler/Ramachandran stimuli are CVCV, not clusters. Keep
  but tag as conventional.
- `final_fortition: 0.5` — Same. Whissell's most-active phonemes (/p t
  k/) are all stops, so word-final stops do concentrate Harsh-loading
  at perceivable word edges — but this is composition of the
  underlying stop-vs-continuant effect, not a separate primitive.
- `final_cluster_rate: 0.5` — Same as cluster_density.
- `vowel_final_bias: -0.4` — UNIVERSAL. Open-syllable endings load
  Gentle; closed endings load Harsh (Whissell 2000).
- `soft_consonants: -0.5` — UNIVERSAL. Sonorants /l m n/ are firmly
  Gentle (Whissell). Keep but flag the /r/ inconsistency.
- `stop_vs_continuant: 0.4` — UNIVERSAL. **Under-weighted at v1.**
  Whissell rates every plosive as Harsh; Fort et al. 2015 + Sidhu &
  Pexman 2024 confirm manner as the strongest single bouba-kiki
  predictor. v1.1: bump to 0.6.

**v1 → v1.1 adjustments:**
- `stop_vs_continuant: 0.4 → 0.6` (Whissell 2000; Fort 2015)
- Add `aspirated_voiceless: 0.4` (Whissell 2000: voiceless stops most-
  active phonemes; "minor explosion of breath")
- Add `vowel_backness: -0.2` (Knoeferle 2017: F2-driven sharpness)
- Add `vowel_height: 0.2` (Ohala 1994: high front for "small/sharp"
  signal in Harsh's spiky register)

### pastoral

**Verdict:** UNIVERSAL — best-grounded register in the catalog.

**Citations:**
- `vowel_final_bias: 0.3` — UNIVERSAL. Open syllables, vowel exposure,
  Whissell Gentle inventory is ~50% vowels.
- `soft_consonants: 0.4` — UNIVERSAL via /l m n/. BUT: includes /r/ in
  the dimension definition, which Whissell rates as Harsh. Flagged for
  schema-level split.
- `cluster_density: -0.3` — Indirect universal support: sonorant-heavy
  + low-cluster patterns are the "lullaby phonology" common to
  Japanese mimetics, Zulu, IE.
- Semantic tags (plant, animal, water, agriculture, tree, bird) —
  cultural overlay; no phonological claim.

**v1 → v1.1 adjustments:**
- Add `stop_vs_continuant: -0.3` (Whissell 2000; the load-bearing
  dimension was missing).
- Add `vowel_backness: 0.2` (back-rounded vowels for round-shape
  pastoral via bouba-kiki + Knoeferle 2017 F2-roundness).

### devotional

**Verdict:** IE-CONVENTIONAL (Latin/Hebrew/Sanskrit liturgical
inheritance).

**Citations:**
- No cross-linguistic universal signals "sacred."
- `polysyllabic_bias: 0.3` — Pure Latinate convention. English
  liturgical register inherits from Latin → polysyllabic = formal/
  sacred.
- `soft_consonants: 0.2` — Universal Gentle phonemes (Whissell /l m/),
  but applied for cultural reasons (monastic naming aesthetic).
- `semantic_tags: {saint, religious}` — Pure semantic, no phonological
  claim.

**v1 → v1.1 adjustments:**
- Add `vowel_final_bias: 0.2` (open syllables for Latin/Romance feel +
  Whissell's Gentle vowel-heavy inventory).
- Add `stop_vs_continuant: -0.2` (Whissell; reinforces the soft cluster).

### mortuary

**Verdict:** IE-CONVENTIONAL — no universal "death" phonology.

**Citations:**
- Empty phonological vector is appropriate; "death" is a semantic
  cluster, not a phonological one.
- Whissell's Sad radius (unpleasant + passive) corresponds to /uː ɔ/
  in low-arousal contexts — could ground a low-vowel-backness +
  low-arousal phonology if desired, but the v1 deliberate-neutrality
  is fine.

**v1 → v1.1 adjustments:**
- Optional: add `aspirated_voiceless: -0.2` (Whissell: voiceless stops
  are active; mortuary should AVOID activation to keep low-arousal
  feel). Conservative; defer to future iteration if user wants.
- v1.1 keeps mortuary unchanged. Phonological-neutrality is the
  design intent.

### noble

**Verdict:** IE-CONVENTIONAL with mis-applied frequency-code reasoning.

**Citations:**
- `polysyllabic_bias: 0.5` — Latinate prestige convention. Not
  frequency-coded (Whissell 1999 doesn't distinguish "pleasant" from
  "dignified"; Ohala 1994 size-signaling is segmental, not length-
  based). Keep as IE-conventional.
- `soft_consonants: 0.3` — Gentle phonemes for refinement
  (Whissell). BUT note: refinement ≠ dominance. "Noble" intent
  conflates two readings:
  - Refinement/politeness (Whissell Gentle) → soft + vowel-final
  - Dominance/authority (Ohala frequency code) → low-back vowels +
    voiced obstruents
- `vowel_final_bias: 0.4` — Open syllables = Romance/Latinate feel,
  IE-conventional.
- **Missing**: `vowel_height` + `vowel_backness` weights. Ohala's
  frequency code unambiguously ties grandeur/authority to low-back
  vowels (large vocalizer signal). v1 omitted these — major gap.

**v1 → v1.1 adjustments:**
- Add `vowel_height: -0.3` (Ohala 1994 p. 335: low vowels signal large/
  authoritative).
- Add `vowel_backness: 0.3` (Ohala 1994: back vowels = low F2 = large/
  dominant; Auracher 2010 back-vowels → dominance).
- Add `stop_vs_continuant: -0.2` (Whissell continuants pattern with
  Gentle/dignified).

### mystical

**Verdict:** MIXED — Ohala frequency-code chain supports "ethereal/
small" reading; v1 weight choices partially right but missed primary
vowel signals.

**Citations:**
- `sibilance: 0.4` — MIXED. Whissell rates /ʃ/ as firmly Harsh +
  active. Reading: "ethereal-whispering" (Cheery/active corner)
  vs "active-unpleasant" depends on operator intent. Defensible if
  the register is meant to signal "magical-otherworldly" (active)
  rather than "elvish-serene" (passive).
- `palatalization: 0.3` — UNIVERSAL but as "small/diminutive"
  (Kochetov & Alderete 2011), NOT specifically "magical." The
  small-signal CHAINS to ethereal via Ohala frequency code (small =
  ethereal), but it's chained, not direct.
- `soft_consonants: 0.2` — Universal Gentle phonemes.
- Semantic tags (magic, fantasy, monster) — cultural overlay.
- **Missing**: `vowel_height` + `vowel_backness` weights. Ohala
  frequency code is unambiguous: ethereal = small = high-frequency =
  high-front vowels.

**v1 → v1.1 adjustments:**
- Add `vowel_height: 0.5` (Ohala 1994: high vowels = small/ethereal).
- Add `vowel_backness: -0.4` (Ohala 1994: front vowels = small/sharp).
- Document the Kochetov-Alderete 2011 palatalization-is-diminutive
  finding in the YAML comment.
- Consider splitting `mystical` into:
  - `mystical-ethereal` (current: pleasant + slightly active, Cheery
    corner)
  - `mystical-eerie` (sibilance-driven, active-unpleasant)
  Defer to follow-up ticket.

### melodic

**Verdict:** UNIVERSAL — alongside `pastoral`, best-grounded register.

**Citations:**
- `vowel_final_bias: 0.5` — Universal. Open syllables maximize Whissell
  Gentle vowel exposure.
- `polysyllabic_bias: 0.4` — IE-conventional only (Whissell-
  independent). Keep but tag.
- `soft_consonants: 0.3` — Universal /l m n/. Same /r/-inclusion
  caveat.
- `vowel_height: 0.2` — Universal: high-front vowels are Whissell-
  pleasant (/iː/) and Cheery.

**v1 → v1.1 adjustments:**
- Add `stop_vs_continuant: -0.4` (Whissell 2000; the load-bearing
  dimension was missing; melodic should max-suppress stops to maximize
  Gentle inventory).
- Add `vowel_backness: -0.2` (front vowels = Whissell-pleasant).

### sinister

**Verdict:** MIXED — universal sharpness core but sibilance pulls
toward "creepy hissing" not "menacing dread."

**Citations:**
- `cluster_density: 0.4` — IE-conventional (Germanic stereotype).
- `final_fortition: 0.3` — Same.
- `sibilance: 0.3` — Universal but for "active/arousing", not
  "menacing/dread". Grawunder & Winter 2021 meta-analysis: dominance/
  threat = LOW frequency; sibilance is high-frequency, so it pulls
  AGAINST the dominance reading. Uno et al. (2022, "What's in a
  villain's name?") finds VOICED obstruents (/b d g dʒ v z/) favored
  in villain names, NOT high-frequency sibilants.
- Semantic tags (monster, magic, undead) — cultural overlay.

**v1 → v1.1 adjustments:**
- Add `vowel_height: -0.3` (Ohala 1994: low vowels = large/dominant
  for menacing).
- Add `vowel_backness: 0.3` (Ohala 1994; Auracher 2010).
- Add `aspirated_voiceless: -0.2` (Uno 2022: voiced obstruents favored
  in villain names; suppress voiceless to lean voiced).
- Consider splitting `sinister` into:
  - `sinister-eerie` (sibilance + palatalization, "hissing creepy")
  - `sinister-dread` (low-back vowels + voiced obstruents + pharyngeal,
    "looming menace")
  Defer to follow-up ticket (filing now).

### ancient

**Verdict:** NO LITERATURE BACKING — pure genre convention.

**Citations:**
- No frequency-code or sound-symbolism literature speaks to
  perceptions of "age" in phonology.
- `polysyllabic_bias: 0.3` — IE-conventional via Latinate archaism.
- `vowel_final_bias: 0.2` — IE-conventional via Romance / pre-Conquest
  Old English open-syllable feel.
- `pharyngeal: 0.2` — Identity-marking, not sound-symbolic. Suggests
  Semitic / pre-IE substrate; English speakers code this as
  "old-foreign" but no perceptual primitive backs it.
- Semantic tags (topography, geology, descriptive) — cultural.

**v1 → v1.1 adjustments:**
- No adjustments to weights. v1 weights are defensible as IE-
  conventional / genre tropes.
- **Add a comment** in the YAML documenting that this register has
  no phonosymbolic grounding — pure convention. Operators forking
  for non-IE settings should expect this register to NOT transfer.

### exotic

**Verdict:** IDENTITY-MARKING — functional design, not sound-symbolic.

**Citations:**
- The four weighted features (palatalization, retroflexion, pharyngeal,
  back vowels) carry no documented cross-cultural sound-symbolic
  payload. Bouba-kiki / Whissell / Blasi 2016 are all silent on
  retroflexion + pharyngeal as "spiky" or "round" or "pleasant" or
  "unpleasant" — these are unmarked features in their native
  inventories (Russian / Irish / Japanese for palatal; Hindi / Tamil
  / Mandarin for retroflex; Arabic / Hebrew for pharyngeal).
- This is the design intent of the register: the operator's default
  is English phonology, and `exotic` is the register that marks
  "phonological features outside that default" for naming use cases
  where the generator wants a distinct color (e.g. a desert /
  highland / overseas culture in a British-Isles-default world).
- The v1 YAML docstring's framing is accurate: "phonology features
  less common in English ... phonology-only — the exotic feeling is
  structural, not semantic." The work to do here is documentation,
  not removal: spell out that these weights are identity-marking,
  not crossmodal perception, so future readers don't mis-cite
  bouba-kiki literature in support of them.

**v1 → v1.1 adjustments:**
- No weight changes. v1 weights are appropriate for the register's
  intent.
- Update the YAML comment to call out that these are identity-marking
  weights (no sound-symbolic literature backs them as
  "exotic-as-perception" — the perception is operator-side, which is
  the register's design).

### martial

**Verdict:** MIXED — universal core (`stop_vs_continuant`,
`aspirated_voiceless`) wrapped in IE-conventional cluster features.

**Citations:**
- `final_fortition: 0.4` — IE-conventional, but Whissell's most-
  active phonemes are all stops, so word-final stops do concentrate
  activation.
- `cluster_density: 0.3` — IE-conventional.
- `stop_vs_continuant: 0.4` — UNIVERSAL. Whissell 2000; Auracher
  2010 (plosives = dominant/active).
- Semantic tags (military, monster) — cultural.

**v1 → v1.1 adjustments:**
- Add `aspirated_voiceless: 0.3` (Whissell 2000: voiceless stops are
  most-active; reinforces martial's active-dominant feel).
- Add `vowel_backness: 0.2` (Ohala 1994: back vowels for dominant
  authority; martial = disciplined-authoritative).

---

## 6. Follow-up tickets identified during this research pass

These are out of scope for wyrd-2166 but surfaced repeatedly:

1. **Split `soft_consonants` dimension.** Whissell 2000 + Fort 2015
   consistently rate /l m n/ as Gentle but /r/ as Harsh. Lumping them
   makes /r/-heavy lemmas score "soft" when they read harsh. Schema
   change: split into `liquid_l_m_n` and `rhotic_r` (or finer-grained
   place-of-articulation features). Requires coordinating with
   wyrd-kq7w.1 corpus enrichment pass.

2. **Tense vs lax vowel distinction.** Whissell's vowel ratings are
   non-monotonic in height: /iː/ Gentle, /ɪ/ Harsh; /ɔ/ Gentle, /uː/
   Harsh. The 14 v1 dimensions don't capture tense/lax. Schema change:
   add `vowel_tenseness` dimension or split existing vowel features
   by tenseness.

3. **Optional: split `exotic` into concrete identity-marker registers**
   (`palatalized`, `pharyngeal`, `retroflex`) so operators can pick a
   specific phonological color instead of a generic non-English mix.
   The current `exotic` register works fine for "non-default
   phonological color"; splitting is only worth doing if operators
   start asking for finer-grained control. Not a defect, just a
   potential ergonomic refinement.

4. **Split `mystical` into `mystical-ethereal` vs `mystical-eerie`.**
   v1 conflates two readings — pleasant-active (Cheery, elvish) and
   active-unpleasant (sibilant, hissing). Different operators want
   different things; one register can't serve both.

5. **Split `sinister` into `sinister-eerie` vs `sinister-dread`.**
   Same conflation: sibilance-driven creepy vs low-back-vowel +
   voiced-obstruent menacing dread. Uno 2022 finds these are distinct
   in actual villain naming.

6. **Add `liturgical` register alongside `devotional`.** Per ticket
   notes: devotional vs liturgical (open-syllable + Latin-shaped
   polysyllables) may want to be separate registers.

7. **Add `monumental` register.** Per ticket notes: extra-long
   syllables + low-formant vowels for "epic-scale" naming distinct
   from `noble` (refined-dignified).

8. **Tag each weight with `universal` vs `IE-conventional`
   metadata.** Allows future forks (non-IE fantasy settings) to know
   which weights are perceptual primitives vs Western literary
   conventions.

9. **Calibrate semantic_tags weights.** Current values in [0.2, 0.8]
   are intuition; an empirical pass would look at meaning-DB tag
   co-occurrence rates and propose calibrated weights.

10. **Phonotactic-legality gate on output.** Ćwiek 2022 / Styles &
    Gawne 2017 find that sound-symbolic effects DIE if the stimulus
    violates the target phonotactic system. Names like `Tkrxv-`
    overshoot. Add a phonotactic-legality check before scoring.

---

## 7. Sources

### Primary sound-symbolism / sound-meaning

- Hinton, L., Nichols, J., & Ohala, J. J. (eds.) (1994). *Sound
  Symbolism.* Cambridge UP. (Foundational compilation.)
- Köhler, W. (1929). *Gestalt Psychology.* New York: Liveright.
  (Original maluma/takete observation.)
- Blasi, D. E., Wichmann, S., Hammarström, H., Stadler, P. F., &
  Christiansen, M. H. (2016). Sound-meaning association biases
  evidenced across thousands of languages. *PNAS* 113(39),
  10818-10823.
- Ohala, J. J. (1994). "The frequency code underlies the sound-
  symbolic use of voice pitch." In Hinton/Nichols/Ohala eds., Ch.
  22, pp. 325-347.

### Experimental bouba-kiki + phoneme-level

- Fort, M., Martin, A., & Peperkamp, S. (2015). Consonants are More
  Important than Vowels in the Bouba-kiki Effect.
- Sidhu, D. M., & Pexman, P. M. (2018/2024). Phonological iconicity
  / phonetic underpinnings of sound symbolism across multiple
  domains.
- Knoeferle, K., et al. (2017). What drives sound symbolism?
  Different acoustic cues underlie sound-size and sound-shape
  mappings.
- Ćwiek, A., et al. (2022). The bouba/kiki effect is robust across
  cultures and writing systems. *Phil. Trans. R. Soc. B* 377(1841).
- Styles, S. J., & Gawne, L. (2017). When does maluma/takete fail?
  Two key failures and a meta-analysis suggest that phonology and
  phonotactics matter. *i-Perception* 8(4).
- Bremner, A. J., et al. (2013). "Bouba" and "Kiki" in Namibia? A
  remote culture make similar shape-sound matches, but different
  shape-taste matches to Westerners. *Cognition*.

### Whissell phonemic-emotion

- Whissell, C. (1999). Phonosymbolism and the Emotional Nature of
  Sounds. *Perceptual and Motor Skills* 89, 19-48.
- Whissell, C. (2000). Phonoemotional Profiling: A Description of
  the Emotional Flavour of English Texts on the Basis of the
  Phonemes Employed.
- Whissell, C. (2017). Sound Symbolism in Shakespeare's Sonnets.
  *English Language and Literature Studies* 7(4). [Open access;
  restates the 1999/2000 Gentle/Harsh inventories on p. 3.]

### Frequency-code follow-ups + replication

- Grawunder, S., & Winter, B. (2021). Rethinking the frequency
  code: a meta-analytic review. *Phil. Trans. R. Soc. B*.
- Auracher, J., et al. (2010). P is for happiness, N is for
  sadness: universals in sound iconicity to detect emotions in
  poetry.
- Aryani, A., et al. (2018). Affective congruence between sound
  and meaning of words facilitates semantic decision.
- Uno, R., et al. (2022). What's in a villain's name? Sound-
  symbolic values of voiced obstruents and bilabial consonants.
  *Review of Cognitive Linguistics*.

### Cross-cultural sound symbolism

- Lockwood, G., & Dingemanse, M. (2015). Iconicity in the lab: a
  review of behavioral, developmental, and neuroimaging research
  into sound-symbolism. *Frontiers in Psychology*.
- Dingemanse, M., et al. (2015). Arbitrariness, iconicity, and
  systematicity in language. *Trends in Cognitive Sciences*.
- Lockwood, G., Dingemanse, M., & Hagoort, P. (2016). Sound-
  symbolism boosts novel word learning.
- Shinohara, K., & Kawahara, S. (2010). A cross-linguistic study
  of sound symbolism: the images of size.
- Kochetov, A., & Alderete, J. (2011). Patterns and scales of
  expressive palatalization. *Canadian Journal of Linguistics* 56.
- Mooshammer, C., et al. (2024). Does Orkish sound evil?
  Phonological features of fictional villain-language.

### English-language reference

- Crystal, D. (1995). *The Cambridge Encyclopedia of the English
  Language.* Cambridge UP.

---

**Status of v1.1 catalog update:** see `data/register_effects.yaml`
for the actual weight changes. This document is the citation-
bearing companion that grounds those changes.

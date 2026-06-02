<script>
  // wyrd-yxf6 + wyrd-hpjg: one card per morpheme in InspectorColumn.
  // Shows usage as the card header + meanings + tags + per-source-
  // language panels carrying form / IPA / reader_pronunciation /
  // original_script / dialect (wyrd-cp2d + wyrd-03cx data).
  //
  // wyrd-hpjg: form rows are clickable buttons — picking one adds a
  // Swap step to the pipeline (replaces this morpheme's usage with
  // the picked form). v1 "siblings" = the alternate forms within
  // THIS morpheme's own sources/renderings; cluster-walking (other
  // morphemes sharing an etymon) requires a backend endpoint and
  // lands in a follow-up.
  import { languageLabel } from '../lib/languageLabels.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { accentFold, graftPosition } from '../lib/accents.js';
  import { activeRendering } from '../lib/variants.js';
  import MorphemeBreakdown from './MorphemeBreakdown.svelte';

  let { morpheme, morphemeIndex } = $props();

  const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');
  // Case- + dash-insensitive key for MATCHING a surface to a form row.
  // Accents are KEPT (so "hy" and "hȳ" stay distinct — clicking the
  // accented row is a real upgrade, not a no-op).
  const norm = (s) => stripDashes(s).toLowerCase();

  // The cell's ORIGINAL generated surface (col-2 result), used by
  // setSwap's value-based revert. Intentionally the RAW pre-accent
  // surface ("hy"), which may differ from the inspector's accent-
  // upgraded default ("hȳ"); reverts route through swapTo's
  // sameModuloCase → clearSwap (identity-based) first, so this value
  // path is a backstop and the divergence is benign.
  let originalUsage = $derived(
    appState.currentResult?.morphemes_by_word?.[morpheme._wordIndex]?.[
      morphemeIndex
    ]?.usage || '',
  );

  // wyrd-2b50 follow-up / wyrd-thhb: swap to the ACCENTED display form
  // (original_script, e.g. "Wālum"/"hȳ") when the row has one, not the ASCII
  // transliteration, so the name keeps its accents. `lang` is the source-
  // language panel the clicked row belongs to (null for the synthetic
  // original-surface row) — it's pinned on the swap so a cross-language
  // homograph (Old Norse "by" /biː/ vs Old English bȳ /byː/, or OE/OF/Celtic
  // "don") selects the RIGHT pronunciation even when the surface is identical.
  function swapTo(form, rendering, lang = null) {
    // wyrd-at53: graft the chosen surface into this slot's position (dashes
    // from the ORIGINAL generated usage) + its case rule, so a suffix "-hām"
    // swapped to "Hamm" becomes "-hamm" (dash kept, lowercased) — not "Hamm".
    const target = graftPosition(
      originalUsage || morpheme.usage,
      rendering?.original_script || form,
    );
    // Clicking the row that is ALREADY active reverts to the canonical
    // generated default. For a language-panel row the "active" test is the
    // (lang, form) pair — not surface alone — so clicking a different
    // language's same-surface row still registers as a real selection. The
    // synthetic original row (lang == null) keeps the surface-only test.
    const isCurrent =
      lang != null
        ? currentRef?.lang === lang &&
          stripDashes(currentRef?.form).toLowerCase() ===
            stripDashes(form).toLowerCase()
        : stripDashes(morpheme.usage).toLowerCase() ===
          stripDashes(target).toLowerCase();
    if (isCurrent) {
      pipeline.clearSwap({
        wordIndex: morpheme._wordIndex,
        morphemeIndex,
      });
      return;
    }
    pipeline.setSwap({
      wordIndex: morpheme._wordIndex,
      morphemeIndex,
      to: target,
      original: originalUsage,
      language: lang || undefined,
    });
  }


  // Language display order, mirroring _ANCHOR_LANG_PREFERENCE in
  // wyrd/generators/kenning/era/rewind.py — the etymological-anchor
  // priority that the CLI rewinder uses to pick a source-language
  // for time-rewind. OE first, then Norse (Danelaw), then OF/Celtic,
  // then modern_english as a last-resort anchor. Frequency-wise
  // modern_english is the most common rendering in the bundle, but
  // showing it first would bury the historically-interesting forms
  // — the inspector's point is to surface etymology, not
  // distribution. Unknown languages fall through to alphabetical.
  const LANG_PRIORITY = [
    'old_english',
    'old _english',
    'old_scandinavian',
    'old_scandanavian',
    'old_french',
    'celtic_mix',
    'latin',
    'modern_english',
  ];

  // Union of languages present in EITHER sources (the bundle's
  // per-morpheme picks) OR renderings (cross-language rendering
  // data, often richer — the etymological cluster's OE attestations
  // show up in renderings even when sources only carries the modern
  // pick). For each language, union the forms across the two dicts.
  let orderedSourceEntries = $derived.by(() => {
    const sources = morpheme.sources || {};
    const renderings = morpheme.renderings || {};
    const langs = new Set([
      ...Object.keys(sources),
      ...Object.keys(renderings),
    ]);
    const entries = [...langs].map((lang) => {
      // wyrd-0h96: dedup forms that collapse to the same surface under
      // CASE + ACCENT folding, not just the raw key. A row renders its
      // accented original_script when present (form "tun" displays as
      // "tūn (tun)"), so a plain "tūn" form, a "tun" form whose
      // original_script is "tūn", AND a bare "By" next to "bȳ (by)" are all
      // the SAME variant shown two/three times. Key on the accent-folded,
      // case-folded display surface (original_script || form) so "bȳ", "by",
      // and "By" collapse to one row.
      //
      // On collision, keep the form whose rendering carries data
      // (IPA/original_script/reader) so the surviving row is the accented one
      // WITH its pronunciation — not the bare "By" that has neither. (The
      // accent-upgrade is reached by clicking that accented row, so folding
      // the bare twin away doesn't lose the upgrade.)
      const byKey = new Map(); // displayKey -> { form, rich }
      for (const f of [
        ...(sources[lang] || []),
        ...Object.keys(renderings[lang] || {}),
      ]) {
        const r = renderingFor(lang, f);
        const k = accentFold(r.original_script || f);
        if (!k) continue;
        // "rich" = the row would render distinct data: real IPA / reader, or
        // an original_script that actually DIFFERS from the form (a plain
        // form whose original_script === itself renders no differently, so
        // it isn't rich and shouldn't block upgrading to a transliterated
        // twin that carries IPA).
        const rich = !!(
          r.ipa ||
          (r.original_script && r.original_script !== f) ||
          r.reader_pronunciation
        );
        const existing = byKey.get(k);
        if (!existing) {
          byKey.set(k, { form: f, rich });
        } else if (rich && !existing.rich) {
          existing.form = f; // upgrade to the form that carries rendering data
          existing.rich = true;
        }
        // When two DISTINCT forms are both rich and share a display key,
        // first-seen wins (sources before rendering keys). Today's data has
        // at most one rich twin per surface, so this is a documented
        // assumption, not an observed case.
      }
      const forms = [...byKey.values()].map((v) => v.form);
      return [lang, forms];
    });
    entries.sort(([a], [b]) => {
      const ai = LANG_PRIORITY.indexOf(a);
      const bi = LANG_PRIORITY.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a.localeCompare(b);
    });
    return entries;
  });

  // The SINGLE active row across the whole card. wyrd-thhb: derived from
  // activeRendering (the pinned `_lang`, else the canonical `sources`
  // language, else the first match) so exactly ONE row highlights — the SAME
  // language the pronunciation guide shows — even when the surface appears
  // under several languages ("don" in OE/OF/Celtic; "by" in OE + Old Norse).
  // The active rendering's form is mapped to the surviving (deduped) row in
  // its language via the same accent-fold the panel dedup uses. Null when the
  // surface isn't in any rendering (e.g. "-ton"/"-ham" generated surfaces),
  // so the synthetic original-surface row highlights instead.
  let currentRef = $derived.by(() => {
    const active = activeRendering(morpheme);
    if (!active) return null;
    const entry = orderedSourceEntries.find(([lang]) => lang === active.lang);
    if (!entry) return { lang: active.lang, form: active.form };
    const activeKey = accentFold(
      renderingFor(active.lang, active.form).original_script || active.form,
    );
    // The surviving form of active.form's fold-group folds to the same key,
    // so this find matches it (active.form itself if it survived, else its
    // deduped twin). Only a degenerate all-dashes form yields activeKey ===
    // '' with no match → fall back to active.form (no row highlights, but no
    // wrong-row highlight either).
    const form = entry[1].find(
      (f) => accentFold(renderingFor(active.lang, f).original_script || f) === activeKey,
    );
    return { lang: active.lang, form: form ?? active.form };
  });

  // wyrd-7lg1: persistent synthetic row for the ORIGINAL generated surface
  // when it isn't one of the etymon's own forms ("-ham", "-le-", "-ton"...).
  // Keyed on the ORIGINAL surface — NOT the current usage — so after a swap
  // the original stays listed as a revert target. Previously this was keyed
  // on the current usage (`!usageInForms`), so swapping to a real form made
  // `usageInForms` flip true and the original-surface row vanished ("click
  // tūn → the -ton option disappears", "click read → Red- disappears").
  let originalInForms = $derived.by(() => {
    const u = norm(originalUsage);
    if (!u) return false;
    for (const [lang, forms] of orderedSourceEntries) {
      for (const f of forms) {
        if (u === norm(f) || u === norm(renderingFor(lang, f).original_script)) {
          return true;
        }
      }
    }
    return false;
  });
  // The original-surface row is the active (highlighted) one exactly when no
  // etymon form row matches the current usage (currentRef === null). Keying
  // off currentRef rather than a usage===originalUsage string compare keeps
  // the highlight correct when the inspector accent-upgrades the default
  // surface (usage "hȳ" vs raw original "hy"), and guarantees a single
  // highlight: if a form row matches, IT highlights and this row doesn't.
  let originalIsCurrent = $derived(currentRef === null);

  // For each form within a source language, pull the matching
  // renderings slot (ipa / reader_pronunciation / original_script /
  // dialect) so the panel can render the four-column data with one
  // template iteration.
  function renderingFor(lang, form) {
    return morpheme.renderings?.[lang]?.[form] || {};
  }

  // wyrd-mf2u: era-render display. The card renders the era form (what's in the
  // name) + its own pronunciation as the primary breakdown, with a second
  // breakdown for the modern morpheme beside it (the glossed anchor — era
  // reflexes are cluster-cognates, so meaning rides with the modern form, shown
  // once below). rendered_pron is resolved server-side for THIS exact form
  // (rich IPA where the bundle has it, rule respelling else).
  let isEraRender = $derived(!!(morpheme.rendered_language && morpheme.rendered));
  // The modern anchor's pronunciation object, restricted to modern_english so
  // we never mislabel a historical-cluster pronunciation as "modern". Null when
  // the bundle carries no modern_english rendering for the surface (common —
  // then the modern breakdown shows just the surface + the shared gloss below).
  let modernPron = $derived.by(() => {
    const u = norm(morpheme.usage);
    const forms = morpheme.renderings?.modern_english || {};
    for (const [f, data] of Object.entries(forms)) {
      if (norm(f) === u || norm(data.original_script || '') === u) return data;
    }
    return null;
  });

  // wyrd-438r: total clickable variant rows (the synthetic original-surface
  // row, when shown, + every per-language form) — for the "Variants (N)"
  // disclosure summary.
  let variantFormCount = $derived(
    (!originalInForms && originalUsage ? 1 : 0) +
      orderedSourceEntries.reduce((n, [, forms]) => n + forms.length, 0),
  );

  // wyrd-atc6: the original generated surface ("-ton") belongs UNDER its
  // source language ("Old English") rather than in a separate "original
  // surface" section. Its language is the morpheme's canonical source (the
  // etymon it was generated from). Render the row at the top of that
  // language's panel; only fall back to a standalone section when the surface
  // has no source-language panel at all (sources empty — rare).
  let originalLang = $derived(Object.keys(morpheme.sources || {})[0] || null);
  let originalLangInEntries = $derived(
    !!originalLang && orderedSourceEntries.some(([lang]) => lang === originalLang),
  );
  let showOriginalRow = $derived(!originalInForms && !!originalUsage);
</script>

<article class="morpheme" aria-labelledby="m-{morpheme.usage}-{morpheme._wordIndex}">
  <header>
    <!-- AT heading-nav: h3 (result name) → h4 (Morphemes section) → h5
         (per-morpheme card). The h5 carries the card's label + id; the
         reusable MorphemeBreakdown(s) provide the visible form/pronunciation. -->
    {#if isEraRender}
      <!-- wyrd-mf2u: era render — the period form (in the name) + its own
           pronunciation as the primary breakdown, beside the modern anchor.
           Same reusable card, two instances. Meaning rides with the modern
           form (shown once in the shared gloss below). -->
      <h5 class="sr-only" id="m-{morpheme.usage}-{morpheme._wordIndex}">
        {morpheme.rendered}, {languageLabel(morpheme.rendered_language)} (modern {morpheme.usage})
      </h5>
      <div class="form-pair">
        <MorphemeBreakdown
          primary
          form={morpheme.rendered}
          pron={morpheme.rendered_pron}
          label={languageLabel(morpheme.rendered_language)}
        />
        <MorphemeBreakdown form={morpheme.usage} pron={modernPron} label="modern" />
      </div>
    {:else}
      <h5 class="usage" id="m-{morpheme.usage}-{morpheme._wordIndex}">
        {morpheme.usage}
      </h5>
      {#if morpheme.rendered && morpheme.rendered !== morpheme.usage}
        <span class="rendered" title="D18/D8 substituted variant">
          → {morpheme.rendered}
        </span>
      {/if}
    {/if}
  </header>

  {#if morpheme.meaning_groups?.length}
    <!-- wyrd-0y3k: each group is one etymon/sense (deduped) — shown as a
         separate line so distinct senses ('valley/dale' vs 'farm/town' vs
         'people') read as groups, not one ~50-item run. -->
    <div class="meaning-groups">
      {#each morpheme.meaning_groups as group}
        <p class="meaning-group">{group.join(', ')}</p>
      {/each}
    </div>
  {:else if morpheme.meanings?.length}
    <p class="meanings">{morpheme.meanings.join(', ')}</p>
  {/if}

  {#if morpheme.derivative_meanings?.length}
    <!-- wyrd-o53o: morphological-derivative glosses (e.g. 'alternative
         form of X', 'soft mutation of Y') surfaced separately from
         primary semantic meanings. Dimmed + labeled so the operator
         sees the data exists but knows it's a pointer, not a definition.
         Empty / missing field renders nothing (most morphemes). -->
    <p class="derivative-meanings" title="Morphological pointers (alternative form / plural / mutation / inflection of another lemma) — kept for reference; not the primary meaning.">
      <span class="derivative-label">related</span>
      {morpheme.derivative_meanings.join(', ')}
    </p>
  {/if}

  {#if morpheme.tags?.length}
    <p class="tags">
      {#each morpheme.tags as tag}
        <span class="tag">{tag}</span>
      {/each}
    </p>
  {/if}

  {#if morpheme.citations?.length}
    <!-- wyrd-bvwu: scholarly attestations (source_ids, wyrd-9kh.1) surfaced
         on demand — collapsed by default so they don't clutter the card,
         expandable to view which sources attest this morpheme. -->
    <details class="citations">
      <summary>Citations ({morpheme.citations.length})</summary>
      <ul>
        {#each morpheme.citations as src}
          <li>{src}</li>
        {/each}
      </ul>
    </details>
  {/if}

  {#if variantFormCount > 0}
    <!-- wyrd-438r: fold the per-language variant rows behind a clear
         "Variants" disclosure. Collapsed by default so the card stays compact
         (and more of the transform stack shows); the summary signals that the
         rows inside are click-to-switch. -->
    <details class="variants">
      <summary
        >Variants ({variantFormCount})<span class="variants-hint">
          — click a form to switch</span
        ></summary
      >

  {#if showOriginalRow && !originalLangInEntries}
    <!-- wyrd-7lg1/atc6: the original generated surface, shown standalone ONLY
         when it has no source-language panel to live under (sources empty). -->
    <section class="source-lang">
      <h4>original surface</h4>
      <table class="forms">
        <tbody>
          <tr class="form-row" class:current={originalIsCurrent}>
            <td class="form">
              <button
                type="button"
                class="form-btn"
                onclick={() => swapTo(originalUsage, null)}
                title={originalIsCurrent
                  ? 'Original generated surface (current)'
                  : 'Original generated surface — click to revert'}
              >
                {originalUsage}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  {/if}

  {#each orderedSourceEntries as [lang, forms] (lang)}
    <section class="source-lang">
      <h4>{languageLabel(lang)}</h4>
      <table class="forms">
        <thead class="sr-only">
          <tr>
            <th>Form</th>
            <th>IPA</th>
            <th>Reader pronunciation</th>
            <th>Dialect</th>
          </tr>
        </thead>
        <tbody>
          {#if showOriginalRow && originalLangInEntries && lang === originalLang}
            <!-- wyrd-atc6: the original generated surface, under its source
                 language (tagged 'gen' so it's distinct from curated etymon
                 forms). Clickable to revert; highlighted when it's active. -->
            <tr class="form-row" class:current={originalIsCurrent}>
              <td class="form">
                <button
                  type="button"
                  class="form-btn"
                  onclick={() => swapTo(originalUsage, null)}
                  title={originalIsCurrent
                    ? 'Original generated surface (current)'
                    : 'Original generated surface — click to revert'}
                >
                  {originalUsage}<span class="gen-tag">gen</span>
                </button>
              </td>
              <td class="ipa"></td>
              <td class="reader"></td>
              <td class="dialect"></td>
            </tr>
          {/if}
          {#each forms as form (form)}
            {@const r = renderingFor(lang, form)}
            {@const current = currentRef?.lang === lang && currentRef?.form === form}
            <tr class="form-row" class:current>
              <td class="form">
                <!-- wyrd-hpjg: each form is a click-to-swap button.
                     Routes through pipeline.setSwap so re-clicks edit
                     the single swap step for this (word, morpheme)
                     cell, and clicking the current/original form
                     reverts. Keyboard- + screen-reader-accessible. -->
                <button
                  type="button"
                  class="form-btn"
                  onclick={() => swapTo(form, r, lang)}
                  title={current
                    ? 'Current form — click to revert to the original'
                    : `Swap this morpheme's surface to: ${r.original_script || form}`}
                >
                  {#if r.original_script && r.original_script !== form}
                    <span class="original">{r.original_script}</span>
                    <span class="translit">({form})</span>
                  {:else}
                    {form}
                  {/if}
                </button>
              </td>
              <td class="ipa">{r.ipa || ''}</td>
              <td class="reader" title="reader-friendly pronunciation"
                >{r.reader_pronunciation || ''}</td>
              <td class="dialect">{r.dialect || ''}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/each}
    </details>
  {/if}
</article>

<style>
  .morpheme {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 6px;
  }
  h5.usage {
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  .rendered {
    font-size: 12px;
    color: var(--fg-muted);
  }
  /* wyrd-mf2u: era render lays the period form + modern anchor side-by-side
     (two reusable MorphemeBreakdown cards), wrapping on narrow widths. */
  .form-pair {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
    align-items: flex-start;
  }
  .form-pair :global(.morpheme-breakdown) {
    flex: 1 1 auto;
  }
  .meanings {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--fg);
    line-height: 1.5;
  }
  /* wyrd-o53o: derivative glosses rendered dimmed + italic so the
     visual hierarchy is clear (primary meanings dominate; pointers
     are secondary reference). 'related' label sits inline as a
     monospace tag for a 'metadata-ish' feel. */
  .derivative-meanings {
    margin: -4px 0 8px;
    font-size: 11px;
    color: var(--muted);
    font-style: italic;
    line-height: 1.5;
  }
  .derivative-label {
    display: inline-block;
    margin-right: 6px;
    padding: 0 4px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-style: normal;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: var(--bg-elev);
    border-radius: 2px;
    color: var(--muted);
  }
  .tags {
    margin: 0 0 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .tag {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    background: var(--border);
    color: var(--fg-muted);
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .source-lang {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
  }
  .source-lang h4 {
    margin: 0 0 6px;
    font-size: 10px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  table.forms {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  table.forms td {
    padding: 3px 8px 3px 0;
    vertical-align: top;
    line-height: 1.5;
  }
  td.form {
    font-weight: 600;
    color: var(--fg);
    min-width: 60px;
  }
  .form-btn {
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
    font: inherit;
    font-weight: 600;
    padding: 0;
    text-align: left;
  }
  .form-btn:hover {
    color: var(--accent);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  /* wyrd-hpjg round 2 (frontend MED): keyboard focus indicator.
     Without this, Tab users had no visible affordance on the
     swap targets (transparent bg + no border + no outline). */
  .form-btn:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 2px;
  }
  .form-row.current .form-btn {
    color: var(--accent);
  }
  .form-row.current .form-btn::before {
    content: '● ';
    font-size: 8px;
    vertical-align: middle;
  }
  .original {
    font-family: serif;
    font-size: 13px;
  }
  .translit {
    color: var(--fg-muted);
    font-size: 11px;
    margin-left: 4px;
  }
  /* wyrd-atc6: 'gen' marker on the original generated-surface row so it reads
     as distinct from the curated etymon forms it now sits beside. */
  .gen-tag {
    margin-left: 5px;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0 3px;
    vertical-align: middle;
  }
  td.ipa {
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
  }
  td.reader {
    color: var(--accent);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
    letter-spacing: 0.04em;
  }
  td.dialect {
    color: var(--fg-muted);
    font-size: 10px;
    font-style: italic;
  }
  /* wyrd-bvwu: collapsed-by-default citations disclosure. */
  .citations {
    margin: 0 0 10px;
  }
  .citations summary {
    cursor: pointer;
    font-size: 10px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .citations summary:hover {
    color: var(--accent);
  }
  .citations ul {
    margin: 6px 0 0;
    padding-left: 16px;
    list-style: disc;
  }
  .citations li {
    font-size: 11px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    line-height: 1.5;
  }
  /* wyrd-0y3k: render each sense group on its own line with a light left
     rule so distinct senses read as separate clusters. */
  .meaning-groups {
    margin: 0 0 8px;
  }
  .meaning-group {
    margin: 0;
    padding-left: 8px;
    border-left: 2px solid var(--border);
    font-size: 12px;
    color: var(--fg);
    line-height: 1.5;
  }
  .meaning-group + .meaning-group {
    margin-top: 4px;
  }
  /* wyrd-438r: variants disclosure (collapsed by default), styled like the
     citations one; the hint de-emphasizes the click-to-switch affordance. */
  .variants {
    margin: 10px 0 0;
  }
  .variants > summary {
    cursor: pointer;
    font-size: 10px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .variants > summary:hover {
    color: var(--accent);
  }
  .variants-hint {
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
    color: var(--muted);
  }
</style>

<script>
  // wyrd-yxf6 + wyrd-kppy: col 3 — morpheme detail inspector +
  // transform pipeline.
  //
  // Reads appState.currentResult (set by OutputColumn when the user
  // clicks a result card). Renders the CURRENT name (post-pipeline
  // if steps exist, else the original) + per-morpheme cards carrying
  // sources/IPA/reader_pronunciation/original_script.
  //
  // wyrd-kppy: the pipeline engine (lib/pipeline.svelte.js) tracks
  // a user-editable stack of transform steps. An \$effect below
  // re-runs the pipeline whenever the original (col-2 selection)
  // or the step list changes — the "editable recipe" model from
  // the design session (vs an undo/redo snapshot history).
  import { untrack } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { renderName } from '../lib/transforms/swap.js';
  import { languageLabel } from '../lib/languageLabels.js';
  import MorphemeCard from '../components/MorphemeCard.svelte';
  import NameGuideCard from '../components/NameGuideCard.svelte';
  import TransformStack from '../components/TransformStack.svelte';
  import DefectModal from '../components/DefectModal.svelte';
  import { upgradeAccents } from '../lib/accents.js';
  import { activeRendering, pronunciationFor } from '../lib/variants.js';
  // wyrd-8jjx: SaveWorkspaceButton + ShareWorkspaceButton moved to
  // the Header (universal across workspaces). Components stay in
  // the tree for potential reuse but no longer rendered here.

  let result = $derived(appState.currentResult);

  // wyrd-z3fl: "report defective" lives here (moved from the Output
  // column) so the user flags the result they're inspecting. We flag the
  // original generated result (appState.currentResult) — the reproducible
  // generator output — not the post-transform displayState.
  //
  // Track the SPECIFIC result being flagged rather than a boolean: the
  // modal's open state is then `flaggedResult === result`, derived
  // synchronously. When the inspected result changes the modal closes in
  // the same render (the derived comparison goes false) — no effect drives
  // `open`, so no one-frame flash / double-render.
  let flaggedResult = $state(null);

  // Housekeeping only: null out a stale flaggedResult once the inspected
  // result has moved off it. Without this, navigating BACK to a result the
  // user had previously flagged would re-satisfy `flaggedResult === result`
  // and reopen the modal unexpectedly. This effect never drives `open`
  // (that stays derived + already false here), so it adds no flash; untrack
  // keeps the flaggedResult write from re-triggering the effect.
  $effect(() => {
    const current = result;
    untrack(() => {
      if (flaggedResult !== null && flaggedResult !== current) {
        flaggedResult = null;
      }
    });
  });

  const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');

  // wyrd-de5t: upgradeAccents/accentedUsage moved to lib/accents.js so the
  // Output column (col 2) shares the exact same accent-upgrade logic and the
  // two columns agree on the accented surfaces + name.

  // The "original" state the pipeline feeds — the col-2 selection, with
  // accents applied by default (the user's "show hȳ, not hy" ask). The
  // accented head name is only re-rendered when renderName is PROVEN
  // lossless for this name (reproduces the plain bundle name) — the
  // renderer reproduces ~9/10 names, so an unsafe one keeps the bundle
  // name + plain morphemes (1-click accent via the cards) rather than
  // risk corrupting casing (e.g. "HamySide" -> "Hamyside").
  let original = $derived.by(() => {
    if (!result) return null;
    const plain = result.morphemes_by_word || [];
    const up = upgradeAccents(plain);
    if (!up.changed) {
      return { name: result.result, morphemes_by_word: plain };
    }
    const lossless = renderName(plain) === result.result;
    return lossless
      ? { name: renderName(up.words), morphemes_by_word: up.words }
      : { name: result.result, morphemes_by_word: plain };
  });

  // wyrd-kppy round 2: unified subject-change + pipeline-run
  // effect so the clear-then-run ordering is atomic (pre-fix two
  // sibling effects depended on currentResultIndex; declaration
  // order made it work but it was fragile to file reorganization).
  // When the user clicks a different result we clear the pipeline
  // synchronously THEN kick off a fresh run; child reads see the
  // post-clear state immediately. wyrd-34tn (PR #6 save/load) is
  // where users will get explicit "keep this pipeline" preservation.
  let lastResultIndex = $state(null);
  $effect(() => {
    const stepsSnapshot = $state.snapshot(pipeline.steps);
    const idx = appState.currentResultIndex;
    // wyrd-34tn round 2 (Gemini HIGH): always consume + clear the
    // loading flag, regardless of whether idx actually changed. If
    // a user loads a saved workspace for the result they're already
    // inspecting, idx won't change — but the flag still needs to be
    // cleared so the NEXT subject change behaves normally.
    // wyrd-34tn round 3 (Gemini MED): untrack() the read+write so
    // clearing the flag doesn't re-trigger this effect (which would
    // double-run pipeline.run; the runToken protects state but the
    // wasted run is inefficient).
    const isLoad = untrack(() => {
      const val = appState.isLoadingSavedWorkspace;
      if (val) appState.isLoadingSavedWorkspace = false;
      return val;
    });
    if (idx !== lastResultIndex) {
      lastResultIndex = idx;
      if (!isLoad) pipeline.clear();
    }
    if (!original) return;
    pipeline.run(original);
    void stepsSnapshot;
  });

  // The currently-displayed state — post-pipeline if any, else the
  // original. Drives the head + morpheme cards.
  let displayState = $derived(pipeline.currentState);

  let allMorphemes = $derived.by(() => {
    if (!displayState?.morphemes_by_word) return [];
    const out = [];
    displayState.morphemes_by_word.forEach((word, wi) => {
      // wyrd-hpjg: thread the in-word morpheme index too so
      // MorphemeCard's click-to-swap UX can target the right
      // (wordIndex, morphemeIndex) cell on the pipeline state.
      word.forEach((m, mi) =>
        out.push({ ...m, _wordIndex: wi, _morphemeIndex: mi }),
      );
    });
    return out;
  });

  // The rendering slot (ipa / reader_pronunciation) for a morpheme's CURRENT
  // usage. wyrd-thhb: the CANONICAL language (the etymon in `sources`) by
  // default — or the user's pinned `_lang`. wyrd-cxwl: pronunciationFor adds a
  // fallback to the etymon's pronunciation when the generated surface's
  // spelling doesn't match a form that carries IPA ("ay" ← "ey"), so the guide
  // isn't blank. (currentRef in the card still uses activeRendering directly,
  // so the highlight only fires on a real form match.)
  function renderingForUsage(m) {
    return pronunciationFor(m);
  }

  // wyrd-bapd: a morpheme's primary definition — the first gloss of the first
  // (deduped) sense group, else the first flat meaning — shown in the guide so
  // the user sees what each morpheme MEANS, not just how it sounds.
  function primaryDef(m) {
    return m?.meaning_groups?.[0]?.[0] || m?.meanings?.[0] || '';
  }

  // wyrd-2ien: the breakdown is two reusable NameGuideCards — the generated
  // (era) name + its guide, and beside it the MODERN name + its guide (dimmed).
  // Each guide row is a morpheme's surface over its reader-pronunciation / IPA /
  // gloss (gaps render as a dim '·' — pronunciation coverage is sparse).
  // strip leading/trailing position dashes for clean display ("-ton" → "ton").
  const stripDash = (s) => (s || '').replace(/^-+|-+$/g, '');

  // wyrd-2ien: the MODERN card is the stable root — the ORIGINAL generated
  // morphemes' modern forms (the clearest, most common version of the name). It
  // does NOT follow variant swaps / transforms (those mutate `displayState`);
  // only the era card is the working/mutable side. So everything modern + the
  // era-render detection reads `original`; only the era guide reads
  // `displayState`.
  let isEraName = $derived(
    (original?.morphemes_by_word || []).flat().some((m) => m.rendered_language),
  );
  let eraLabel = $derived.by(() => {
    const m = (original?.morphemes_by_word || []).flat().find((m) => m.rendered_language);
    return m ? languageLabel(m.rendered_language) : '';
  });
  // Modern name = the original morphemes composed in their modern (usage) form.
  let modernName = $derived(renderName(original?.morphemes_by_word || []));
  // Era card rows: the era form + its OWN pronunciation, from the WORKING state
  // (reflects swaps/transforms). A no-reflex morpheme falls back to its modern
  // surface + modern pronunciation.
  let eraRows = $derived.by(() =>
    (displayState?.morphemes_by_word || []).flatMap((word) =>
      word
        .filter((m) => m.usage?.trim())
        .map((m) => {
          const era = m.rendered_language && m.rendered;
          const slot = era ? m.rendered_pron : renderingForUsage(m);
          return {
            surface: stripDash(m.rendered || m.usage),
            reader: slot?.reader_pronunciation,
            ipa: slot?.ipa,
            gloss: primaryDef(m),
          };
        }),
    ),
  );
  // Modern card rows: the ORIGINAL modern surface + modern pronunciation (stable
  // root lemma; unaffected by swaps so it stays the clearest reference).
  let modernRows = $derived.by(() =>
    (original?.morphemes_by_word || []).flatMap((word) =>
      word
        .filter((m) => m.usage?.trim())
        .map((m) => {
          const slot = renderingForUsage(m);
          return {
            surface: stripDash(m.usage),
            reader: slot?.reader_pronunciation,
            ipa: slot?.ipa,
            gloss: primaryDef(m),
          };
        }),
    ),
  );
</script>

<section class="column">
  <h2>Inspect &amp; Transform</h2>

  {#if !result}
    <p class="placeholder">
      Click a result in the middle column to inspect its morphemes,
      etymology, and pronunciation.
    </p>
  {:else}
    <!-- wyrd-o7lp (PR #316 Gemini MED): dropped the {:else if
         !displayState} branch — it was unreachable since
         pipeline.currentState falls back to the original whenever
         appState.currentResult is non-null, which is the very
         condition that brought us into this {:else}. -->
    <header class="head">
      <div class="head-top">
        {#if pipeline.isRunning}
          <span class="pending-flag" title="pipeline running">…</span>
        {/if}
        <button
          type="button"
          class="flag"
          onclick={() => (flaggedResult = result)}
          title="Report defective"
        ><span aria-hidden="true">⚑</span> Report defective</button>
      </div>
      <!-- wyrd-2ien: the generated name + its guide, and (for an era render) the
           modern name + its guide beside it — two instances of one card. -->
      <div class="name-cards">
        <NameGuideCard
          headingId="inspect-result-name"
          name={displayState.name}
          label={isEraName ? eraLabel : ''}
          rows={eraRows}
        />
        {#if isEraName}
          <NameGuideCard name={modernName} label="modern" dim rows={modernRows} />
        {/if}
      </div>
      {#if pipeline.steps.length > 0 && displayState.name !== original.name}
        <p class="provenance">
          from <span class="orig">{original.name}</span>
        </p>
      {/if}
    </header>

    <section class="morphemes">
      <h4 class="section-head">
        Morphemes ({allMorphemes.length})
      </h4>

      {#if allMorphemes.length === 0}
        <p class="placeholder">
          This generator doesn't expose per-morpheme metadata.
        </p>
      {:else}
        <!-- wyrd-hpjg round 2 (Gemini MED): key by position, not
             usage. Pre-fix, swapping a morpheme's usage changed
             the key, causing MorphemeCard destroy/recreate (lose
             focus + scroll state on the card). Position-based key
             is stable across swaps. -->
        {#each allMorphemes as morpheme (morpheme._wordIndex + ':' + morpheme._morphemeIndex)}
          <MorphemeCard {morpheme} morphemeIndex={morpheme._morphemeIndex} />
        {/each}
      {/if}
    </section>

    <section class="transforms">
      <h4 class="section-head">Transforms</h4>
      <TransformStack />
    </section>

    <DefectModal
      open={flaggedResult === result}
      {result}
      onclose={() => (flaggedResult = null)}
    />
  {/if}
</section>

<style>
  .head {
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  /* wyrd-2ien: the name moved into the cards below; this row now holds the
     pipeline-running indicator + the "report defective" action, kept top-right. */
  .head-top {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    margin-bottom: 10px;
  }
  /* wyrd-2ien: the era + modern cards sit side by side — a 2-col grid so the
     guide content can't wrap them onto separate rows; collapses to one column
     only when the inspector is genuinely narrow. A single (non-era) card fills
     the row. */
  .name-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px 24px;
    align-items: start;
    margin-top: 4px;
  }
  .flag {
    flex-shrink: 0;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-muted);
    cursor: pointer;
    font: inherit;
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 3px;
    white-space: nowrap;
  }
  .flag:hover {
    color: #e06c6c;
    border-color: #e06c6c;
  }
  .flag:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .pending-flag {
    font-size: 14px;
    color: var(--fg-muted);
    margin-left: 4px;
  }
  .provenance {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--fg-muted);
    font-style: italic;
  }
  .provenance .orig {
    color: var(--fg);
    font-style: normal;
  }
  .section-head {
    font-size: 11px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 12px;
  }
  .transforms {
    margin-top: 24px;
  }
  .placeholder {
    color: var(--fg-muted);
    font-style: italic;
    line-height: 1.5;
  }
</style>

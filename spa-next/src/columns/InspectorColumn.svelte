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
  import MorphemeCard from '../components/MorphemeCard.svelte';
  import TransformStack from '../components/TransformStack.svelte';
  import DefectModal from '../components/DefectModal.svelte';
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
  // synchronously. When the inspected result changes (selection,
  // deselection, re-roll) the modal closes in the same render — no $effect,
  // so no one-frame flash / double-render from effect-after-DOM timing.
  let flaggedResult = $state(null);

  const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');
  const norm = (s) => stripDashes(s).toLowerCase();
  // A string carries a diacritic if NFD-decomposing it yields combining
  // marks (i.e. it changes under decomposition).
  const hasAccent = (s) => !!s && s.normalize('NFD') !== s;

  // wyrd-2b50 follow-up: the bundle's generated `usage` is often the
  // lossy ASCII surface ("hy"), while the etymon's renderings carry the
  // accented original_script ("hȳ"). Return the accented surface —
  // grafted onto the usage's dash markers — when a rendering for this
  // morpheme's own surface supplies one. Null otherwise.
  function accentedUsage(m) {
    const u = norm(m.usage);
    if (!u) return null;
    const R = m.renderings || {};
    for (const lang of Object.keys(R)) {
      for (const form of Object.keys(R[lang])) {
        const os = R[lang][form].original_script;
        if (os && norm(form) === u && hasAccent(os)) {
          const lead = m.usage.match(/^-+/)?.[0] || '';
          const trail = m.usage.match(/-+$/)?.[0] || '';
          return lead + stripDashes(os) + trail;
        }
      }
    }
    return null;
  }

  function upgradeAccents(mbw) {
    let changed = false;
    const words = mbw.map((word) =>
      word.map((m) => {
        const acc = accentedUsage(m);
        if (acc && acc !== m.usage) {
          changed = true;
          return { ...m, usage: acc };
        }
        return m;
      }),
    );
    return { words, changed };
  }

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

  // The rendering slot (ipa / reader_pronunciation) for a morpheme's
  // CURRENT usage — matched against either the plain form key or its
  // accented original_script, since a swapped usage carries the accent.
  function renderingForUsage(m) {
    const u = stripDashes(m.usage);
    const renderings = m.renderings || {};
    for (const lang of Object.keys(renderings)) {
      for (const form of Object.keys(renderings[lang])) {
        const slot = renderings[lang][form];
        if (u === stripDashes(form) || u === stripDashes(slot.original_script)) {
          return slot;
        }
      }
    }
    return null;
  }

  // wyrd-2b50 follow-up: a pronunciation guide at the top, where it
  // matters — instead of only buried per-form in the cards. Rendered
  // PER-MORPHEME and aligned (surface over its reader-pronunciation /
  // IPA) rather than as one joined string, because pronunciation
  // coverage is sparse: most generated morphemes carry no rendering,
  // so a joined string would silently show one morpheme's sound as if
  // it were the whole name. Gaps render as a dim '·' so the guide is
  // honest about what it knows. Reflects swaps live (displayState is
  // post-pipeline). Hidden entirely when no morpheme has any data.
  let hasPronunciation = $derived(
    (displayState?.morphemes_by_word || [])
      .flat()
      .some((m) => {
        const s = renderingForUsage(m);
        return s?.reader_pronunciation || s?.ipa;
      }),
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
        <h3 class="name">
          {displayState.name}
          {#if pipeline.isRunning}
            <span class="pending-flag" title="pipeline running">…</span>
          {/if}
        </h3>
        <button
          type="button"
          class="flag"
          onclick={() => (flaggedResult = result)}
          title="Report defective"
        ><span aria-hidden="true">⚑</span> Report defective</button>
      </div>
      {#if displayState.morphemes_by_word?.length > 0}
        <p class="breakdown">
          {displayState.morphemes_by_word
            .map((word) => word.map((m) => m.usage).join(' '))
            .filter((s) => s.trim())
            .join(' · ')}
        </p>
      {/if}
      {#if hasPronunciation}
        <div class="pronunciation" aria-label="pronunciation guide">
          {#each displayState.morphemes_by_word as word}
            <span class="pron-word">
              {#each word as m}
                {#if m.usage?.trim()}
                  {@const slot = renderingForUsage(m)}
                  <span class="pron-col">
                    <span class="pron-surface">{m.usage}</span>
                    <span class="pron-reader"
                      >{slot?.reader_pronunciation || '·'}</span>
                    {#if slot?.ipa}
                      <span class="pron-ipa">{slot.ipa}</span>
                    {/if}
                  </span>
                {/if}
              {/each}
            </span>
          {/each}
        </div>
      {/if}
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
  /* wyrd-z3fl: name on the left, "report defective" action on the right. */
  .head-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
  }
  .name {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: var(--fg);
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
  .breakdown {
    margin: 6px 0 0;
    font-size: 12px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
  /* wyrd-2b50 follow-up: per-morpheme aligned pronunciation guide —
     each surface over its reader-pronunciation (accent) + IPA, words
     spaced apart, gaps shown as a dim '·'. */
  .pronunciation {
    margin: 10px 0 0;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
  }
  .pron-word {
    display: flex;
    gap: 10px;
  }
  .pron-col {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1px;
  }
  .pron-surface {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 12px;
    color: var(--fg);
  }
  .pron-reader {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--accent);
  }
  .pron-ipa {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 10px;
    color: var(--fg-muted);
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

<script>
  // wyrd-tz35: "Share workspace" button in col 3. Generates a URL
  // with the full workspace encoded in the `?s=` param, copies it
  // to the clipboard. Recipient pastes URL → SPA's boot-restore
  // (App.svelte onMount) detects the param + restores all 3
  // columns to the shared state.
  //
  // Distinct from the save button (which writes to localStorage):
  // share is one-shot transfer, not persistence. The two share the
  // same payload shape so future "save AND share" combos are easy.
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { encodeWorkspace, buildShareUrl } from '../lib/shareLink.js';

  let shareNote = $state('');

  async function share() {
    const r = appState.currentResult;
    if (!r) return;
    const encoded = encodeWorkspace({
      generator: appState.resultsGenerator,
      params: appState.currentParams,
      seed: appState.seed,
      original: {
        name: r.result,
        morphemes_by_word: r.morphemes_by_word || [],
        explanation: r.explanation || '',
      },
      pipeline: pipeline.steps.map((s) => ({
        kind: s.kind,
        params: { ...s.params },
      })),
    });
    const url = buildShareUrl(encoded);
    try {
      await navigator.clipboard.writeText(url);
      shareNote = `Link copied (${url.length} chars)`;
    } catch (err) {
      // Clipboard permission denied — surface the URL inline so the
      // user can copy it manually.
      shareNote = `Copy failed; URL: ${url.slice(0, 80)}…`;
      console.warn('share: clipboard write failed', err);
    }
    setTimeout(() => {
      shareNote = '';
    }, 3000);
  }
</script>

<button class="share" type="button" onclick={share}>
  🔗 Share workspace link
</button>
{#if shareNote}
  <p class="note">{shareNote}</p>
{/if}

<style>
  .share {
    background: transparent;
    color: var(--fg);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    width: 100%;
    text-align: left;
    margin-top: 4px;
  }
  .share:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .share:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .note {
    margin: 6px 0 0;
    font-size: 11px;
    color: var(--accent);
    font-style: italic;
  }
</style>

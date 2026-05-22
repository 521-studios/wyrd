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
  let fallbackUrl = $state('');

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
      fallbackUrl = '';
      setTimeout(() => {
        shareNote = '';
      }, 3000);
    } catch (err) {
      // wyrd-tz35 round 2 (frontend MED): clipboard write failed
      // (non-secure context, permission denied). Render the FULL
      // URL in an auto-selecting readonly textarea so the user can
      // ⌘C / Ctrl-C it manually. No timeout — leave visible until
      // the user dismisses by clicking Share again.
      shareNote = 'Copy failed — select the link below + ⌘C / Ctrl-C:';
      fallbackUrl = url;
      console.warn('share: clipboard write failed', err);
    }
  }

  function selectAll(node) {
    node.focus();
    node.select();
  }
</script>

<button class="share" type="button" onclick={share}>
  🔗 Share workspace link
</button>
{#if shareNote}
  <p class="note">{shareNote}</p>
{/if}
{#if fallbackUrl}
  <textarea
    class="fallback"
    readonly
    rows="3"
    use:selectAll
  >{fallbackUrl}</textarea>
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
  .fallback {
    width: 100%;
    margin-top: 6px;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 8px;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
    line-height: 1.4;
    resize: vertical;
    word-break: break-all;
  }
</style>

<script>
  // wyrd-mf2u: a small, reusable presentational card — renders ONE form's
  // breakdown (the surface + its pronunciation + an optional role/language
  // chip). It takes only what it displays, so the same component composes:
  //   - one instance for a plain (modern) morpheme, OR
  //   - an era form instance + a modern instance side-by-side for an era
  //     render (the period form that's in the name, anchored by the modern).
  // Pure presentation: no etymology/swap machinery (that stays in the parent
  // MorphemeCard). Keeping this dumb is what makes it reusable + robust.
  let { form, pron = null, label = '', primary = false } = $props();

  // pron: { reader_pronunciation?, ipa?, original_script? } | null
  let reader = $derived(pron?.reader_pronunciation || '');
  let ipa = $derived(pron?.ipa || '');
</script>

<div class="morpheme-breakdown" class:primary>
  <span class="bd-form">{form}</span>
  {#if label}<span class="bd-label">{label}</span>{/if}
  {#if reader || ipa}
    <span class="bd-pron">
      {#if reader}<span class="bd-reader">{reader}</span>{/if}
      {#if ipa}<span class="bd-ipa">{ipa}</span>{/if}
    </span>
  {/if}
</div>

<style>
  .morpheme-breakdown {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .bd-form {
    font-size: 15px;
    font-weight: 700;
    color: var(--fg);
    font-variant-numeric: tabular-nums;
    word-break: break-word;
  }
  .primary .bd-form {
    color: var(--accent);
    font-size: 16px;
  }
  .bd-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--fg-muted);
  }
  .bd-pron {
    font-size: 12px;
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .bd-reader {
    font-weight: 600;
  }
</style>

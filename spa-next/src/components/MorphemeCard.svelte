<script>
  // wyrd-yxf6: one card per morpheme in InspectorColumn. Shows
  // usage as the card header + meanings + tags + per-source-language
  // panels carrying form / IPA / reader_pronunciation / original_script /
  // dialect (the wyrd-cp2d + wyrd-03cx data that the morphemes_by_word
  // / renderings dicts now carry through the API envelope).
  //
  // wyrd-hpjg (PR #5) will make the form chips clickable + open a
  // popover of cluster siblings for direct manipulation.
  import { languageLabel } from '../lib/languageLabels.js';

  let { morpheme } = $props();

  // Sources is a sparse dict {lang: [form, ...]}. Order isn't
  // guaranteed; we sort with the most common British-place-name
  // languages first so OE / Norse / Celtic land at the top.
  const LANG_PRIORITY = [
    'old_english',
    'old _english',
    'old_scandinavian',
    'old_scandanavian',
    'celtic_mix',
    'old_french',
    'norman_french',
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
      const forms = new Set([
        ...(sources[lang] || []),
        ...Object.keys(renderings[lang] || {}),
      ]);
      return [lang, [...forms]];
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

  // For each form within a source language, pull the matching
  // renderings slot (ipa / reader_pronunciation / original_script /
  // dialect) so the panel can render the four-column data with one
  // template iteration.
  function renderingFor(lang, form) {
    return morpheme.renderings?.[lang]?.[form] || {};
  }
</script>

<article class="morpheme">
  <header>
    <span class="usage">{morpheme.usage}</span>
    {#if morpheme.rendered && morpheme.rendered !== morpheme.usage}
      <span class="rendered" title="D18/D8 substituted variant">
        → {morpheme.rendered}
      </span>
    {/if}
  </header>

  {#if morpheme.meanings?.length}
    <p class="meanings">{morpheme.meanings.join(', ')}</p>
  {/if}

  {#if morpheme.tags?.length}
    <p class="tags">
      {#each morpheme.tags as tag}
        <span class="tag">{tag}</span>
      {/each}
    </p>
  {/if}

  {#each orderedSourceEntries as [lang, forms] (lang)}
    <section class="source-lang">
      <h4>{languageLabel(lang)}</h4>
      <table class="forms">
        <tbody>
          {#each forms as form (form)}
            {@const r = renderingFor(lang, form)}
            <tr>
              <td class="form">
                {#if r.original_script && r.original_script !== form}
                  <span class="original" title="original script"
                    >{r.original_script}</span>
                  <span class="translit">({form})</span>
                {:else}
                  {form}
                {/if}
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
  .usage {
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }
  .rendered {
    font-size: 12px;
    color: var(--fg-muted);
  }
  .meanings {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--fg);
    line-height: 1.5;
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
  .original {
    font-family: serif;
    font-size: 13px;
  }
  .translit {
    color: var(--fg-muted);
    font-size: 11px;
    margin-left: 4px;
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
</style>

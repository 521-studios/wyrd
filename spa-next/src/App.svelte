<script>
  // wyrd-z3lp: 3-column layout shell. No functionality this PR;
  // subsequent PRs (wyrd-hcmc, wyrd-yxf6, wyrd-kppy, ...) wire each
  // column in turn. Column components are stubs that render their
  // role + a placeholder; this PR's bar is "the layout exists and
  // builds clean."
  import ConfigureColumn from './columns/ConfigureColumn.svelte';
  import OutputColumn from './columns/OutputColumn.svelte';
  import InspectorColumn from './columns/InspectorColumn.svelte';
</script>

<div class="layout">
  <ConfigureColumn />
  <OutputColumn />
  <InspectorColumn />
</div>

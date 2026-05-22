<script>
  // wyrd-14hn: fallback workspace for generators that don't have a
  // dedicated layout registered. Form on the left, result list on
  // the right — the minimum viable generator UX.
  //
  // Today: kenning-explain + kenning-era-map fall here (the
  // workspace registry only points 'kenning' + 'kenning-rewind' at
  // KenningWorkspace). Their current output (a list of decompositions
  // / an era-render table) renders adequately via OutputColumn's
  // result list; dedicated workspaces can graduate them later.
  //
  // No mobile drawer logic — the 2-column shape collapses to a
  // single stacked layout at narrow widths via CSS.
  //
  // wyrd-14hn round 2 (frontend MED): no openMenu() exported.
  // App.svelte checks `workspaceRef?.openMenu` to decide whether
  // to render the Header's hamburger — workspaces without a drawer
  // don't expose the method, and the hamburger doesn't render. Pre-
  // fix DefaultWorkspace had a no-op openMenu stub which caused
  // the hamburger to render but tap nothing.
  import ConfigureColumn from '../columns/ConfigureColumn.svelte';
  import OutputColumn from '../columns/OutputColumn.svelte';
</script>

<div class="layout">
  <ConfigureColumn />
  <OutputColumn />
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: var(--col-1-width) 1fr;
    height: calc(100dvh - 52px);
  }

  /* At narrow widths, stack the two columns. No fixed-position
     drawer machinery — the default workspace assumes its content
     is light enough to scroll naturally. */
  @media (max-width: 899px) {
    .layout {
      display: flex;
      flex-direction: column;
      height: calc(100dvh - 48px);
      overflow-y: auto;
    }
  }
</style>

<script>
  // wyrd-kppy: the transform stack — list of TransformStep cards +
  // the TransformPalette below. Step keying uses index + kind so
  // adding/removing in the middle doesn't trip up Svelte's keyed
  // each-block invalidation.
  import { pipeline } from '../lib/pipeline.svelte.js';
  import TransformStep from './TransformStep.svelte';
  import TransformPalette from './TransformPalette.svelte';
</script>

<div class="stack">
  {#each pipeline.steps as step, i (step.id)}
    <TransformStep {step} index={i} />
  {/each}
  <TransformPalette />
</div>

<style>
  .stack {
    margin-top: 4px;
  }
</style>

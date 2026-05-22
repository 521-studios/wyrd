// wyrd-14hn: per-generator workspace registry.
//
// User design call (2026-05-22): generators may want totally
// different interfaces. Today's kenning-family generators all use
// the same morpheme inspector + pipeline shape (KenningWorkspace);
// future non-kenning generators (NPC stat blocks, tavern menus,
// maps, loot tables) will register their own workspace components.
//
// Default fallback (DefaultWorkspace) is the minimum viable
// generator UX: form on the left, result list on the right, no
// inspector. Anything that doesn't have a registered workspace
// gets it.
//
// Adding a new workspace:
//   1. Build the component as src/workspaces/<Name>Workspace.svelte
//   2. Import + add an entry below
// The registry is the single discovery point.

import KenningWorkspace from '../workspaces/KenningWorkspace.svelte';
import DefaultWorkspace from '../workspaces/DefaultWorkspace.svelte';

const WORKSPACES = {
  kenning: KenningWorkspace,
  // kenning-rewind shares the morpheme inspector + pipeline shape;
  // the workspace re-uses KenningWorkspace's columns. Distinct
  // generator backend, same SPA workspace.
  'kenning-rewind': KenningWorkspace,
  // kenning-explain + kenning-era-map fall back to DefaultWorkspace
  // for now; can graduate to dedicated workspaces when the UX is
  // designed (kenning-explain wants a canonical-reading panel;
  // kenning-era-map wants a multi-name table).
};

export function workspaceFor(generatorName) {
  return WORKSPACES[generatorName] || DefaultWorkspace;
}

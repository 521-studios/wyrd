// wyrd-34tn: localStorage-backed bookmark store. User-decided design
// (2026-05-21 session): a "save" captures the WHOLE current workspace
// — generator + params + seed + original generated name + current
// transform pipeline. Loading a saved entry restores all 3 columns
// to that exact state. The "flat list now, tags later" call also
// from the design session.
//
// Persistence model:
//   - localStorage key: `wyrd.saved.v1` (versioned so a future schema
//     change is migration-detectable; pre-v1 data simply doesn't load)
//   - JSON shape: { schema_version: 1, entries: [SavedEntry, ...] }
//   - SavedEntry has `schema_version` too so per-entry migration is
//     possible (e.g., a v2 entry coexists with v1 entries during a
//     gradual schema bump)
//   - All mutations write-through immediately (no debounce); list
//     size is bounded by browser quota (~5MB → 250-500 saves at
//     typical payload size)
//
// Reactivity: a Svelte 5 $state-class singleton — components import
// `savedStore` and any property read becomes reactive automatically.

const STORAGE_KEY = 'wyrd.saved.v1';
const CURRENT_SCHEMA = 1;

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    // wyrd-34tn: future schema bumps land migration logic here.
    // For v1, accept entries with matching or unset version; drop
    // entries with a NEWER schema_version (user downgraded the app
    // — better than corrupting their data).
    if (parsed.schema_version !== CURRENT_SCHEMA) {
      console.warn(
        `saved-store: schema ${parsed.schema_version} != ${CURRENT_SCHEMA}; treating as empty`,
      );
      return [];
    }
    return Array.isArray(parsed.entries) ? parsed.entries : [];
  } catch (err) {
    console.warn('saved-store: load failed', err);
    return [];
  }
}

function writeToStorage(entries) {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ schema_version: CURRENT_SCHEMA, entries }),
    );
  } catch (err) {
    // Quota exceeded is the most likely failure. Surface it loudly
    // but don't crash — the user's session keeps working, they just
    // don't get a persisted save.
    console.error('saved-store: write failed (quota?)', err);
  }
}

/**
 * Generate a stable id for a new save. Uses crypto.randomUUID when
 * available (modern browsers) and falls back to a timestamp+random
 * shape for older environments. Saved-entry ids are user-visible
 * only via the export/import JSON, so non-UUID fallback is fine.
 */
function newId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `save-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

class SavedStore {
  // Reactive list of saved entries. Loaded from localStorage at
  // construction; every mutation writes through.
  entries = $state(loadFromStorage());

  /** Add a new save. Returns the new entry's id. Accepts a payload
   *  with everything needed to rehydrate the workspace (generator,
   *  params, seed, original, pipeline) + an optional user label. */
  add({ label, generator, params, seed, original, pipeline }) {
    const entry = {
      id: newId(),
      schema_version: CURRENT_SCHEMA,
      saved_at: new Date().toISOString(),
      label: label || (original?.name ?? '(unlabeled)'),
      generator,
      params,
      seed,
      original,
      pipeline,
    };
    this.entries = [entry, ...this.entries];
    writeToStorage(this.entries);
    return entry.id;
  }

  /** Update a save's user-editable label. */
  rename(id, label) {
    this.entries = this.entries.map((e) =>
      e.id === id ? { ...e, label } : e,
    );
    writeToStorage(this.entries);
  }

  /** Remove a save by id. */
  remove(id) {
    this.entries = this.entries.filter((e) => e.id !== id);
    writeToStorage(this.entries);
  }

  /** Get a save by id (for the load-back-into-workspace flow). */
  get(id) {
    return this.entries.find((e) => e.id === id) || null;
  }

  /** True if an entry exists for this original-name AND generator
   *  combination — used by the star icon to render filled/empty
   *  based on whether the result is already saved. */
  isSaved({ generator, originalName }) {
    return this.entries.some(
      (e) => e.generator === generator && e.original?.name === originalName,
    );
  }

  /** Find the id of a saved entry by (generator, originalName).
   *  Used to support star-toggle: clicking a filled star removes
   *  the existing save instead of adding a duplicate. */
  findId({ generator, originalName }) {
    const e = this.entries.find(
      (m) => m.generator === generator && m.original?.name === originalName,
    );
    return e?.id || null;
  }

  /** Export all entries as a downloadable JSON blob (the same
   *  shape we persist). Caller triggers a download via URL.create. */
  exportJSON() {
    return JSON.stringify(
      { schema_version: CURRENT_SCHEMA, entries: this.entries },
      null,
      2,
    );
  }

  /** Import entries from a JSON string. Merges with existing —
   *  duplicate-id entries are skipped (the existing entry wins).
   *  Returns { added, skipped, error } so the UI can surface the
   *  result. */
  importJSON(json) {
    let parsed;
    try {
      parsed = JSON.parse(json);
    } catch (err) {
      return { added: 0, skipped: 0, error: `parse failed: ${err.message}` };
    }
    if (parsed.schema_version !== CURRENT_SCHEMA) {
      return {
        added: 0,
        skipped: 0,
        error: `unsupported schema_version ${parsed.schema_version} (expected ${CURRENT_SCHEMA})`,
      };
    }
    if (!Array.isArray(parsed.entries)) {
      return { added: 0, skipped: 0, error: 'entries is not an array' };
    }
    const existingIds = new Set(this.entries.map((e) => e.id));
    const fresh = parsed.entries.filter((e) => e.id && !existingIds.has(e.id));
    const skipped = parsed.entries.length - fresh.length;
    if (fresh.length > 0) {
      this.entries = [...fresh, ...this.entries];
      writeToStorage(this.entries);
    }
    return { added: fresh.length, skipped, error: null };
  }
}

export const savedStore = new SavedStore();

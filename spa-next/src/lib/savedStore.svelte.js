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
    // wyrd-34tn round 2 (clarity P3): strict-equality check — drops
    // entries from older / newer / unset schemas. Future migrations
    // land here: branch on parsed.schema_version, transform entries,
    // re-tag with CURRENT_SCHEMA. v1 has nothing to migrate.
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
    return null;
  } catch (err) {
    // wyrd-34tn round 2 (frontend MED): return the error to the
    // caller so the UI can surface "save failed (quota?)". Quota
    // exceeded, Safari private mode, and disabled storage all
    // surface as setItem throws.
    console.error('saved-store: write failed', err);
    return err;
  }
}

// wyrd-34tn round 2 (frontend HIGH): deep-clone via JSON round-trip
// so persisted entries don't alias live $state objects (params,
// original, pipeline). Subsequent edits to the form / pipeline don't
// silently mutate the in-memory entry; the next writeToStorage doesn't
// corrupt the saved JSON.
//
// wyrd-34tn round 3: NOT structuredClone — Svelte 5 $state values are
// Proxy-wrapped and structuredClone throws on Proxies. JSON.stringify
// works because it iterates enumerable properties via the proxy and
// extracts plain values. Since we're already persisting as JSON, the
// round-trip is the right shape anyway.
function deepClone(obj) {
  if (obj === null || obj === undefined) return obj;
  return JSON.parse(JSON.stringify(obj));
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
   *  params, original, pipeline) + an optional user label. No seed — a
   *  save freezes the actual result, it does not re-derive it. Older
   *  entries may still carry a seed field; it is simply ignored on load.
   *  wyrd-34tn round 2 (HIGH x2): every nested payload field is
   *  deep-cloned so subsequent edits to the live $state forms /
   *  pipeline don't alias-mutate the saved entry. */
  add({ label, generator, params, original, pipeline }) {
    const entry = {
      id: newId(),
      schema_version: CURRENT_SCHEMA,
      saved_at: new Date().toISOString(),
      label: label || (original?.name ?? '(unlabeled)'),
      generator,
      params: deepClone(params),
      original: deepClone(original),
      pipeline: deepClone(pipeline),
    };
    this.entries = [entry, ...this.entries];
    // wyrd-34tn round 3 (Gemini HIGH): propagate the write error to
    // the caller so the UI can show "save failed (quota?)". In-memory
    // state is still updated (the session keeps working); only the
    // disk persist failed.
    const err = writeToStorage(this.entries);
    if (err) return { id: null, error: err };
    return { id: entry.id, error: null };
  }

  /** Update a save's user-editable label. Returns the write error
   *  (or null on success) so callers can surface quota failures. */
  rename(id, label) {
    this.entries = this.entries.map((e) =>
      e.id === id ? { ...e, label } : e,
    );
    return writeToStorage(this.entries);
  }

  /** Remove a save by id. Returns the write error (or null). */
  remove(id) {
    this.entries = this.entries.filter((e) => e.id !== id);
    return writeToStorage(this.entries);
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
    // A JSON literal that parses to a non-object — `null` most notably, but also
    // a number / string / array — has no `schema_version`. `null.schema_version`
    // would THROW (TypeError) past this method's "returns {added, skipped, error}"
    // contract, the way an imported file of literally `null` (or a truncated
    // export) does. Reject it as a clean error instead of crashing the importer.
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { added: 0, skipped: 0, error: 'not a saved-store export (expected a JSON object)' };
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
    // wyrd-34tn round 2 (frontend LOW): validate per-entry shape so
    // malformed entries don't crash SavedList.load on entry.original.name.
    // Minimum shape: id + original.name. Drop anything missing those.
    const valid = parsed.entries.filter(
      (e) => e && typeof e.id === 'string' && e.original && typeof e.original.name === 'string',
    );
    const existingIds = new Set(this.entries.map((e) => e.id));
    const fresh = valid.filter((e) => !existingIds.has(e.id));
    const skipped = parsed.entries.length - fresh.length;
    if (fresh.length > 0) {
      this.entries = [...fresh, ...this.entries];
      writeToStorage(this.entries);
    }
    return { added: fresh.length, skipped, error: null };
  }
}

export const savedStore = new SavedStore();

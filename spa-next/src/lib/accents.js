// wyrd-de5t: shared morpheme accent-upgrade helper.
//
// The bundle's generated `usage` is often the lossy ASCII surface ("by",
// "hy"), while the etymon's renderings carry the accented original_script
// ("bȳ", "hȳ"). `accentedUsage` returns the accented surface — grafted onto
// the usage's dash markers — when a rendering for this morpheme's OWN
// surface supplies one; null otherwise.
//
// Extracted from InspectorColumn (col 3, which already showed accents) so
// the Output column (col 2) can render the same accented surfaces instead
// of the raw ASCII.

const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');
const norm = (s) => stripDashes(s).toLowerCase();
// A string carries a diacritic if NFD-decomposing it yields combining
// marks (i.e. it changes under decomposition).
const hasAccent = (s) => !!s && s.normalize('NFD') !== s;

/**
 * Return this morpheme's usage upgraded to its accented original_script
 * surface when a rendering for the same (dash-stripped, lowercased) form
 * supplies one with a diacritic; otherwise null. Dash markers on the
 * original usage ("-by") are preserved on the result ("-bȳ").
 */
export function accentedUsage(morph) {
  const u = norm(morph.usage);
  if (!u) return null;
  const R = morph.renderings || {};
  for (const lang of Object.keys(R)) {
    for (const form of Object.keys(R[lang])) {
      const os = R[lang][form].original_script;
      if (os && norm(form) === u && hasAccent(os)) {
        const lead = morph.usage.match(/^-+/)?.[0] || '';
        const trail = morph.usage.match(/-+$/)?.[0] || '';
        return lead + stripDashes(os) + trail;
      }
    }
  }
  return null;
}

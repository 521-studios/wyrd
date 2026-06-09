// wyrd-yxf6: display labels for source-language keys used in the
// bundle's `sources` and `renderings` dicts. The keys are snake_case
// identifiers ("old_english", "celtic_mix"); the SPA wants
// human-readable labels ("Old English", "Celtic"). Keeping the map
// here so other panels (PR #5 morpheme-sibling popover) reuse it.

const LANGUAGE_LABELS = {
  old_english: 'Old English',
  middle_english: 'Middle English',
  modern_english: 'Modern English',
  old_scandinavian: 'Old Norse',
  old_scandanavian: 'Old Norse', // legacy misspelling preserved by bundle
  'old _english': 'Old English', // ditto (stray space)
  old_french: 'Old French',
  middle_french: 'Middle French',
  modern_french: 'French',
  norman_french: 'Norman French',
  celtic_mix: 'Celtic',
  welsh: 'Welsh',
  irish: 'Irish',
  scottish_gaelic: 'Scottish Gaelic',
  breton: 'Breton',
  latin: 'Latin',
  vulgar_latin: 'Vulgar Latin',
  germanic: 'Germanic',
  proto_germanic: 'Proto-Germanic',
  greek: 'Greek',
  // wyrd-qc0g: the era_grid keys its stages by the HYPHENATED canonical
  // language tags from era/cells.py (era_reflexes), not the underscored
  // `sources`/`renderings` keys above. Same human labels.
  'old-english': 'Old English',
  'middle-english': 'Middle English',
  'modern-english': 'Modern',
  'old-norse': 'Old Norse',
  'old-welsh': 'Old Welsh',
  'middle-welsh': 'Middle Welsh',
  'old-irish': 'Old Irish',
  'middle-irish': 'Middle Irish',
  'old-french': 'Old French',
};

export function languageLabel(key) {
  if (!key) return '';
  if (LANGUAGE_LABELS[key]) return LANGUAGE_LABELS[key];
  // Fallback: title-case the key (snake_case OR hyphenated) so an unknown
  // language shows as "Some Language" not "some_language" / "some-language".
  // Better than a hardcoded "Unknown" — the bundle's language space grows.
  return key
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

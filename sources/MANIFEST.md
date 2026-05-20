# Place-name etymology corpus

The corpus drives the `mine-toponym-mentions-staged` cascade
(`wyrd kenning lexicon mine-toponym-mentions-staged`). Each `<source_id>.txt`
under this directory is a plain-text scholarly source whose mined output
lands in `data/mining/phase2/<source_id>.jsonl`. The `source_id` is the
filename without `.txt`; it becomes a foreign key in downstream artifacts —
don't rename a source after it has been mined.

Most sources are Internet Archive `_djvu.txt` OCR of pre-1930
public-domain works (US 95-year rule). Post-1930 entries are eligible
too: we extract facts (attested forms, dates, citations), not creative
expression, and facts aren't copyrightable (*Feist v. Rural Telephone*,
1991).

## England — general

- `taylor_1893_words_and_places.txt` — Isaac Taylor, *Words and Places: or,
  Etymological Illustrations of History, Ethnology, and Geography* (1893 ed.).
  Foundational comparative work on toponymy.
- `mcclure_1910_british_place_names.txt` — Edmund McClure, *British Place-Names
  in their Historical Setting* (1910).
- `johnston_1915_place_names_of_england_and_wales.txt` — James B. Johnston,
  *The Place-Names of England and Wales* (1915).
- `mawer_1922_place_names_and_history.txt` — Allen Mawer, *Place-Names and
  History* (1922).
- `mawer_stenton_1924_introduction_to_survey.txt` — A. Mawer & F.M. Stenton,
  *Introduction to the Survey of English Place-Names* (1924).
- `ekwall_1928_river_names.txt` — Eilert Ekwall, *English River-Names* (1928).
  Single-theme monograph; high morpheme density for water-related elements.
- `mawer_1924_chief_elements.txt` — Allen Mawer, *The Chief Elements Used in
  English Place-Names* (1924).

## England — regional

- `skeat_1901_cambridgeshire.txt` — W.W. Skeat, *The Place-Names of
  Cambridgeshire* (1901).
- `skeat_1904_hertfordshire.txt` — W.W. Skeat, *The Place-Names of
  Hertfordshire* (1904).
- `skeat_1904_huntingdonshire.txt` — W.W. Skeat, *The Place-Names of
  Huntingdonshire* (1904).
- `skeat_1906_bedfordshire.txt` — W.W. Skeat, *The Place-Names of Bedfordshire*
  (1906).
- `skeat_1911_berkshire.txt` — W.W. Skeat, *The Place-Names of Berkshire*
  (1911).
- `skeat_1913_suffolk.txt` — W.W. Skeat, *The Place-Names of Suffolk* (1913).
- `mawer_1920_northumberland_durham.txt` — Allen Mawer, *The Place-Names of
  Northumberland and Durham* (1920).
- `mawer_stenton_1925_buckinghamshire.txt` — A. Mawer & F.M. Stenton,
  *The Place-Names of Buckinghamshire* (1925). EPNS vol II.
- `mawer_stenton_1927_worcestershire.txt` — A. Mawer & F.M. Stenton,
  *The Place-Names of Worcestershire* (1927). EPNS vol IV.
- `smith_1928_north_riding_yorkshire.txt` — A.H. Smith, *The Place-Names of
  the North Riding of Yorkshire* (1928). EPNS vol V.
- `gover_stenton_1936_warwickshire.txt` — J.E.B. Gover, A. Mawer & F.M. Stenton,
  *The Place-Names of Warwickshire* (1936). EPNS vol XIII.
- `smith_1937_east_riding_yorkshire.txt` — A.H. Smith, *The Place-Names of
  the East Riding of Yorkshire and York* (1937). EPNS vol XIV; with the
  1928 (North) above and the 1961/1986 (West) volumes below, completes
  the Yorkshire-Ridings trio.
- `smith_1961_west_riding_yorkshire_pt1.txt` — A.H. Smith, *The Place-Names
  of the West Riding of Yorkshire*, pt 1 (1961). EPNS vol XXX. West Riding
  is an eight-part EPNS survey; this scan covers pt 1 — Lower Strafforth,
  Upper Strafforth, and Staincross wapentakes.
- `smith_1986_west_riding_yorkshire_pt2.txt` — A.H. Smith, *The Place-Names
  of the West Riding of Yorkshire*, pt 2 (EPNS reissue 1986). EPNS vol XXXI
  — Osgoldcross and Agbrigg wapentakes.
- `roberts_1914_sussex.txt` — Richard G. Roberts, *The Place-Names of Sussex*
  (Cambridge University Press, 1914). Earlier non-EPNS Sussex volume,
  superseded by the 1930 EPNS edition but useful as an independent witness.
- `mawer_stenton_1930_sussex.txt` — A. Mawer & F.M. Stenton, *The Place-Names
  of Sussex* (1930). EPNS vols VI–VII (Sussex part 1 was 1929, part 2 1930;
  this scan covers the combined work).
- `ekwall_1922_lancashire.txt` — Eilert Ekwall, *The Place-Names of Lancashire*
  (1922).
- `wyld_hirst_1911_lancashire.txt` — H.C. Wyld & T.O. Hirst, *The Place-Names
  of Lancashire: Their Origin and History* (1911).
- `moorman_1910_west_riding_yorkshire.txt` — F.W. Moorman, *Place-Names of the
  West Riding of Yorkshire* (1910).
- `goodall_1914_sw_yorkshire.txt` — Armitage Goodall, *Place-Names of
  South-West Yorkshire* (1914).
- `duignan_1902_staffordshire.txt` — W.H. Duignan, *Notes on Staffordshire
  Place-Names* (1902).
- `harrison_1898_liverpool.txt` — Henry Harrison, *The Place-Names of the
  Liverpool District* (1898).
- `lindkvist_1912_scandinavian_place_names.txt` — Harald Lindkvist,
  *Middle-English Place-Names of Scandinavian Origin* (1912).
- `zachrisson_1909_anglo_norman_influence.txt` — R.E. Zachrisson, *A
  Contribution to the Study of Anglo-Norman Influence on English Place-Names*
  (1909).
- `bannister_1871_cornish_names.txt` — Rev. J. Bannister, *A Glossary of
  Cornish Names* (1871). Cornish-Brythonic substrate corpus for Cornwall —
  distinct from the Anglo-Saxon / Old-Norse strata dominating most
  England-regional sources. Early dictionary-style alphabetical headwords.
- `bannister_1916_herefordshire.txt` — A.T. Bannister, *The Place-Names of
  Herefordshire* (Cambridge: Clay, 1916). Welsh-Marches county; Bannister
  explicitly characterizes Herefordshire as "the most thoroughly Normanized
  of all the English counties" — Skeat-style alphabetical-headword body
  with Norman lordship compounds (Acton Beauchamp, Allensmore <
  *Alan de Plokenet*) plus Welsh-substrate Celtic strata (`allt`, `dwr`,
  `ynys`). Targeted as the Norman-French corpus expansion via the
  Welsh-Marches angle (wyrd-hub).
- `sedgefield_1915_cumberland_westmorland.txt` — W.J. Sedgefield, *The
  Place-Names of Cumberland and Westmorland* (1915). North-West English
  counties; rich Old Norse / Cumbric / Old English stratification along
  the Anglo-Scandinavian frontier.
- `hill_1914_somerset.txt` — J.S. Hill, *The Place-Names of Somerset*
  (1914). South-West English county; Old English layered over Brythonic
  substrate.
- `horsley_1921_kent.txt` — J.W. Horsley, *Place Names in Kent* (1921).
  Kent place-names; complements the Anglo-Norman and Old-English elements
  prominent in South-East England. Sourced from Project Gutenberg
  (eBook #63263), not Internet Archive.
- `coates_2020_grimsby_cleethorpes.txt` — Richard Coates, *The Place-Names
  of Grimsby and Cleethorpes* (2020). Focused modern Lincolnshire study;
  Old Norse / Anglian density along the Humber estuary.

## Scotland

- `johnston_1892_place_names_of_scotland.txt` — James B. Johnston,
  *Place-Names of Scotland* (1892).
- `johnston_1904_stirlingshire.txt` — James B. Johnston, *The Place-Names of
  Stirlingshire* (1904).
- `watson_1904_ross_and_cromarty.txt` — W.J. Watson, *Place-Names of Ross and
  Cromarty* (1904).
- `watson_1926_celtic_place_names_of_scotland.txt` — W.J. Watson, *The History
  of the Celtic Place-Names of Scotland* (1926).
- `macbain_1922_highlands_and_islands.txt` — Alexander Macbain (ed. Watson),
  *Place Names, Highlands & Islands of Scotland* (1922).
- `gillies_1906_argyll.txt` — H.C. Gillies, *The Place-Names of Argyll* (1906).

## Wales

- `morgan_1887_wales_monmouthshire.txt` — Thomas Morgan, *Handbook of the
  Origin of Place-Names in Wales and Monmouthshire* (1887).
- `morgan_1912_wales.txt` — Thomas Morgan, *The Place-Names of Wales* (1912).

## Ireland

- `joyce_1875_irish_names_vol1.txt` — P.W. Joyce, *The Origin and History of
  Irish Names of Places*, vol. 1 (1875 ed.).
- `joyce_1875_irish_names_vol2.txt` — P.W. Joyce, *The Origin and History of
  Irish Names of Places*, vol. 2 (1875 ed., Dublin: McGlashan).
- `joyce_1898_irish_names_vol3.txt` — P.W. Joyce, *The Origin and History of
  Irish Names of Places*, vol. 3 (1898).
- `joyce_1913_irish_names_of_places.txt` — P.W. Joyce, *Irish Names of Places*
  (1913).

## Isle of Man

- `moore_1890_isle_of_man.txt` — A.W. Moore, *The Surnames and Place-Names of
  the Isle of Man* (1890).

## Brittany / France

- `quilgars_1906_loire_inferieure.txt` — Henri Quilgars, *Dictionnaire
  topographique du département de la Loire-Inférieure* (Paris:
  Imprimerie nationale, 1906). Brittany-adjacent toponymic dictionary with
  ~1020 alphabetical headword entries dense in `Plou-`, `Ker-`, `Tré-`,
  `Lan-`, `Pen-`, `Loc-`, `Pleu-` morphemes. Drives the Breton-register
  morpheme corpus expansion (wyrd-fmg) so generation can produce real
  French-Celtic place names rather than the existing English+Celtic+FR
  fallback.
- `longnon_1920_noms_de_lieu_france_v1.txt` — Auguste Longnon, *Les noms
  de lieu de la France: leur origine, leur signification, leurs
  transformations*, vol 1 (Paris: Champion, 1920). Treatise on Greek /
  Phoenician / Ligure / Iberian / Celtic substrate origins of French
  place-names; chapters subdivided by suffix (`-dunum`, `-duros`,
  `-briga`, `-asca`). Mined via Haiku 4.5: 67 parsed → 16 accepted, 27
  etymons touched (wyrd-efx).
- `longnon_1920_noms_de_lieu_france_v2.txt` — Same series, vol 2.
  Saint-name (`Saint-X` ← Latin saint name) and feudal-castle entries
  organized as numbered list. Parser yields 2,775 candidates but most
  are front-matter / OCR fragments; bulk-list segmentation needs
  wyrd-5af treatise-aware extractor before full mining (wyrd-ech).
- `loth_1890_chrestomathie_bretonne.txt` — Joseph Loth, *Chrestomathie
  bretonne (armoricain, gallois, cornique)* (Paris: E. Bouillon, 1890).
  Anthology of Middle Breton texts (mystery plays, manuscript prose)
  with embedded etymology commentary in dedicated 'Noms de Lieux et de
  Peuples' sections. Bulk-text shape blocks the alphabetical parser;
  needs wyrd-5af before mining (wyrd-mlu).
- `arbois_1890_recherches_propriete_fonciere.txt` — Henri d'Arbois de
  Jubainville, *Recherches sur l'origine de la propriété foncière et
  des noms de lieux habités en France (période celtique et période
  romaine)* (Paris: Thorin, 1890). Foundational Celtic-and-Roman-period
  French toponymy by the era's leading Celticist. Treatise shape with
  per-toponym sections (Aria, Artia, Rennius, etc.); 960 parser entries,
  40% accept on smoke; full mining in flight (wyrd-cmz).

## Modern open data

- `os_opennames/` — Ordnance Survey OpenNames (CSV, GB-wide, ~3.04M rows).
  Released under the Open Government Licence v3 (`Doc/licence.txt`). Schema
  header in `Doc/OS_Open_Names_Header.csv`; data split by 100 km grid square in
  `Data/*.csv`. Original zip retained as `opname_csv_gb.zip`. Useful columns
  for the kenning generator: `NAME1`, `NAME1_LANG`, `LOCAL_TYPE`,
  `COUNTY_UNITARY`, `COUNTRY`.

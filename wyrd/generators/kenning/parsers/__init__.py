"""The kenning ``parsers/`` subpackage — regex-based scholarly-OCR parsers.

Owns the source-text → ``ParsedEntry[]`` step of the mining pipeline.
Both parsers consume OCR'd 19th/20th-century public-domain reference
works and segment them into entry-shaped rows for the downstream LLM
extractors in ``extractors/``.

Modules:

* ``skeat.py`` (was ``skeat_parser.py``) — Skeat-format parser.
  Body shape: ``N. The suffix -X`` sections, each enumerating
  toponyms whose name ends in suffix X. Defines ``ParsedEntry`` /
  ``ParsedElement`` dataclasses + the section/TOC machinery.

* ``dictionary.py`` (was ``dictionary_parser.py``) — alphabetical
  + numbered-list parser. Body shape: ``Headword (Anc.Eng. form
  citation)... §1 etymology §2 forms``. Imports ``ParsedEntry`` and
  ``_shorten`` from ``.skeat``.

Shared-base extraction (proposed in wyrd-1j6b's design notes) is
deferred per rule-of-N: with N=2, the abstraction shape isn't yet
constrained enough to know which parts to abstract. The two parsers
have very different internal shapes (section/suffix-driven vs
headword/numbered-list-driven); a forced common base today would be
speculative. Revisit when a 3rd format-distinct parser lands.
"""

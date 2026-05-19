"""The ``wyrd kenning`` extractor family.

This package holds the LLM provider clients (Ollama via ``llm``,
``gemini``, ``anthropic``) and the downstream extractors that
consume them (``toponym_mentions`` for body-text mining,
``pfsrd2_monsters`` for the wyrd-ami creature corpus,
``fantasy`` for the fantasy-name etymology research pipeline).

Each module is independent today. A future ticket (wyrd-ru5d
follow-up) extracts the shared ``chat_json`` protocol, transport-
error normalization, schema-dialect registry, and form-in-body
validation (D3) into a ``_runner.py`` shared base so the providers
become thin concrete impls. Until that lands, the modules
co-exist as siblings.

Back-compat: ``wyrd.generators.kenning`` aliases the old top-level
module names (``llm_extractor``, ``gemini_extractor`` etc.) to
these so existing call sites keep working. New code should import
from this package directly:

    from wyrd.generators.kenning.extractors import gemini
    from wyrd.generators.kenning.extractors.llm import OllamaClient
"""

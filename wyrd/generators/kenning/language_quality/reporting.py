"""Markdown + JSON output formatters for the language-quality dashboard.

Companion to ``models.py`` (dataclasses) and ``audits.py`` (computation).
This module owns the human-facing output: ``report_to_markdown`` for the
table the dashboard CLI prints; ``report_to_json`` for the
machine-readable snapshot.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .models import LanguageQualityReport, LanguageScorecard

# --- formatters -----------------------------------------------------------


def report_to_json(report: LanguageQualityReport) -> str:
    """JSON snapshot — schema-versioned, ISO-timestamped, full per-language
    metric set. Persistable for trend tracking."""
    return json.dumps(asdict(report), indent=2, sort_keys=False)


def _format_pct(num: int, denom: int) -> str:
    if not denom:
        return "0.0%"
    return f"{100 * num / denom:.1f}%"


def _format_chain(scorecard: LanguageScorecard) -> str:
    if not scorecard.chain_known:
        return "(no chain config)"
    older = ", ".join(scorecard.chain_older) if scorecard.chain_older else "terminus-back"
    younger = ", ".join(scorecard.chain_younger) if scorecard.chain_younger else "terminus-forward"
    return f"older=[{older}] younger=[{younger}]"


def report_to_markdown(report: LanguageQualityReport) -> str:
    """Human-readable per-language scorecard. Summary table at top,
    then one detail block per language. Section letters (A → I) match
    the dataclass docstring + ticket scope."""
    lines: list[str] = []
    lines.append(f"# Language-quality scorecard — {report.generated_at}")
    lines.append("")
    lines.append(f"Schema version: {report.schema_version}")
    lines.append(f"Reference profile (top {len(report.reference_tags)} English tags):")
    lines.append("  " + ", ".join(report.reference_tags))
    lines.append(f"Bundle total words: {report.bundle_total_words}")
    # wyrd-5ecx: when an active slice filter is in play, surface it in
    # the header so the operator knows what subset of the corpus is
    # being measured ('rando-port view', 'fantasy view', etc.).
    if report.source_filter is not None:
        lines.append(f"**Slice filter: source = `{report.source_filter}`**")
    if report.tag_filter is not None:
        lines.append(f"**Slice filter: tag = `{report.tag_filter}`**")
    lines.append("")

    # --- summary table ---
    lines.append("## Summary")
    lines.append("")
    # wyrd-bfv1: the "Etymons" column shows eligible / raw — eligible
    # is the load-bearing number (denominator for every coverage %);
    # raw is shown for DB-depth context. All coverage cells use the
    # eligible denominator.
    lines.append(
        "| Language | Etymons (elig/raw) | Lemmas (elig/raw) | Promo (≥thr) | Bundle | "
        "Scholar atst | IPA (db / bundle) | Grandfather | Lex tag hit | "
        "Bundle tag hit | Inflect | Variants | Era reflex (lex / bundle) |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|---:|---|---|---|---|---|")
    for c in report.languages:
        promo = f"{c.promotion_eligible} (≥{c.promotion_threshold})"
        bundle_pct = _format_pct(c.bundle_word_count, report.bundle_total_words)
        bundle_cell = f"{c.bundle_word_count} ({bundle_pct})"
        ref_n = len(c.reference_tags)
        lex_hit = f"{int(round(c.lexicon_tag_hit_rate * ref_n))}/{ref_n}"
        bundle_hit = f"{int(round(c.bundle_tag_hit_rate * ref_n))}/{ref_n}"
        etymons_cell = f"{c.eligible_etymons}/{c.total_etymons}"
        lemmas_cell = f"{c.eligible_lemmas}/{c.total_lemmas}"
        inflect = (
            f"{c.lemmas_with_inflections} "
            f"({_format_pct(c.lemmas_with_inflections, c.eligible_lemmas)})"
        )
        variants = (
            f"{c.etymons_with_variants} "
            f"({_format_pct(c.etymons_with_variants, c.eligible_etymons)})"
        )
        lex_era_pct = _format_pct(c.any_reflex_count, c.eligible_etymons)
        if c.bundle_attestation_total == 0:
            bundle_era_pct = "n/a"
        else:
            bundle_era_pct = _format_pct(c.bundle_subjects_with_reflex, c.bundle_attestation_total)
        era = f"{lex_era_pct} / {bundle_era_pct}"
        # Scholar-attestation column: scholar / total bundle subjects
        # for this language. Renders 'n/a' when language has no bundle
        # representation (denominator zero) so a dashboard reader can
        # tell 'absent' apart from '0% attested'.
        if c.bundle_attestation_total == 0:
            scholar_atst = "n/a"
        else:
            scholar_atst = (
                f"{c.bundle_scholar_attested}/"
                f"{c.bundle_attestation_total} "
                f"({_format_pct(c.bundle_scholar_attested, c.bundle_attestation_total)})"
            )
        # IPA column: 'db_pct / bundle_pct' so the gap between the
        # two is visible at a glance. A high bundle / 0 db reading
        # flags pure cluster-mate inheritance (modern_english is the
        # canonical example post-wyrd-69s5). DB pct uses the eligible
        # denominator (wyrd-bfv1).
        lex_ipa_pct = _format_pct(c.lexicon_etymons_with_ipa, c.eligible_etymons)
        if c.bundle_attestation_total == 0:
            bundle_ipa_pct = "n/a"
        else:
            bundle_ipa_pct = _format_pct(c.bundle_subjects_with_ipa, c.bundle_attestation_total)
        ipa_cell = f"{lex_ipa_pct} / {bundle_ipa_pct}"
        # Grandfather column: pure-grandfather families / cited families.
        # 'n/a' when this language has no citation-bearing families at
        # all (denominator zero).
        if c.rando_total_cited_families == 0:
            grandfather_cell = "n/a"
        else:
            grandfather_cell = (
                f"{c.rando_pure_grandfather_families}/{c.rando_total_cited_families} "
                f"({_format_pct(c.rando_pure_grandfather_families, c.rando_total_cited_families)})"
            )
        lines.append(
            f"| {c.language} | {etymons_cell} | {lemmas_cell} | {promo} | "
            f"{bundle_cell} | {scholar_atst} | {ipa_cell} | {grandfather_cell} | "
            f"{lex_hit} | {bundle_hit} | {inflect} | {variants} | {era} |"
        )
    lines.append("")

    # --- per-language detail ---
    lines.append("## Per-language detail")
    lines.append("")
    for c in report.languages:
        lines.append(f"### {c.language}")
        lines.append(f"_chain: {_format_chain(c)}_")
        lines.append("")
        # A. Corpus depth — eligible counts are the load-bearing
        # operator signal; the raw counts are shown in parens for
        # DB-depth context. The eligible denominator is what every
        # coverage % below divides by (wyrd-bfv1).
        lines.append(
            f"- **A. Corpus depth:** {c.eligible_etymons} generator-eligible etymons "
            f"({c.eligible_lemmas} eligible lemmas) of {c.total_etymons} raw etymons "
            f"({c.total_lemmas} raw lemmas in DB); "
            f"{c.promotion_eligible} promotion-eligible at threshold ≥{c.promotion_threshold}, "
            f"avg {c.avg_witnesses} witnesses across {c.source_count} sources."
        )
        if c.bundle_sibling is None:
            bundle_line = (
                "- **B. Bundle representation:** no dedicated bundle sibling for this language."
            )
        else:
            bundle_line = (
                f"- **B. Bundle representation:** {c.bundle_word_count} words "
                f"({_format_pct(c.bundle_word_count, report.bundle_total_words)} of "
                f"{report.bundle_total_words} bundle words; sibling=`{c.bundle_sibling}`"
            )
            if c.bundle_shared_with:
                bundle_line += f"; **shared with** {', '.join(c.bundle_shared_with)}"
            bundle_line += ")."
        lines.append(bundle_line)
        # B₂. Bundle attestation breakdown — wyrd-lc94. Surfaces the
        # actionable runtime-quality signal: of what's actually shipping,
        # what fraction is scholar-attested vs grandfathered. Distinct
        # from A (DB-level corpus depth) which counts mining residue
        # that never reaches the bundle.
        if c.bundle_attestation_total == 0:
            atst_line = "- **B₂. Bundle attestation:** n/a — no bundle subjects for this language."
        else:
            atst_line = (
                f"- **B₂. Bundle attestation:** "
                f"{c.bundle_scholar_attested} scholar-attested "
                f"({_format_pct(c.bundle_scholar_attested, c.bundle_attestation_total)}), "
                f"{c.bundle_empirical_only} wiktionary-empirical-only "
                f"({_format_pct(c.bundle_empirical_only, c.bundle_attestation_total)}), "
                f"{c.bundle_rando_only} rando-port-grandfather-only "
                f"({_format_pct(c.bundle_rando_only, c.bundle_attestation_total)}), "
                f"{c.bundle_uncited} uncited "
                f"({_format_pct(c.bundle_uncited, c.bundle_attestation_total)}); "
                f"of {c.bundle_attestation_total} subjects shipped."
            )
        lines.append(atst_line)
        # K. Rando-port grandfather audit (wyrd-vn09). Family-level
        # classification of which morpheme families ride solely on the
        # legacy seed. Distinct from B₂ above because that operated on
        # the bundle's per-word citation list (cluster-propagated). The
        # 'pure_grandfather' count is the curation candidate set —
        # families that would lose their bundle slot if include_rando
        # were turned off.
        if c.rando_total_cited_families == 0:
            grandfather_line = (
                "- **K. Rando-port grandfather audit:** n/a — no "
                "citation-bearing families rooted in this language."
            )
        else:
            grandfather_line = (
                f"- **K. Rando-port grandfather audit:** "
                f"{c.rando_pure_grandfather_families} pure-grandfather "
                f"({_format_pct(c.rando_pure_grandfather_families, c.rando_total_cited_families)}), "
                f"{c.rando_mixed_grandfather_families} mixed (rando-port + scholar/empirical), "
                f"of {c.rando_total_cited_families} cited families "
                f"rooted in this language."
            )
        lines.append(grandfather_line)
        # H. Pronunciation / IPA coverage (wyrd-dxu2). Two-axis line:
        # lexicon-side (true per-language IPA we have in the DB) and
        # bundle-side (what users see, which can include cluster-mate
        # inheritance). The gap between the two is the diagnostic
        # signal — a 0% / 20% reading means the bundle's IPA on this
        # language is entirely inherited from cluster mates.
        # Lex-side IPA % uses eligible denominator (wyrd-bfv1).
        ipa_db_pct = _format_pct(c.lexicon_etymons_with_ipa, c.eligible_etymons)
        if c.bundle_attestation_total == 0:
            ipa_line = (
                f"- **H. Pronunciation coverage:** lexicon "
                f"{c.lexicon_etymons_with_ipa}/{c.eligible_etymons} ({ipa_db_pct}); "
                f"bundle n/a — no bundle subjects for this language."
            )
        else:
            ipa_bundle_pct = _format_pct(c.bundle_subjects_with_ipa, c.bundle_attestation_total)
            ipa_line = (
                f"- **H. Pronunciation coverage:** lexicon "
                f"{c.lexicon_etymons_with_ipa}/{c.eligible_etymons} ({ipa_db_pct}); "
                f"bundle {c.bundle_subjects_with_ipa}/{c.bundle_attestation_total} "
                f"({ipa_bundle_pct})."
            )
            # Diagnostic note when bundle reading is entirely inherited
            # from cluster mates (lexicon side is 0 but bundle side is
            # non-zero). Helps the reader distinguish 'we have IPA for
            # this language' from 'we're showing cluster mates' IPA'.
            if c.lexicon_etymons_with_ipa == 0 and c.bundle_subjects_with_ipa > 0:
                ipa_line += (
                    " ⚠ Bundle IPA is INHERITED from cluster mates "
                    "(no native IPA in the DB for this language)."
                )
        lines.append(ipa_line)
        lex_hit = sum(1 for v in c.lexicon_tag_coverage.values() if v > 0)
        lex_missing = [t for t, v in c.lexicon_tag_coverage.items() if v == 0]
        lex_miss_str = ", ".join(lex_missing) if lex_missing else "none"
        lines.append(
            f"- **C. Semantic-tag coverage (lexicon):** {c.lexicon_tagged_etymons}/"
            f"{c.eligible_etymons} eligible etymons "
            f"({_format_pct(c.lexicon_tagged_etymons, c.eligible_etymons)}) "
            f"carry ≥1 tag. Reference-tag hits: {lex_hit}/{len(c.reference_tags)} "
            f"({_format_pct(lex_hit, len(c.reference_tags))}). Missing: {lex_miss_str}."
        )
        bundle_hit = sum(1 for v in c.bundle_tag_coverage.values() if v > 0)
        if c.bundle_sibling is None:
            bundle_tag_line = (
                "- **C₂. Bundle tag visibility:** n/a — language has no dedicated bundle sibling."
            )
        else:
            bundle_missing = [t for t, v in c.bundle_tag_coverage.items() if v == 0]
            bundle_miss_str = ", ".join(bundle_missing) if bundle_missing else "none"
            bundle_tag_line = (
                f"- **C₂. Bundle tag visibility:** {bundle_hit}/{len(c.reference_tags)} "
                f"({_format_pct(bundle_hit, len(c.reference_tags))}). Missing: {bundle_miss_str}."
            )
        lines.append(bundle_tag_line)
        lines.append(
            f"- **D. Inflection coverage:** {c.lemmas_with_inflections}/{c.eligible_lemmas} "
            f"eligible lemmas ({_format_pct(c.lemmas_with_inflections, c.eligible_lemmas)}) "
            f"with linked inflections; {c.distinct_inflection_labels} distinct case labels."
        )
        lines.append(
            f"- **E. Spelling-variant coverage:** {c.etymons_with_variants}/{c.eligible_etymons} "
            f"eligible etymons ({_format_pct(c.etymons_with_variants, c.eligible_etymons)})."
        )
        # F. era-reflex
        forward_label = f"{c.tier2_forward_count}" if c.chain_younger else "n/a (terminus-forward)"
        back_label = f"{c.tier2_back_count}" if c.chain_older else "n/a (terminus-back)"
        if c.bundle_attestation_total == 0:
            bundle_reflex_detail = (
                f"{c.bundle_subjects_with_reflex}/{c.bundle_attestation_total} (n/a)"
            )
        else:
            bundle_reflex_detail = (
                f"{c.bundle_subjects_with_reflex}/{c.bundle_attestation_total} "
                f"({_format_pct(c.bundle_subjects_with_reflex, c.bundle_attestation_total)})"
            )
        # F (Tier 4) — phonology-rule synthesized count (wyrd-pmne). -1
        # encodes 'no phonology chain registered for this language'
        # OR 'skipped via --no-tier4', both rendered as 'n/a' so the
        # cell visually flags absence rather than 0 firing.
        if c.tier4_phonology_count == -1:
            tier4_label = "n/a"
        else:
            tier4_label = (
                f"{c.tier4_phonology_count} "
                f"({_format_pct(c.tier4_phonology_count, c.eligible_etymons)})"
            )
        lines.append(
            f"- **F. Era-reflex coverage:** Tier 1 cognate {c.tier1_cognate_count}, "
            f"Tier 2 forward {forward_label}, Tier 2 back {back_label}, "
            f"Tier 3 period-form {c.tier3_period_form_count}, "
            f"Tier 4 phonology {tier4_label}; "
            f"ANY reflex {c.any_reflex_count} "
            f"({_format_pct(c.any_reflex_count, c.eligible_etymons)}). "
            f"Bundle subjects with reflex stop targeting this language: "
            f"{bundle_reflex_detail}."
        )
        lines.append(
            f"- **G. Stratum coverage:** {c.stratum_classified}/{c.eligible_etymons} "
            f"({_format_pct(c.stratum_classified, c.eligible_etymons)})."
        )
        lines.append(
            f"- **I. Citation depth:** {c.etymons_with_citations}/{c.eligible_etymons} "
            f"({_format_pct(c.etymons_with_citations, c.eligible_etymons)}) cited; "
            f"avg {c.avg_citations} citations per cited etymon."
        )
        lines.append("")
    return "\n".join(lines)

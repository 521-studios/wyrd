"""Tests for the canonicalization assertion L2 layer (u6fn.2, D50)."""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.canonicalization import (
    Assertion,
    AssertionValidationError,
    NodeRef,
    append_assertion,
    effective_assertions,
    load_assertions,
    mint_canonical_id,
    stream_filename,
    validate,
)
from wyrd.generators.kenning.canonicalization.assertions import mint_assertion_id


def _bind(etymon_ref: str, canonical: str, **kw) -> Assertion:
    return Assertion(
        predicate="bind",
        subject=NodeRef("etymon", etymon_ref),
        object=NodeRef("canonical_morpheme", canonical),
        qualifiers={"kind": "same-morpheme"},
        confidence="high",
        **kw,
    )


def _bind_with_conf(conf) -> Assertion:
    return Assertion(
        predicate="bind",
        subject=NodeRef("etymon", "x"),
        object=NodeRef("canonical_morpheme", "CM-x"),
        qualifiers={"kind": "same-morpheme"},
        confidence=conf,
    )


def test_row_round_trip_binary():
    a = _bind("671826", "CM-new-oe", source="agent:test", rationale="niwe").with_id()
    assert Assertion.from_row(a.to_row()) == a


def test_row_round_trip_unary_and_omits_empty_fields():
    a = Assertion(
        predicate="mint-canonical",
        subject=NodeRef("canonical_morpheme", "CM-new-oe"),
    ).with_id()
    row = a.to_row()
    assert "object" not in row
    assert "qualifiers" not in row
    assert "rationale" not in row
    assert row["_type"] == "canonical_assertion"
    assert Assertion.from_row(row) == a


def test_qualifiers_survive_round_trip():
    a = Assertion(
        predicate="decomposes-into",
        subject=NodeRef("toponym", "1725"),
        object=NodeRef("etymon", "671826"),
        qualifiers={"ordinal": 0},
    ).with_id()
    assert Assertion.from_row(a.to_row()).qualifiers == {"ordinal": 0}


def test_assertion_id_is_deterministic():
    assert mint_assertion_id(_bind("671826", "CM-new-oe", source="s1")) == (
        mint_assertion_id(_bind("671826", "CM-new-oe", source="s1"))
    )


def test_assertion_id_varies_by_source():
    a1 = _bind("671826", "CM-new-oe", source="s1")
    a2 = _bind("671826", "CM-new-oe", source="s2")
    assert mint_assertion_id(a1) != mint_assertion_id(a2)


def test_with_id_is_idempotent_and_preserves_existing():
    a = _bind("671826", "CM-new-oe").with_id()
    assert a.with_id() is a
    assert a.id.startswith("a-")


def test_mint_canonical_id_prefix_and_determinism():
    cid = mint_canonical_id("canonical_morpheme", "old-english", "niwe")
    assert cid.startswith("CM-")
    assert cid == mint_canonical_id("canonical_morpheme", "old-english", "niwe")
    assert mint_canonical_id("canonical_place", "x").startswith("CP-")
    assert mint_canonical_id("canonical_sense", "x").startswith("CS-")


def test_mint_canonical_id_rejects_bad_type():
    with pytest.raises(AssertionValidationError):
        mint_canonical_id("etymon", "x")


def test_unknown_predicate_rejected():
    with pytest.raises(AssertionValidationError, match="unknown predicate"):
        validate(Assertion(predicate="same-morpheme-as", subject=NodeRef("etymon", "x")))


def test_bind_requires_observation_subject_and_canonical_object():
    with pytest.raises(AssertionValidationError, match="object type"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", "671826"),
                object=NodeRef("etymon", "377353"),
                qualifiers={"kind": "same-morpheme"},
            )
        )
    with pytest.raises(AssertionValidationError, match="subject type"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("canonical_morpheme", "CM-x"),
                object=NodeRef("canonical_morpheme", "CM-y"),
                qualifiers={"kind": "same-morpheme"},
            )
        )


def test_unary_predicate_rejects_object():
    with pytest.raises(AssertionValidationError, match="unary"):
        validate(
            Assertion(
                predicate="mint-canonical",
                subject=NodeRef("canonical_morpheme", "CM-x"),
                object=NodeRef("canonical_morpheme", "CM-y"),
            )
        )


def test_binary_predicate_requires_object():
    with pytest.raises(AssertionValidationError, match="requires an object"):
        validate(
            Assertion(
                predicate="descends-from",
                subject=NodeRef("etymon", "town"),
                qualifiers={"edge_type": "inheritance"},
            )
        )


def test_missing_required_qualifier_rejected():
    with pytest.raises(AssertionValidationError, match="missing required qualifier"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", "x"),
                object=NodeRef("canonical_morpheme", "CM-x"),
            )
        )


def test_unknown_qualifier_rejected():
    with pytest.raises(AssertionValidationError, match="unknown qualifier"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", "x"),
                object=NodeRef("canonical_morpheme", "CM-x"),
                qualifiers={"kind": "same-morpheme", "stratum": "oe-late"},
            )
        )


def test_bad_bind_kind_rejected():
    with pytest.raises(AssertionValidationError, match="bind kind"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", "x"),
                object=NodeRef("canonical_morpheme", "CM-x"),
                qualifiers={"kind": "same-thing"},
            )
        )


def test_bad_edge_type_rejected():
    with pytest.raises(AssertionValidationError, match="edge_type"):
        validate(
            Assertion(
                predicate="descends-from",
                subject=NodeRef("etymon", "town"),
                object=NodeRef("etymon", "tun"),
                qualifiers={"edge_type": "begat"},
            )
        )


def test_bad_ordinal_rejected():
    for bad in (-1, "0", True, 1.5):
        with pytest.raises(AssertionValidationError, match="ordinal"):
            validate(
                Assertion(
                    predicate="decomposes-into",
                    subject=NodeRef("toponym", "1725"),
                    object=NodeRef("etymon", "671826"),
                    qualifiers={"ordinal": bad},
                )
            )


def test_retract_requires_retracts_id():
    with pytest.raises(AssertionValidationError, match="retract polarity requires"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", "x"),
                object=NodeRef("canonical_morpheme", "CM-x"),
                qualifiers={"kind": "same-morpheme"},
                polarity="retract",
            )
        )


def test_retracts_without_retract_polarity_rejected():
    with pytest.raises(AssertionValidationError, match="only valid when polarity"):
        validate(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", "x"),
                object=NodeRef("canonical_morpheme", "CM-x"),
                qualifiers={"kind": "same-morpheme"},
                retracts="a-deadbeef",
            )
        )


def test_confidence_accepts_label_and_float_rejects_other():
    for ok in ("high", "medium", "low", 0.0, 1.0, 0.42):
        validate(_bind_with_conf(ok))
    for bad in ("certain", -0.1, 1.1, True):
        with pytest.raises(AssertionValidationError, match="confidence"):
            validate(_bind_with_conf(bad))


def test_stream_filename_layout():
    assert stream_filename("mint-canonical") == "_canonical_nodes.jsonl"
    assert stream_filename("bind") == "_assert_bind.jsonl"
    assert stream_filename("decomposes-into") == "_assert_decomposes_into.jsonl"


def test_append_creates_per_predicate_files(tmp_path):
    append_assertion(
        tmp_path,
        Assertion(
            predicate="mint-canonical",
            subject=NodeRef("canonical_morpheme", "CM-new-oe"),
        ),
    )
    append_assertion(tmp_path, _bind("671826", "CM-new-oe"))
    cdir = tmp_path / "canonicalization"
    assert (cdir / "_canonical_nodes.jsonl").exists()
    assert (cdir / "_assert_bind.jsonl").exists()


def test_append_load_round_trip(tmp_path):
    written = [
        append_assertion(
            tmp_path,
            Assertion(
                predicate="mint-canonical",
                subject=NodeRef("canonical_morpheme", "CM-new-oe"),
            ),
        ),
        append_assertion(tmp_path, _bind("671826", "CM-new-oe")),
        append_assertion(tmp_path, _bind("2229070", "CM-new-oe")),
    ]
    loaded = list(load_assertions(tmp_path))
    assert {a.id for a in loaded} == {a.id for a in written}


def test_append_is_append_only(tmp_path):
    append_assertion(tmp_path, _bind("671826", "CM-new-oe"))
    append_assertion(tmp_path, _bind("2229070", "CM-new-oe"))
    body = (tmp_path / "canonicalization" / "_assert_bind.jsonl").read_text()
    assert len([ln for ln in body.splitlines() if ln.strip()]) == 2


def test_load_empty_dir_yields_nothing(tmp_path):
    assert list(load_assertions(tmp_path)) == []


def test_effective_assertions_drops_retracted_and_the_retraction():
    bind = _bind("4631595", "CM-ne-legacy").with_id()
    retract = Assertion(
        predicate="bind",
        subject=NodeRef("etymon", "4631595"),
        object=NodeRef("canonical_morpheme", "CM-ne-legacy"),
        qualifiers={"kind": "same-morpheme"},
        polarity="retract",
        retracts=bind.id,
    ).with_id()
    assert effective_assertions([bind, retract]) == []


def test_effective_assertions_keeps_refute():
    refute = Assertion(
        predicate="decomposition-spurious",
        subject=NodeRef("toponym_etymology", "780"),
        polarity="refute",
        qualifiers={"reason": "contaminated"},
    ).with_id()
    assert effective_assertions([refute]) == [refute]


def test_newton_worked_example_round_trips_and_resolves(tmp_path):
    cm = "CM-new-oe"
    seq = [
        Assertion(predicate="mint-canonical", subject=NodeRef("canonical_morpheme", cm)),
        _bind("671826", cm, rationale="niwe, gloss 'new'"),
        _bind("2229070", cm, rationale="nīwe macron variant"),
    ]
    written = [append_assertion(tmp_path, a) for a in seq]

    legacy_bind = _bind("4631595", "CM-ne-legacy", source="legacy-import").with_id()
    append_assertion(tmp_path, legacy_bind)
    retract = append_assertion(
        tmp_path,
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon", "4631595"),
            object=NodeRef("canonical_morpheme", "CM-ne-legacy"),
            qualifiers={"kind": "same-morpheme"},
            confidence="high",
            polarity="retract",
            retracts=legacy_bind.id,
            rationale="new belongs with niwe, not the negative-particle node",
        ),
    )
    rebind = append_assertion(tmp_path, _bind("4631595", cm, source="agent:test"))

    spurious = append_assertion(
        tmp_path,
        Assertion(
            predicate="decomposition-spurious",
            subject=NodeRef("toponym_etymology", "780"),
            polarity="refute",
            qualifiers={"reason": "contaminated"},
            rationale="note gives Newtona/Neuton (=new+tun); ford/botl bled in",
        ),
    )
    append_assertion(
        tmp_path,
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-newton-makerfield"),
            qualifiers={"stratum": "domesday", "value": "Neutune"},
        ),
    )

    live_ids = {a.id for a in effective_assertions(list(load_assertions(tmp_path)))}

    assert legacy_bind.id not in live_ids
    assert retract.id not in live_ids
    assert rebind.id in live_ids
    assert spurious.id in live_ids
    assert all(a.id for a in written)

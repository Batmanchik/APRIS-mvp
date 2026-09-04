from __future__ import annotations

import pytest

from apris.cheops.domain.entity_resolution import (
    ENTITY_TYPE_IP,
    ENTITY_TYPE_TOO,
    ENTITY_TYPE_UNKNOWN,
    group_entity_records,
    normalize_entity_record,
    normalize_external_entity_id,
    resolve_entity_key,
)


def test_normalize_external_entity_id_removes_separators() -> None:
    assert normalize_external_entity_id(" too-001 / ab ") == "TOO001AB"


def test_normalize_entity_record_maps_entity_type_aliases() -> None:
    ip_record = normalize_entity_record(
        {"entity_type": "individual_entrepreneur", "iin": "123456789012", "jurisdiction": "kz"}
    )
    too_record = normalize_entity_record(
        {"entity_type": "llp", "bin": "123456789012", "jurisdiction": "kz"}
    )
    unknown_record = normalize_entity_record({"source_entity_id": "x-1", "jurisdiction": "kz"})

    assert ip_record.entity_type == ENTITY_TYPE_IP
    assert too_record.entity_type == ENTITY_TYPE_TOO
    assert unknown_record.entity_type == ENTITY_TYPE_UNKNOWN


def test_resolve_entity_key_is_stable_for_same_bin() -> None:
    first = resolve_entity_key(
        {
            "entity_type": "too",
            "bin": "12-3456789-012",
            "name": "Alpha Trade",
            "jurisdiction": "kz",
        }
    )
    second = resolve_entity_key(
        {
            "entity_type": "LLP",
            "bin": "123456789012",
            "name": "Alpha Trade Group",
            "jurisdiction": "KZ",
        }
    )

    assert first == second


def test_resolve_entity_key_changes_for_different_anchors() -> None:
    first = resolve_entity_key({"entity_type": "too", "bin": "123456789012", "jurisdiction": "kz"})
    second = resolve_entity_key({"entity_type": "too", "bin": "999999999999", "jurisdiction": "kz"})

    assert first != second


def test_validate_entity_schema_rejects_empty_anchor_set() -> None:
    with pytest.raises(ValueError, match="at least one anchor"):
        normalize_entity_record({"entity_type": "ip", "jurisdiction": "kz"})


def test_group_entity_records_merges_duplicates() -> None:
    grouped = group_entity_records(
        [
            {"entity_type": "too", "bin": "123456789012", "name": "Alpha", "jurisdiction": "KZ"},
            {"entity_type": "llp", "bin": "123456789012", "name": "ALPHA 2", "jurisdiction": "kz"},
            {"entity_type": "ip", "iin": "555555555555", "name": "Beta", "jurisdiction": "KZ"},
        ]
    )

    assert len(grouped) == 2
    assert sorted(len(records) for records in grouped.values()) == [1, 2]


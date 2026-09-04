from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Sequence


ENTITY_CONTRACT_VERSION = "1.0.0"

ENTITY_TYPE_IP = "IP"
ENTITY_TYPE_TOO = "TOO"
ENTITY_TYPE_UNKNOWN = "UNKNOWN"

ALLOWED_ENTITY_TYPES = {ENTITY_TYPE_IP, ENTITY_TYPE_TOO, ENTITY_TYPE_UNKNOWN}

_ENTITY_TYPE_ALIASES = {
    "IP": ENTITY_TYPE_IP,
    "ИП": ENTITY_TYPE_IP,
    "SOLE_PROPRIETOR": ENTITY_TYPE_IP,
    "INDIVIDUAL_ENTREPRENEUR": ENTITY_TYPE_IP,
    "SP": ENTITY_TYPE_IP,
    "TOO": ENTITY_TYPE_TOO,
    "ТОО": ENTITY_TYPE_TOO,
    "LLP": ENTITY_TYPE_TOO,
    "LEGAL_ENTITY": ENTITY_TYPE_TOO,
    "COMPANY": ENTITY_TYPE_TOO,
    "UNKNOWN": ENTITY_TYPE_UNKNOWN,
}

_ID_TOKEN_PATTERN = re.compile(r"[^A-Z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class LegalEntityRecord:
    entity_type: str
    jurisdiction: str
    source_entity_id: str | None = None
    name: str | None = None
    bin: str | None = None
    iin: str | None = None
    registration_no: str | None = None
    tax_id: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_external_entity_id(value: Any) -> str:
    text = str(value).strip().upper()
    return _ID_TOKEN_PATTERN.sub("", text)


def _normalize_name(value: Any) -> str:
    text = _WHITESPACE_PATTERN.sub(" ", str(value).strip()).upper()
    return text


def _normalize_entity_type(value: Any) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ENTITY_TYPE_UNKNOWN
    return _ENTITY_TYPE_ALIASES.get(raw, raw)


def _optional_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_external_entity_id(value)
    return normalized or None


def _normalize_aliases(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = {_normalize_name(item) for item in value if str(item).strip()}
    return tuple(sorted(cleaned))


def validate_entity_schema(entity: LegalEntityRecord) -> None:
    if entity.entity_type not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {sorted(ALLOWED_ENTITY_TYPES)}.")
    if not entity.jurisdiction:
        raise ValueError("jurisdiction must be non-empty.")

    if entity.bin is not None and (not entity.bin.isdigit() or len(entity.bin) != 12):
        raise ValueError("bin must be a 12-digit numeric string.")
    if entity.iin is not None and (not entity.iin.isdigit() or len(entity.iin) != 12):
        raise ValueError("iin must be a 12-digit numeric string.")

    anchors = (
        entity.bin,
        entity.iin,
        entity.registration_no,
        entity.tax_id,
        entity.source_entity_id,
        entity.name,
    )
    if not any(anchors):
        raise ValueError("Entity record must contain at least one anchor identifier or name.")


def normalize_entity_record(raw: LegalEntityRecord | dict[str, Any]) -> LegalEntityRecord:
    if isinstance(raw, LegalEntityRecord):
        validate_entity_schema(raw)
        return raw

    metadata_raw = raw.get("metadata", {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    record = LegalEntityRecord(
        entity_type=_normalize_entity_type(raw.get("entity_type", ENTITY_TYPE_UNKNOWN)),
        jurisdiction=str(raw.get("jurisdiction", "KZ")).strip().upper(),
        source_entity_id=_optional_id(raw.get("source_entity_id") or raw.get("entity_id")),
        name=(
            _normalize_name(raw["name"])
            if raw.get("name") is not None and str(raw.get("name")).strip()
            else None
        ),
        bin=_optional_id(raw.get("bin")),
        iin=_optional_id(raw.get("iin")),
        registration_no=_optional_id(raw.get("registration_no")),
        tax_id=_optional_id(raw.get("tax_id")),
        aliases=_normalize_aliases(raw.get("aliases")),
        metadata=metadata,
    )
    validate_entity_schema(record)
    return record


def resolve_entity_key(raw: LegalEntityRecord | dict[str, Any]) -> str:
    entity = normalize_entity_record(raw)
    anchors: list[str] = []
    # Use the strongest available anchor to avoid accidental key split
    # when cosmetic fields (e.g., name spelling) differ.
    if entity.bin:
        anchors.append(f"bin:{entity.bin}")
    elif entity.iin:
        anchors.append(f"iin:{entity.iin}")
    elif entity.registration_no:
        anchors.append(f"reg:{entity.registration_no}")
    elif entity.tax_id:
        anchors.append(f"tax:{entity.tax_id}")
    elif entity.source_entity_id:
        anchors.append(f"src:{entity.source_entity_id}")
    else:
        if entity.name:
            anchors.append(f"name:{entity.name}")
        anchors.extend(f"alias:{alias}" for alias in entity.aliases)

    payload = {
        "contract_version": ENTITY_CONTRACT_VERSION,
        "entity_type": entity.entity_type,
        "jurisdiction": entity.jurisdiction,
        "anchors": sorted(set(anchors)),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = sha256(encoded.encode("utf-8")).hexdigest()
    return f"entity-{digest[:24]}"


def group_entity_records(
    records: Sequence[LegalEntityRecord | dict[str, Any]],
) -> dict[str, list[LegalEntityRecord]]:
    grouped: dict[str, list[LegalEntityRecord]] = {}
    for raw in records:
        normalized = normalize_entity_record(raw)
        entity_key = resolve_entity_key(normalized)
        grouped.setdefault(entity_key, []).append(normalized)
    return grouped

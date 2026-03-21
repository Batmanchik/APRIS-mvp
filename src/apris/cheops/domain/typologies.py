from __future__ import annotations

from enum import Enum


class FraudTypology(str, Enum):
    LEGAL_LAYERING = "LEGAL_LAYERING"
    LEGAL_TO_CRYPTO_BRIDGE = "LEGAL_TO_CRYPTO_BRIDGE"
    CRYPTO_MIXING = "CRYPTO_MIXING"
    STRUCTURED_SPLITTING = "STRUCTURED_SPLITTING"
    CASH_OUT = "CASH_OUT"


TYPOLOGY_NAMES: tuple[str, ...] = tuple(t.value for t in FraudTypology)


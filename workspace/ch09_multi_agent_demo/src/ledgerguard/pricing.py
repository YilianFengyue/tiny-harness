from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


CENT = Decimal("0.01")


def _decimal(amount: object) -> Decimal:
    if isinstance(amount, Decimal):
        return amount
    return Decimal(str(amount))


def cents(amount: object) -> str:
    """Return a two-decimal money string using audit-safe Decimal rounding."""
    return format(_decimal(amount).quantize(CENT, rounding=ROUND_HALF_UP), ".2f")


def add_money(left: object, right: object) -> Decimal:
    return _decimal(left) + _decimal(right)


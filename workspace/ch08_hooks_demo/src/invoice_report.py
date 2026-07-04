from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


CENT = Decimal("0.01")


def invoice_total(lines: list[dict]) -> str:
    total = Decimal("0")
    for line in lines:
        total += Decimal(str(line["amount"]))
    return str(total.quantize(CENT, rounding=ROUND_HALF_UP))


def render_report(lines: list[dict]) -> str:
    total = invoice_total(lines)
    return f"Invoice total: ${total}"

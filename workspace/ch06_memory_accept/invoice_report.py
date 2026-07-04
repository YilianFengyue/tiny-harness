from __future__ import annotations

from collections import defaultdict
from decimal import Decimal


CENT = Decimal("0.01")


def summarize_invoices(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    totals: defaultdict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for row in rows:
        totals[row["customer"]] += Decimal(row["amount"])

    return [
        {"customer": customer, "total": f"{total.quantize(CENT):.2f}"}
        for customer, total in sorted(totals.items())
    ]

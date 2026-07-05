from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from .models import Order, ParseIssue


CENT = Decimal("0.01")


def parse_orders(path: str | Path) -> tuple[list[Order], list[ParseIssue]]:
    """Parse the CSV into valid orders and ordered parse issues."""
    orders: list[Order] = []
    issues: list[ParseIssue] = []
    seen_invoice_ids: set[str] = set()

    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            invoice_id = (row.get("invoice_id") or "").strip()

            if invoice_id in seen_invoice_ids:
                issues.append(ParseIssue(row_number, invoice_id, "duplicate invoice id"))
                continue

            try:
                amount = Decimal(row.get("amount") or "0").quantize(
                    CENT,
                    rounding=ROUND_HALF_UP,
                )
            except (InvalidOperation, ValueError):
                issues.append(ParseIssue(row_number, invoice_id, "invalid amount"))
                continue

            flags = tuple(
                part.strip()
                for part in (row.get("risk_flags") or "").split("|")
                if part.strip()
            )
            orders.append(
                Order(
                    invoice_id=invoice_id,
                    customer_id=(row.get("customer_id") or "").strip(),
                    amount=amount,
                    currency=(row.get("currency") or "").strip().upper(),
                    status=(row.get("status") or "").strip().lower(),
                    days_overdue=int(row.get("days_overdue") or "0"),
                    risk_flags=flags,
                    notes=(row.get("notes") or "").strip(),
                )
            )
            seen_invoice_ids.add(invoice_id)
    return orders, issues


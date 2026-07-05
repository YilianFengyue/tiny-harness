from __future__ import annotations

import json
from pathlib import Path

from .parser import parse_orders
from .pricing import add_money, cents
from .risk import is_high_risk


def generate_audit_report(csv_path: str | Path) -> dict:
    orders, issues = parse_orders(csv_path)

    booked_by_currency: dict[str, object] = {}
    exposure_by_customer: dict[str, object] = {}
    high_risk: list[str] = []

    for order in orders:
        booked_by_currency[order.currency] = add_money(
            booked_by_currency.get(order.currency, 0), order.amount
        )
        if order.status != "paid":
            exposure_by_customer[order.customer_id] = add_money(
                exposure_by_customer.get(order.customer_id, 0), order.amount
            )
        if is_high_risk(order):
            high_risk.append(order.invoice_id)

    top_customers = [
        {"customer_id": customer_id, "open_exposure": cents(amount)}
        for customer_id, amount in sorted(
            exposure_by_customer.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    return {
        "processed_orders": len(orders),
        "invalid_rows": len(issues),
        "booked_by_currency": {
            currency: cents(amount)
            for currency, amount in sorted(booked_by_currency.items())
        },
        "top_customers": top_customers,
        "high_risk_invoice_ids": sorted(high_risk),
        "issues": [
            {"row_number": issue.row_number, "invoice_id": issue.invoice_id, "reason": issue.reason}
            for issue in issues
        ],
    }


def write_report_json(csv_path: str | Path, output_path: str | Path) -> None:
    report = generate_audit_report(csv_path)
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


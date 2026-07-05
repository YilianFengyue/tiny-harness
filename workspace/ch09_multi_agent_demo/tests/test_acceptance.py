from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ledgerguard import generate_audit_report, write_report_json


DATA = ROOT / "data" / "orders.csv"


def test_report_has_exact_money_totals_and_invalid_row_count():
    report = generate_audit_report(DATA)

    assert report["processed_orders"] == 5
    assert report["invalid_rows"] == 2
    assert report["booked_by_currency"] == {
        "EUR": "50.00",
        "USD": "1312.68",
    }


def test_duplicate_invoice_is_reported_as_issue():
    report = generate_audit_report(DATA)

    duplicate_issues = [
        issue for issue in report["issues"]
        if issue["invoice_id"] == "INV-1002" and "duplicate" in issue["reason"]
    ]
    assert duplicate_issues


def test_high_risk_invoices_use_flags_amount_and_overdue_rules():
    report = generate_audit_report(DATA)

    assert report["high_risk_invoice_ids"] == ["INV-1003", "INV-1005"]


def test_top_customers_are_sorted_by_numeric_open_exposure():
    report = generate_audit_report(DATA)

    assert report["top_customers"][:3] == [
        {"customer_id": "C-002", "open_exposure": "1200.00"},
        {"customer_id": "C-004", "open_exposure": "99.99"},
        {"customer_id": "C-001", "open_exposure": "12.69"},
    ]


def test_write_report_json_is_deterministic(tmp_path):
    out = tmp_path / "audit_report.json"

    write_report_json(DATA, out)
    first = out.read_text(encoding="utf-8")
    write_report_json(DATA, out)
    second = out.read_text(encoding="utf-8")

    assert first == second
    parsed = json.loads(second)
    assert parsed["booked_by_currency"]["USD"] == "1312.68"


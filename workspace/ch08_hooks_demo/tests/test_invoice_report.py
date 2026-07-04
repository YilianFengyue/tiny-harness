import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_report import invoice_total, render_report


def test_invoice_total_uses_decimal_rounding():
    assert invoice_total([
        {"amount": "2.675"},
    ]) == "2.68"


def test_render_report_uses_total():
    assert render_report([
        {"amount": "1.10"},
        {"amount": "2.20"},
    ]) == "Invoice total: $3.30"

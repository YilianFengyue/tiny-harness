import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from invoice.parser import parse_amount
from invoice.pricing import discounted_total
from invoice.report import summarize


def test_parse_amount_accepts_currency_and_commas():
    assert parse_amount('$1,200.50') == Decimal('1200.50')


def test_discount_applies_at_threshold():
    assert discounted_total(Decimal('1000')) == Decimal('900.00')


def test_summary_is_sorted_and_discounted():
    lines = ['Zen,$1000', 'Ada,$1,200.50', 'Ada,$99.50']
    assert summarize(lines) == ['Ada: 1170.00', 'Zen: 900.00']

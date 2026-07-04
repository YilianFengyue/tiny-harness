from decimal import Decimal

from .parser import parse_line
from .pricing import discounted_total


def summarize(lines: list[str]) -> list[str]:
    totals: dict[str, Decimal] = {}
    for line in lines:
        customer, amount = parse_line(line)
        totals[customer] = totals.get(customer, Decimal('0')) + amount
    return [f'{name}: {discounted_total(total)}' for name, total in sorted(totals.items())]

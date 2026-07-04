from decimal import Decimal


def parse_amount(text: str) -> Decimal:
    cleaned = text.strip().replace('$', '').replace(',', '')
    return Decimal(cleaned)


def parse_line(line: str) -> tuple[str, Decimal]:
    customer, amount = line.split(',', 1)
    return customer.strip(), parse_amount(amount)

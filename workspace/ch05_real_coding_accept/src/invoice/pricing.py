from decimal import Decimal


def discounted_total(subtotal: Decimal) -> Decimal:
    if subtotal >= Decimal('1000'):
        return (subtotal * Decimal('0.90')).quantize(Decimal('0.01'))
    return subtotal.quantize(Decimal('0.01'))

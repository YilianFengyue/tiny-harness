from __future__ import annotations

from decimal import Decimal

from .models import Order


HIGH_RISK_FLAGS = {"sanctions_hit", "chargeback"}
AMOUNT_RISK_THRESHOLD = Decimal("1000.00")


def risk_score(order: Order) -> int:
    score = 0
    flags = set(order.risk_flags)
    if order.days_overdue > 90:
        score += 60
    if Decimal(str(order.amount)) > AMOUNT_RISK_THRESHOLD:
        score += 30
    if flags & HIGH_RISK_FLAGS:
        score += 70
    if "manual_review" in flags:
        score += 10
    return score


def is_high_risk(order: Order) -> bool:
    return risk_score(order) >= 70


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    invoice_id: str
    customer_id: str
    amount: object
    currency: str
    status: str
    days_overdue: int
    risk_flags: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ParseIssue:
    row_number: int
    invoice_id: str
    reason: str


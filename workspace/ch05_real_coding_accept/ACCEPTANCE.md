# Acceptance Summary

## Changes
- `src/invoice/parser.py`: stripped thousands separators in `parse_amount` so money strings such as `$1,200.50` parse as `Decimal('1200.50')`.
- `src/invoice/pricing.py`: changed the discount threshold from greater than 1000 to at least 1000, applying the 10% discount at `Decimal('1000')`.
- `src/invoice/report.py`: sorted customer totals alphabetically before rendering summaries for deterministic output.

## Why
These changes implement the README requirements: support currency symbols and thousands separators, apply the 10% discount when subtotal is at least 1000, and render customer summaries in alphabetical order.

## Test Result
- `pytest -q`: `3 passed in 0.02s`

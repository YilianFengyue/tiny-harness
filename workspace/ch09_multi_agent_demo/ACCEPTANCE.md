# Acceptance Notes

## Summary of fixes

- `src/ledgerguard/parser.py`
  - Replaced float money parsing with `Decimal` parsing quantized to cents using `ROUND_HALF_UP`.
  - Added duplicate invoice detection; duplicate rows are reported as parse issues and skipped.
  - Changed risk flag parsing from comma-separated to pipe-separated values.
  - Normalized currencies to uppercase and statuses to lowercase.

- `src/ledgerguard/pricing.py`
  - Replaced float arithmetic/formatting with `Decimal` addition and cent formatting.

- `src/ledgerguard/risk.py`
  - Replaced float threshold checks with `Decimal` comparisons.
  - Applied exact high-risk flags (`sanctions_hit`, `chargeback`) in the high-risk score.

- `src/ledgerguard/report.py`
  - Sorted top customers by numeric open exposure descending, with customer ID as a deterministic tie-breaker.

## Key expected audit results

- Processed valid orders: `5`.
- Invalid rows: `2` (one invalid amount and one duplicate invoice).
- USD booked total: `1312.68` (`2.68 + 10.01 + 1200.00 + 99.99`).
- EUR booked total: `50.00`.
- High-risk invoices: `INV-1003`, `INV-1005`.

## Verification

- Main-agent test run: `python -m pytest -q`
- Result: `5 passed in 0.08s`

## Multi-agent review notes

- An `explore` sub-agent mapped the failing acceptance tests and implicated parser, pricing, risk, and report code.
- A money/risk analysis sub-agent identified the Decimal half-up rounding requirement and high-risk flag semantics.
- An audit semantics review sub-agent confirmed duplicate invoice, invalid row, exposure sorting, and deterministic JSON expectations.
- A `verify` sub-agent inspected the final code read-only; its sandbox could not execute pytest, but it reported the implementation aligned with acceptance semantics.

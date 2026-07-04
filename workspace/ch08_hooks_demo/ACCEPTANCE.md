# Acceptance

## Changes

- Fixed invoice amount aggregation in `src/invoice_report.py` to use `Decimal` instead of binary floating-point arithmetic.
- Applied cent-level quantization with `ROUND_HALF_UP` so invoice totals round monetary values correctly (for example, `2.675` -> `2.68`).
- `render_report` continues to use `invoice_total`, so report output reflects the corrected total.

## Verification

- Reproduced the bug with `python -m pytest`: `test_invoice_total_uses_decimal_rounding` failed because `2.675` was rounded to `2.67`.
- After the fix, ran `python -m pytest` successfully.

## Test Result

```text
2 passed in 0.04s
```

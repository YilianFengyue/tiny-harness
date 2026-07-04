# Acceptance

- Fixed the invoice report amount aggregation bug by using `Decimal` for monetary parsing, summing, and two-decimal formatting.
- Preserved sorted customer output in `summarize_invoices`.
- Verified the report handles half-cent rounding such as `2.675` -> `2.68` and repeated fractional amounts without binary floating-point drift.

Test command:

```powershell
python -m pytest -q
```

Result: `2 passed in 0.02s`.

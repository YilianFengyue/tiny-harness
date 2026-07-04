# Invoice Acceptance Fixture

Fix the invoice package so the tests pass. Keep the tests unchanged.

Expected behavior:
- parse money strings with currency symbols and thousands separators.
- apply a 10% discount when subtotal is at least 1000.
- render customer summaries in deterministic alphabetical order.

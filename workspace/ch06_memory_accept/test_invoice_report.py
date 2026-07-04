from invoice_report import summarize_invoices


def test_summarize_invoices_sorts_customers_and_totals_amounts():
    rows = [
        {"customer": "beta", "amount": "1.25"},
        {"customer": "acme", "amount": "2.675"},
        {"customer": "beta", "amount": "2.00"},
    ]

    assert summarize_invoices(rows) == [
        {"customer": "acme", "total": "2.68"},
        {"customer": "beta", "total": "3.25"},
    ]


def test_summarize_invoices_handles_repeated_fractional_amounts():
    rows = [
        {"customer": "acme", "amount": "0.10"},
        {"customer": "acme", "amount": "0.10"},
        {"customer": "acme", "amount": "0.10"},
    ]

    assert summarize_invoices(rows) == [
        {"customer": "acme", "total": "0.30"},
    ]

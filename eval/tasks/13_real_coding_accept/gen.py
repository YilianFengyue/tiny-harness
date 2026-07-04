import json
from pathlib import Path


def generate(workdir: Path) -> None:
    settings = workdir / ".tiny-harness" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "model": "gpt-5.5",
        "max_turns": 22,
        "permissions": {
            "mode": "acceptEdits",
            "allow": [
                "Bash(python -m pytest *)",
                "Bash(python check.py)"
            ],
            "deny": [
                "Bash(rm *)",
                "Bash(del *)",
                "Write(.env)"
            ]
        },
        "features": {
            "settings_tui": True,
            "app_state": True,
            "coding_acceptance_trace": True
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (workdir / "README.md").write_text(
        "# Invoice Acceptance Fixture\n\n"
        "Fix the invoice package so the tests pass. Keep the tests unchanged.\n\n"
        "Expected behavior:\n"
        "- parse money strings with currency symbols and thousands separators.\n"
        "- apply a 10% discount when subtotal is at least 1000.\n"
        "- render customer summaries in deterministic alphabetical order.\n",
        encoding="utf-8")

    package = workdir / "src" / "invoice"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "parser.py").write_text(
        "from decimal import Decimal\n\n\n"
        "def parse_amount(text: str) -> Decimal:\n"
        "    cleaned = text.strip().replace('$', '')\n"
        "    return Decimal(cleaned)\n\n\n"
        "def parse_line(line: str) -> tuple[str, Decimal]:\n"
        "    customer, amount = line.split(',', 1)\n"
        "    return customer.strip(), parse_amount(amount)\n",
        encoding="utf-8")
    (package / "pricing.py").write_text(
        "from decimal import Decimal\n\n\n"
        "def discounted_total(subtotal: Decimal) -> Decimal:\n"
        "    if subtotal > Decimal('1000'):\n"
        "        return (subtotal * Decimal('0.90')).quantize(Decimal('0.01'))\n"
        "    return subtotal.quantize(Decimal('0.01'))\n",
        encoding="utf-8")
    (package / "report.py").write_text(
        "from decimal import Decimal\n\n"
        "from .parser import parse_line\n"
        "from .pricing import discounted_total\n\n\n"
        "def summarize(lines: list[str]) -> list[str]:\n"
        "    totals: dict[str, Decimal] = {}\n"
        "    for line in lines:\n"
        "        customer, amount = parse_line(line)\n"
        "        totals[customer] = totals.get(customer, Decimal('0')) + amount\n"
        "    return [f'{name}: {discounted_total(total)}' for name, total in totals.items()]\n",
        encoding="utf-8")

    tests = workdir / "tests"
    tests.mkdir()
    (tests / "test_invoice.py").write_text(
        "import sys\n"
        "from decimal import Decimal\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n\n"
        "from invoice.parser import parse_amount\n"
        "from invoice.pricing import discounted_total\n"
        "from invoice.report import summarize\n\n\n"
        "def test_parse_amount_accepts_currency_and_commas():\n"
        "    assert parse_amount('$1,200.50') == Decimal('1200.50')\n\n\n"
        "def test_discount_applies_at_threshold():\n"
        "    assert discounted_total(Decimal('1000')) == Decimal('900.00')\n\n\n"
        "def test_summary_is_sorted_and_discounted():\n"
        "    lines = ['Zen,$1000', 'Ada,$1,200.50', 'Ada,$99.50']\n"
        "    assert summarize(lines) == ['Ada: 1170.00', 'Zen: 900.00']\n",
        encoding="utf-8")
    (workdir / "check.py").write_text(
        "import subprocess\n"
        "import sys\n\n"
        "raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', '-q']))\n",
        encoding="utf-8")

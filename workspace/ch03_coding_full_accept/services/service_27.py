from core.parser import parse_v2
from core.pricing import compute_discount

SENTINEL = "service-27-do-not-touch"

def handle(payload: str, total: float, count: int) -> dict:
    data = parse_v2(payload)
    data["service"] = 27
    data["discount"] = compute_discount(total, count)
    return data

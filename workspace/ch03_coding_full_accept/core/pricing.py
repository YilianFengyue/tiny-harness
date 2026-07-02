def compute_discount(total: float, count: int) -> float:
    if count == 0:
        return 0.0
    return round(total / count, 2)

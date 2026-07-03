def normalize_user(name: str) -> str:
    # BUG: internal whitespace is not collapsed.
    return " ".join(name.strip().lower().split())

def greeting(name: str) -> str:
    return f"hello, {normalize_user(name)}"

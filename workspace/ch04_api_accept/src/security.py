def normalize_user(name: str) -> str:
    return " ".join(name.strip().lower().split())

def is_admin(name: str) -> bool:
    return normalize_user(name) == "admin user"

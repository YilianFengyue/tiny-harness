from src.security import normalize_user, is_admin

assert normalize_user("  Admin   User  ") == "admin user"
assert is_admin(" ADMIN   USER ")
print("all checks passed")

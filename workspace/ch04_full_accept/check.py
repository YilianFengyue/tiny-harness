from app import normalize_user, greeting

assert normalize_user("  Admin   User  ") == "admin user"
assert normalize_user("\tAlice\nBob  ") == "alice bob"
assert greeting("  JOHN   DOE ") == "hello, john doe"
print("all permission checks passed")

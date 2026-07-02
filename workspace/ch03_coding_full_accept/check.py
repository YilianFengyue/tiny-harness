import importlib

failures = []
for i in range(1, 31):
    mod = importlib.import_module(f"services.service_{i:02d}")
    try:
        result = mod.handle("  Hello  ", 10.0, 0)
        if result["engine"] != "v2":
            failures.append(f"case={i:04d} status=FAIL code=LEGACY_ENGINE")
        if result["discount"] != 0.0:
            failures.append(f"case={i:04d} status=FAIL code=BAD_ZERO_DISCOUNT")
    except Exception as e:
        failures.append(f"case={i:04d} status=FAIL code=EXCEPTION detail={type(e).__name__}")

for n in range(1, 2201):
    filler = "x" * 70
    if n in (101, 707, 1313):
        print(f"case={n:04d} status=FAIL code=NOISE_FAIL payload={filler}")
    else:
        print(f"case={n:04d} status=PASS code=OK payload={filler}")

for item in failures:
    print(item)

if failures:
    raise SystemExit(1)

print("ALL TESTS PASSED")

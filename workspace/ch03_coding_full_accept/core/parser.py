def legacy_parse(payload: str) -> dict:
    return {"engine": "legacy", "value": payload.strip().lower()}

def parse_v2(payload: str) -> dict:
    return {"engine": "v2", "value": payload.strip().casefold()}

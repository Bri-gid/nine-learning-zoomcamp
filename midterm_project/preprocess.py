import pandas as pd

FEATURES = ["term", "naturalCourt", "chief", "petitioner", "respondent",
            "jurisdiction", "adminAction", "threeJudgeFdc", "caseOrigin",
            "caseSource", "certReason", "lcDisagreement", "lcDisposition",
            "lcDispositionDirection"]
CAT_COLS = [c for c in FEATURES if c != "term"]
RARE_COLS = ["petitioner", "respondent", "caseOrigin", "caseSource"]


def clean_value(v):
    """12.0 -> '12'; text stays text; missing -> 'missing'."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "missing"
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)


def to_record(row, keep_sets):
    """Turn one raw case (dict-like) into the dict fed to DictVectorizer."""
    rec = {"term": (float(row["term"]) - 1946) / 10}
    for c in CAT_COLS:
        val = clean_value(row.get(c))
        if c in RARE_COLS and val != "missing" and val not in keep_sets[c]:
            val = "other"
        rec[c] = val
    return rec
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from preprocess import FEATURES, CAT_COLS, RARE_COLS, clean_value, to_record

DATA = "data/scdb_case_centered.csv"
OUT = "model.bin"

df = pd.read_csv(DATA)
df = df[df["partyWinning"].isin([0, 1])].copy()
df["y"] = df["partyWinning"].astype(int)

dev = df[df["term"] <= 2016]    # train + validation
test = df[df["term"] > 2016]

# rare-category sets from the training data only
keep_sets = {}
for c in RARE_COLS:
    counts = dev[c].map(clean_value).value_counts()
    keep_sets[c] = set(counts[counts >= 30].index)

def records(frame):
    return [to_record(r, keep_sets) for r in frame[FEATURES].to_dict("records")]

dv = DictVectorizer(sparse=True)
X_dev = dv.fit_transform(records(dev))
X_test = dv.transform(records(test))

model = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=300,
                      random_state=1, eval_metric="logloss")
model.fit(X_dev, dev["y"])

p = model.predict_proba(X_test)[:, 1]
print("TEST AUC:", roc_auc_score(test["y"], p))
print("TEST F1 :", f1_score(test["y"], p >= 0.5))
print("TEST acc:", accuracy_score(test["y"], p >= 0.5))

with open(OUT, "wb") as f:
    pickle.dump({"dv": dv, "model": model, "keep_sets": keep_sets}, f)
print(f"Saved {OUT}")
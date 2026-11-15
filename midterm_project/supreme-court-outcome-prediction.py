import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


df = pd.read_csv("data/scdb_case_centered.csv")

# target
df = df[df["partyWinning"].isin([0, 1])].copy()
y = df["partyWinning"].astype(int)

features = ["term", "naturalCourt", "chief", "petitioner", "respondent",
            "jurisdiction", "adminAction", "threeJudgeFdc", "caseOrigin",
            "caseSource", "certReason", "lcDisagreement", "lcDisposition",
            "lcDispositionDirection"]

X = df[features].copy()
cat_cols = [c for c in features if c != "term"]
X[cat_cols] = X[cat_cols].astype("string")   # codes are categories, not numbers

print(X.shape, y.mean())
# --- check missing values ---
print((X.isna().mean() * 100).round(1).sort_values(ascending=False))

# --- clean codes: 12.0 -> "12" (skip text columns like chief) ---
for c in cat_cols:
    num = pd.to_numeric(df[c], errors="coerce")
    if num.notna().sum() == df[c].notna().sum():      # column is fully numeric-coded
        X[c] = num.astype("Int64").astype("string")
    else:
        X[c] = df[c].astype("string")

# --- time split masks ---
train = df["term"] <= 2009
val   = (df["term"] > 2009) & (df["term"] <= 2016)
test  = df["term"] > 2016

# --- group rare categories (counts from TRAIN only) ---
def group_rare(train_col, col, min_count=30):
    counts = train_col.value_counts()
    keep = set(counts[counts >= min_count].index)
    return col.where(col.isin(keep) | col.isna(), "other")

for c in ["petitioner", "respondent", "caseOrigin", "caseSource"]:
    X[c] = group_rare(X.loc[train, c], X[c])

# --- missing values become their own category ---
X[cat_cols] = X[cat_cols].fillna("missing")

X_train, y_train = X[train], y[train]
X_val,   y_val   = X[val],   y[val]
X_test,  y_test  = X[test],  y[test]
print(len(X_train), len(X_val), len(X_test))


def to_dicts(X):
    X = X.copy()
    X["term"] = (X["term"] - 1946) / 10
    return X.to_dict(orient="records")

dv = DictVectorizer(sparse=True)
Xtr = dv.fit_transform(to_dicts(X_train))     # fit on train only
Xva = dv.transform(to_dicts(X_val))

# majority-class baseline
majority = y_train.mode()[0]
print("Baseline acc (val):", accuracy_score(y_val, [majority] * len(y_val)))

# logistic regression
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(Xtr, y_train)
p = lr.predict_proba(Xva)[:, 1]
print("AUC:", roc_auc_score(y_val, p))
print("F1 :", f1_score(y_val, p >= 0.5))
print("Acc:", accuracy_score(y_val, p >= 0.5))

results = []

def evaluate(name, model, params):
    model.set_params(**params)
    model.fit(Xtr, y_train)
    p = model.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(y_val, p)
    results.append({"model": name, **params, "val_auc": auc})
    return auc

# 1. Logistic regression: tune C
for C in [0.001, 0.01, 0.1, 1, 10]:
    evaluate("logreg", LogisticRegression(max_iter=2000), {"C": C})

# 2. Decision tree: tune depth and leaf size
for d in [3, 5, 8, 12]:
    for leaf in [5, 20, 50]:
        evaluate("tree", DecisionTreeClassifier(random_state=1),
                 {"max_depth": d, "min_samples_leaf": leaf})

# 3. Random forest
for n in [100, 300]:
    for d in [8, 12, None]:
        for leaf in [5, 20]:
            evaluate("rf", RandomForestClassifier(random_state=1, n_jobs=-1),
                     {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf})

# 4. XGBoost
for lr_ in [0.03, 0.1]:
    for d in [2, 3, 5]:
        for n in [100, 300]:
            evaluate("xgb", XGBClassifier(random_state=1, eval_metric="logloss"),
                     {"learning_rate": lr_, "max_depth": d, "n_estimators": n})

res = pd.DataFrame(results)
print(res.groupby("model")["val_auc"].max())
print(res.sort_values("val_auc", ascending=False).head(10))



# retrain on train + validation with the chosen parameters
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

dv_final = DictVectorizer(sparse=True)
Xf = dv_final.fit_transform(to_dicts(X_full))
Xte = dv_final.transform(to_dicts(X_test))

final = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=300,
                      random_state=1, eval_metric="logloss")
final.fit(Xf, y_full)

p_test = final.predict_proba(Xte)[:, 1]
print("TEST AUC:", roc_auc_score(y_test, p_test))
print("TEST F1 :", f1_score(y_test, p_test >= 0.5))
print("TEST acc:", accuracy_score(y_test, p_test >= 0.5))
print("Baseline acc:", accuracy_score(y_test, [1] * len(y_test)))

# target over time
(y.groupby(df["term"] // 10 * 10).mean()).plot(kind="bar", title="Petitioner win rate by decade")
plt.show()

# target balance
print(y.value_counts(normalize=True))

# XGBoost feature importance (top 15)
imp = pd.Series(final.feature_importances_, index=dv_final.get_feature_names_out())
imp.sort_values(ascending=False).head(15).plot(kind="barh", title="Top features")
plt.gca().invert_yaxis(); plt.show()
"""Compare logistic regression and random forest on the same split, plus a time-based check."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("clean_invoices.csv", parse_dates=["issue_date"])
NUM = ["amount_log", "credit_utilisation", "payment_terms_days", "disputed",
       "prior_late_rate", "prior_invoice_count", "issue_month"]
CAT = ["segment", "region"]
TARGET = "paid_late"

def build(model):
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return make_pipeline(prep, model)

def evaluate(name, model, X_tr, y_tr, X_te, y_te, threshold=0.5):
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= threshold).astype(int)
    row = {"model": name,
           "ROC-AUC": roc_auc_score(y_te, p),
           "precision": precision_score(y_te, pred),
           "recall": recall_score(y_te, pred)}
    return row, p, confusion_matrix(y_te, pred)

models = {
    "Logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced",
                                            min_samples_leaf=5, n_jobs=-1),
}

# Main split: stratified random, same as the baseline
X = df[NUM + CAT]; y = df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

rows, probs, cms, fitted = [], {}, {}, {}
for name, est in models.items():
    pipe = build(est)
    row, p, cm = evaluate(name, pipe, X_tr, y_tr, X_te, y_te)
    rows.append(row); probs[name] = p; cms[name] = cm; fitted[name] = pipe

results = pd.DataFrame(rows).round(3)
print("=== Random split (75/25, stratified) ===")
print(results.to_string(index=False))
for name in models:
    print(f"\nConfusion matrix, {name} [[TN FP] [FN TP]]:\n{cms[name]}")

# Time-based check: train on the first 75% of invoices by date, test on the last 25%
cut = df["issue_date"].quantile(0.75)
tr, te = df[df["issue_date"] <= cut], df[df["issue_date"] > cut]
trows = []
for name, est in models.items():
    row, _, _ = evaluate(name, build(est), tr[NUM + CAT], tr[TARGET], te[NUM + CAT], te[TARGET])
    trows.append(row)
print(f"\n=== Time-based split (train before {cut.date()}, test after) ===")
print(pd.DataFrame(trows).round(3).to_string(index=False))

# Feature importance from the random forest (permutation on the test set, ROC-AUC drop)
from sklearn.inspection import permutation_importance
rf = fitted["Random forest"]
imp = permutation_importance(rf, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X_te.columns).sort_values(ascending=False)
print("\nPermutation importance (drop in ROC-AUC):\n", imp_s.round(4).to_string())

# Chart 1: ROC curves
fig, ax = plt.subplots(figsize=(5, 4.5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the held-out test set"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("roc_curves.png", dpi=150)

# Chart 2: feature importance
fig, ax = plt.subplots(figsize=(6, 4))
imp_s.sort_values().plot.barh(ax=ax, color="#1F3A5F")
ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("What drives late-payment predictions (random forest)")
fig.tight_layout(); fig.savefig("feature_importance.png", dpi=150)
print("\nSaved roc_curves.png and feature_importance.png")

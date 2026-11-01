"""Clean the raw invoice ledger, build features, train the logistic regression baseline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = {}
df = pd.read_csv("raw_invoices.csv")
log["raw_rows"] = len(df)

# 1. Duplicates
before = len(df)
df = df.drop_duplicates(subset="invoice_id")
log["duplicates_removed"] = before - len(df)

# 2. Amount: strip currency symbol, convert to number, fix sign errors
df["amount"] = df["amount"].astype(str).str.replace("£", "", regex=False).astype(float)
log["negative_amounts_fixed"] = int((df["amount"] < 0).sum())
df["amount"] = df["amount"].abs()

# 3. Dates
for c in ["issue_date", "due_date", "paid_date"]:
    df[c] = pd.to_datetime(df[c])
log["due_before_issue"] = int((df["due_date"] < df["issue_date"]).sum())

# 4. Missing values
log["missing_region"] = int(df["region"].isna().sum())
df["region"] = df["region"].fillna("Unknown")
log["missing_credit_limit"] = int(df["credit_limit"].isna().sum())
df["credit_limit"] = df.groupby("segment")["credit_limit"].transform(lambda s: s.fillna(s.median()))

print("Cleaning log:", log)

# Features (only information known at invoice issue time, so no leakage from paid_date)
df = df.sort_values(["customer_id", "issue_date"]).reset_index(drop=True)
df["amount_log"] = np.log1p(df["amount"])
df["credit_utilisation"] = df["amount"] / df["credit_limit"]
df["issue_month"] = df["issue_date"].dt.month
# Customer's late rate on their earlier invoices only (expanding mean, shifted so the current invoice is excluded)
df["prior_late_rate"] = (df.groupby("customer_id")["paid_late"]
                           .transform(lambda s: s.shift(1).expanding().mean()))
df["prior_late_rate"] = df["prior_late_rate"].fillna(df["paid_late"].mean())
df["prior_invoice_count"] = df.groupby("customer_id").cumcount()

FEATURES_NUM = ["amount_log", "credit_utilisation", "payment_terms_days", "disputed",
                "prior_late_rate", "prior_invoice_count", "issue_month"]
FEATURES_CAT = ["segment", "region"]
TARGET = "paid_late"

df.to_csv("clean_invoices.csv", index=False)
print("Clean rows:", len(df), "| late rate:", round(df[TARGET].mean(), 3))

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])
model = make_pipeline(prep, LogisticRegression(max_iter=1000, class_weight="balanced"))
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("\n=== Logistic regression baseline (threshold 0.5, class_weight=balanced) ===")
print("ROC-AUC:  ", round(roc_auc_score(y_test, proba), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall:   ", round(recall_score(y_test, pred), 3))
print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))

coefs = pd.Series(model[-1].coef_[0], index=model[0].get_feature_names_out()).sort_values(key=np.abs, ascending=False)
print("\nTop coefficients (standardised):\n", coefs.head(10).round(3).to_string())

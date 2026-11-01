"""Generate a synthetic accounts-receivable invoice ledger with known drivers.

Late payment is simulated from customer risk, disputes, invoice size and terms,
so the analysis can be checked against relationships we built in on purpose.
Run: python3 generate_invoices.py  -> writes raw_invoices.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_CUSTOMERS, N_INVOICES = 300, 5000

# Customers
segments = rng.choice(["Small", "Mid", "Large"], N_CUSTOMERS, p=[0.5, 0.35, 0.15])
regions = rng.choice(["UK", "EU", "Middle East", "Asia"], N_CUSTOMERS, p=[0.4, 0.3, 0.2, 0.1])
risk = rng.normal(0, 1, N_CUSTOMERS)  # latent customer payment risk
customers = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(N_CUSTOMERS)],
    "segment": segments,
    "region": regions,
    "credit_limit": np.select(
        [segments == "Small", segments == "Mid"], [rng.normal(15000, 4000, N_CUSTOMERS).clip(2000),
                                                    rng.normal(60000, 15000, N_CUSTOMERS).clip(5000)],
        default=rng.normal(250000, 60000, N_CUSTOMERS).clip(20000)),
    "risk": risk,
})

# Invoices
idx = rng.integers(0, N_CUSTOMERS, N_INVOICES)
inv = customers.iloc[idx].reset_index(drop=True)
inv.insert(0, "invoice_id", [f"INV{i:06d}" for i in range(N_INVOICES)])
base_amount = np.select([inv["segment"] == "Small", inv["segment"] == "Mid"], [1500, 9000], default=45000)
inv["amount"] = np.round(base_amount * rng.lognormal(0, 0.6, N_INVOICES), 2)
inv["payment_terms_days"] = rng.choice([14, 30, 45, 60], N_INVOICES, p=[0.15, 0.45, 0.25, 0.15])
start = pd.Timestamp("2023-01-01")
inv["issue_date"] = start + pd.to_timedelta(rng.integers(0, 3 * 365, N_INVOICES), unit="D")
inv["due_date"] = inv["issue_date"] + pd.to_timedelta(inv["payment_terms_days"], unit="D")
inv["disputed"] = rng.binomial(1, np.clip(0.05 + 0.10 * (inv["risk"] > 0.5), 0, 1))

# Late payment: logit driven by customer risk, disputes, big invoices, long terms
logit = (-1.4 + 0.9 * inv["risk"] + 1.1 * inv["disputed"]
         + 0.6 * (inv["amount"] > 50000) + 0.4 * (inv["payment_terms_days"] >= 45)
         + 0.2 * (inv["region"] == "Middle East"))
p_late = 1 / (1 + np.exp(-logit))
inv["paid_late"] = rng.binomial(1, p_late)

# Days paid past due: late invoices spread out, on-time ones paid early or on time
days_late = np.where(inv["paid_late"] == 1, rng.gamma(2.0, 12.0, N_INVOICES) + 5, rng.normal(-3, 4, N_INVOICES))
inv["paid_date"] = inv["due_date"] + pd.to_timedelta(np.round(days_late).astype(int), unit="D")

# Keep only what a real ledger export would have
inv = inv.drop(columns=["risk", "p_late"], errors="ignore")
inv = inv[["invoice_id", "customer_id", "segment", "region", "credit_limit", "amount",
           "payment_terms_days", "issue_date", "due_date", "paid_date", "disputed", "paid_late"]]

# Realistic messiness so the cleaning step has real work to do
dirty = inv.copy()
dirty["amount"] = dirty["amount"].astype(object)
dirty["region"] = dirty["region"].astype(object)
dirty.loc[rng.choice(N_INVOICES, 150, replace=False), "credit_limit"] = np.nan
dirty.loc[rng.choice(N_INVOICES, 60, replace=False), "region"] = None
sel = rng.choice(N_INVOICES, 40, replace=False)
dirty.loc[sel, "amount"] = dirty.loc[sel, "amount"].map(lambda v: f"£{v:.2f}")
dirty = pd.concat([dirty, dirty.sample(25, random_state=1)], ignore_index=True)  # duplicate rows
neg = rng.choice(len(dirty), 15, replace=False)
dirty.loc[neg, "amount"] = dirty.loc[neg, "amount"].map(lambda v: -abs(float(str(v).replace("£", ""))))  # sign errors
dirty.to_csv("raw_invoices.csv", index=False)
print(f"raw rows written: {len(dirty)}, late rate: {inv['paid_late'].mean():.3f}")

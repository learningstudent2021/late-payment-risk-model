# Late-Payment Risk Model for Accounts Receivable

A model that estimates whether an invoice will be paid late, using only information known when the invoice is issued. It compares a logistic regression baseline with a random forest and reports which factors matter most.

**Data note:** the dataset is synthetic. It was generated with known relationships between customer risk, disputes, invoice size, payment terms and late payment, so the analysis can be checked against them. The results show that the method works. They do not show how the model would perform on real invoices.

## Files

| File | Purpose |
|---|---|
| `generate_invoices.py` | Generates the synthetic ledger (5,000 invoices, seeded) and adds realistic errors |
| `clean_and_baseline.py` | Cleans the raw data, builds features, trains the logistic regression baseline |
| `compare_models.py` | Compares both models on a random split and a time-based split, computes feature importance and charts |
| `roc_curves.png`, `feature_importance.png` | Charts |

Run in order: `generate_invoices.py`, `clean_and_baseline.py`, `compare_models.py`.

## Data cleaning

The raw ledger had 5,025 rows. Cleaning fixed:

- 25 duplicate invoice rows, removed
- 15 negative amounts, converted to absolute values
- 60 missing regions, set to "Unknown"
- 150 missing credit limits, filled with the segment median
- Amounts stored as text with a £ sign, converted to numbers

After cleaning there were 5,000 invoices, and 28.3% were paid late.

## Features

Features use only what is known when an invoice is issued:

- Invoice amount (log) and credit utilisation (amount as a share of the customer's credit limit)
- Payment terms, disputed flag, issue month
- Customer segment and region
- The customer's late-payment rate on earlier invoices, and how many earlier invoices they have

Settlement date, days to settle and days late are outcomes, so they are excluded.

## Results

Both models were trained on 75% of invoices and tested on the remaining 25%, stratified by outcome.

| Model | ROC-AUC | Precision | Recall |
|---|---|---|---|
| Logistic regression | 0.699 | 0.437 | 0.561 |
| Random forest | 0.689 | 0.445 | 0.518 |

Precision and recall use a 0.5 threshold, and the logistic regression uses balanced class weights.

A time-based check (training on earlier invoices, testing on later ones) gave similar results: logistic regression ROC-AUC 0.694, random forest 0.693. The model does not get worse on later invoices.

**The random forest did not outperform the logistic regression.** With this data and this set of features, the simpler model is as good.

## What drives late payment

Permutation importance from the random forest (the drop in ROC-AUC when each feature is shuffled):

| Feature | Importance |
|---|---|
| Customer's earlier late-payment rate | 0.218 |
| Region | 0.073 |
| Invoice amount (log) | 0.064 |
| Credit utilisation | 0.057 |
| Payment terms | 0.052 |
| Issue month | 0.047 |
| Prior invoice count | 0.043 |
| Segment | 0.038 |
| Disputed | 0.027 |

**Findings**

1. A customer's payment history is by far the strongest predictor. Customers who paid late before are much more likely to pay late again.
2. Region, invoice size relative to credit limit, and payment terms each add a smaller amount of signal.
3. Disputes matter less in the model than the simulation design suggests. The disputed flag is correlated with customer history, so the model gets much of that signal through the history feature. This is a limitation of the data, not evidence that disputes are unimportant.

## Limitations

- **Synthetic data.** The relationships were built into the data, so the model is recovering rules I wrote. Real invoices would be messier.
- **Modest performance.** An ROC-AUC of about 0.70 means the model ranks risky invoices above safe ones more often than chance, but it is far from reliable enough for credit decisions.
- **No threshold tuning.** Precision and recall use a 0.5 threshold. A business would choose the threshold based on the cost of chasing a late invoice against the cost of missing one.
- **Limited feature set.** The model has no information on invoice size history or payment method beyond what is listed above.

## Next steps

- Test the pipeline on a real invoice-level dataset.
- Tune the random forest and compare against gradient boosting.
- Choose a decision threshold based on business costs.

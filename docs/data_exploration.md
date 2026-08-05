# PaySim Data Exploration

## Dataset snapshot

- Source file: `PS_20174392719_1491204439457_log.csv`
- File size: 493,534,783 bytes (470.67 MiB)
- Shape: 6,362,620 rows and 11 columns
- Simulated time: steps 1 through 743, where each step represents one hour
- Missing values: none in any column
- Negative monetary values: none
- Zero transaction amounts: 16; all are labelled fraudulent `CASH_OUT` transactions

The zero-amount rows will be retained. Deleting them as generic outliers would remove labelled fraud examples. Zero balances are also common in PaySim and must not be interpreted as missing values without considering the transaction type and account role.

The raw CSV is stored locally under `data/raw/` and is excluded from Git. The generated profile in `reports/data_exploration_summary.json` is safe to commit because it contains only aggregate statistics.

## Fraud distribution

The dataset contains 8,213 labelled fraud transactions, or approximately 0.129% of all transactions. This is a severely imbalanced classification problem: a model that predicts every transaction as legitimate would exceed 99.8% accuracy while detecting no fraud.

| Transaction type | Transactions | Fraud transactions | Fraud rate |
| --- | ---: | ---: | ---: |
| CASH_IN | 1,399,284 | 0 | 0.000% |
| CASH_OUT | 2,237,500 | 4,116 | 0.184% |
| DEBIT | 41,432 | 0 | 0.000% |
| PAYMENT | 2,151,495 | 0 | 0.000% |
| TRANSFER | 532,909 | 4,097 | 0.769% |

All labelled fraud occurs in `TRANSFER` and `CASH_OUT`. This is useful context, but transaction type alone must not be treated as proof of fraud.

Only 16 transactions are marked by `isFlaggedFraud`, while 8,213 are labelled by `isFraud`. The existing flag therefore does not identify most simulated fraud and should be evaluated separately rather than treated as the target prediction.

## Initial data decisions

1. Preserve the raw CSV unchanged and perform all cleaning in reproducible code.
2. Validate the 11 expected columns and binary label values before processing.
3. Use stratified train, validation, and test splits so each split contains fraud examples.
4. Evaluate precision, recall, F1, precision-recall AUC, and confusion matrices rather than relying on accuracy.
5. Keep the chronological `step` field available for a later time-based validation comparison.
6. Treat customer and recipient names as identifiers, not ordinary numeric or categorical features.
7. Investigate balance columns for target leakage before using them in a model, because some balances may reflect transaction outcomes.
8. Measure exact duplicate rows during database-quality validation before training.

## Reproduce the profile

From the repository root, run:

```powershell
python src/data/explore_paysim.py data/raw/PS_20174392719_1491204439457_log.csv
```

The script reads the CSV in chunks, so it does not need to hold the complete dataset in memory.

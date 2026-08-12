# PaySim Exploratory Data Analysis Findings

## Purpose

This report records the descriptive findings reproduced by `notebooks/01_paysim_exploration.ipynb`. It does not train, select, or evaluate a fraud model.

The report deliberately distinguishes official dataset facts, findings calculated by this project, and modelling constraints caused by leakage or simulator artifacts.

## Official dataset facts

According to the [Kaggle PaySim documentation](https://www.kaggle.com/datasets/ealaxi/paysim1/data) and the [PaySim paper](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf):

- PaySim is a synthetic mobile-money simulator informed by aggregated transaction data.
- The Kaggle dataset represents approximately one month of activity.
- One `step` represents one simulated hour.
- The transaction types are `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`.
- Fraudulent agents attempt to empty compromised accounts through transfers and cash-outs.
- Detected fraud transactions are cancelled.
- The four balance fields must not be used for fraud detection.
- Destination balances are not maintained for merchant identifiers beginning with `M`.

## Findings from our analysis

### Dataset shape and quality

- File size: 493,534,783 bytes (470.67 MiB).
- Shape: 6,362,620 transactions and 11 columns.
- Local step range: 1 through 743, covering approximately 31 simulated days.
- Missing values: none.
- Negative transaction amounts or balances: none.
- Sixteen transactions have an amount of zero; all are fraudulent `CASH_OUT` transactions and are retained.

### Class imbalance

| Label | Count | Share |
| --- | ---: | ---: |
| Legitimate | 6,354,407 | 99.8709% |
| Fraud | 8,213 | 0.1291% |

Accuracy alone would be misleading: predicting every transaction as legitimate would exceed 99.8% accuracy while detecting no fraud.

### Fraud by transaction type

| Type | Legitimate | Fraud | Fraud rate within type |
| --- | ---: | ---: | ---: |
| `CASH_IN` | 1,399,284 | 0 | 0.0000% |
| `CASH_OUT` | 2,233,384 | 4,116 | 0.1840% |
| `DEBIT` | 41,432 | 0 | 0.0000% |
| `PAYMENT` | 2,151,495 | 0 | 0.0000% |
| `TRANSFER` | 528,812 | 4,097 | 0.7688% |

All labelled fraud occurs in `TRANSFER` and `CASH_OUT`. Transaction type is informative but is not proof that an individual transaction is fraudulent.

### Existing flag overlap

| Actual label | Not flagged | Flagged |
| --- | ---: | ---: |
| Legitimate | 6,354,407 | 0 |
| Fraud | 8,197 | 16 |

All 16 flagged transactions are fraudulent, but the rule identifies only 0.195% of the 8,213 fraud cases.

### Transaction amounts

| Label | Minimum | Maximum | Median | Mean |
| --- | ---: | ---: | ---: | ---: |
| Legitimate | 0.01 | 92,445,516.64 | 74,684.72 | 178,197.04 |
| Fraud | 0.00 | 10,000,000.00 | 441,423.44 | 1,467,967.30 |

Fraud transactions are generally larger, but amount ranges overlap substantially.

### Time artifacts

Daily fraud counts are comparatively stable while total transaction volume varies sharply. This produces uneven daily fraud rates. Day 31 contains only 272 transactions and all are labelled fraud, making the end of the simulation a strong artifact rather than a general business pattern.

### Identifier behaviour

| Metric | Origin (`nameOrig`) | Destination (`nameDest`) |
| --- | ---: | ---: |
| Unique identifiers | 6,353,307 | 2,722,362 |
| Identifiers appearing more than once | 9,298 | 459,658 |
| Maximum appearances | 3 | 113 |

Origins are almost always unique. Destinations repeat much more frequently, which creates memorisation risk if the same destination occurs in both training and test data.

### Balance leakage evidence — analysis only

> **The following balance fields and every feature derived from them are analysis only and must not be used as predictive features:** `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`.

Origin balances reconcile exactly for 99.45% of fraud transactions but only 38.01% of legitimate transactions. Destination balances for customer recipients reconcile exactly for 40.58% of fraud transactions and 65.90% of legitimate transactions.

The strongest zero-balance difference occurs in `newbalanceOrig`: it is zero in 98.05% of fraud transactions and 56.68% of legitimate transactions. These patterns nearly expose the target and demonstrate PaySim processing behavior rather than trustworthy real-time fraud signals.

## Modelling constraints for later phases

No modelling occurs in Phase 2. Later modelling must follow these constraints:

1. Exclude all four balance fields and every derived balance feature from prediction.
2. Exclude `isFraud` from inputs because it is the target.
3. Exclude `isFlaggedFraud` from an independent detector because it is an existing rule output.
4. Do not use raw `nameOrig` or `nameDest` identifiers as features.
5. Treat `step` as a restricted candidate and use chronological validation to expose end-of-simulation artifacts.
6. Evaluate imbalanced-data metrics such as precision, recall, F1, precision-recall AUC, and confusion matrices rather than accuracy alone.

## Figures

1. [`01_transaction_type_distribution.png`](figures/01_transaction_type_distribution.png)
2. [`02_fraud_by_transaction_type.png`](figures/02_fraud_by_transaction_type.png)
3. [`03_amount_distribution_by_fraud.png`](figures/03_amount_distribution_by_fraud.png)
4. [`04_daily_volume_and_fraud_rate.png`](figures/04_daily_volume_and_fraud_rate.png)
5. [`05_balance_leakage_evidence.png`](figures/05_balance_leakage_evidence.png) — analysis-only leakage demonstration

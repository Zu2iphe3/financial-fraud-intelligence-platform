# Leakage-safe behavioural modelling dataset

## Purpose

`paysim_features_v2` tests whether strictly past destination behaviour improves the
honest `paysim_features_v1` baseline. It preserves the whole-day chronological split
and never uses balances, fraud flags, raw identifiers, labels, the current PaySim step,
the Day 31 indicator, or future activity as predictive inputs.

This is validation-stage evidence. Earlier v1 PH7-02 work evaluated the test partition.
That v1 test information was not used to select or tune the behavioural v2 model. Final
v2 feature selection and XGBoost tuning used train and validation data only. After the
model, feature contract, and `0.49207180738449097` threshold were frozen, they were
evaluated once on the v2 chronological test partition. The test result did not change
the selected model, features, hyperparameters, or threshold.

## Historical boundary

For a transaction at step `s`, 24-hour features use steps `s-24` through `s-1` and
168-hour features use steps `s-168` through `s-1`. All transactions at step `s` are
calculated before that step updates history because PaySim provides no ordering within
an hour.

Raw origin and destination identifiers are temporary grouping keys only. The generator
buckets transactions by destination, computes numeric history, joins by the existing
`source_row_number`, and removes the identifiers before writing model features.

## Finalized feature contract

The contract contains 20 predictive features: the six v1 amount/type features plus
14 destination-history features covering transaction counts, amount totals, unique
origins, time since prior activity, amount deviation from the prior 168-hour mean, and
recent counts for each transaction type. Count and amount-total features use `log1p`.

The complete contract is defined in
[`config/features/paysim_v2.json`](../config/features/paysim_v2.json).

Two proposed relationship features were excluded because no training or validation row
had a prior occurrence of the same origin-destination pair, making both fields constant.
The reviewed Isolation Forest candidate was also excluded: it improved validation
PR-AUC by only `0.003939` while reducing every measured daily top-k result. No anomaly
score or relationship field appears in the finalized predictive contract.

## Fair baseline comparison

Both feature versions use the same no-skill benchmark, balanced Logistic Regression,
and untuned XGBoost configuration. Logistic Regression includes `StandardScaler` fitted
on training days 1-18 only. With a 1,000-iteration limit, v1 converged in 16 iterations
and v2 in 34; validation data did not fit the scaler.

| Feature set and model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 Logistic Regression | 0.074576 | 0.892206 | 0.058627 | 0.729551 | 0.108532 | 1,106 | 17,759 | 410 |
| v1 XGBoost | 0.296950 | 0.922749 | 0.791155 | 0.212401 | 0.334893 | 322 | 85 | 1,194 |
| v2 Logistic Regression | 0.222619 | 0.917839 | 0.341801 | 0.292876 | 0.315453 | 444 | 855 | 1,072 |
| v2 XGBoost (untuned) | 0.485619 | 0.960662 | 0.350284 | 0.690633 | 0.464817 | 1,047 | 1,942 | 469 |

Threshold-level values use each model's validation maximum-F1 point. They describe a
trade-off, not an automatically approved production policy.

## Controlled v2 XGBoost optimization

One bounded 12-candidate search varied important tree, regularization, sampling, and
class-weight settings. Every candidate used training days 1-18 for fitting, validation
days 19-24 for early stopping and selection, seed 42, and validation PR-AUC as the
primary optimization metric. Every run was tracked in MLflow. The test partition was
not loaded.

The winner uses depth 6, minimum child weight 30, learning rate 0.05, 80% row sampling,
all feature columns, L1 regularization 0.1, L2 regularization 5.0, and
`scale_pos_weight=50`. Early stopping selected iteration 95.

| v2 model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.222619 | 0.917839 | 0.341801 | 0.292876 | 0.315453 | 444 | 855 | 1,072 |
| Untuned XGBoost | 0.485619 | 0.960662 | 0.350284 | 0.690633 | 0.464817 | 1,047 | 1,942 | 469 |
| Tuned XGBoost | 0.487886 | 0.962215 | 0.344041 | 0.706464 | 0.462735 | 1,071 | 2,042 | 445 |

Tuning adds `0.002266` absolute PR-AUC (about 0.47%) over untuned v2. At each model's
maximum-F1 point, the tuned model finds 24 more frauds but creates 100 more false alerts
and has slightly lower F1. This is a modest ranking improvement, not evidence that the
maximum-F1 threshold is automatically the right operational choice.

## Train and validation diagnostics

Before opening test data, the stored tuned v2 model was evaluated diagnostically on train and validation using
the same unchanged validation maximum-F1 review threshold (`0.492072`). This threshold
remains a comparison point, not a production policy. No retraining was performed.

| Partition | Fraud rate | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0.0808% | 0.236453 | 0.966912 | 0.049434 | 0.691785 | 0.092273 |
| Validation | 0.7864% | 0.487886 | 0.962215 | 0.344041 | 0.706464 | 0.462735 |

Validation contains about 9.7 times the fraud prevalence of training, so its higher
PR-AUC, precision, and F1 are not directly evidence of better fitting. The more directly
comparable ROC-AUC differs by only `0.0047`, and recall differs by only `0.0147`. There
is no obvious train/validation overfitting signal.

Validation recall by day was `71.88%`, `71.19%`, `72.43%`, `69.53%`, `68.06%`, and
`70.36%` for days 19 through 24 respectively. The approximately 68–72% range shows that
overall validation recall is not dominated by one day. Daily precision and PR-AUC vary
more because the daily fraud rate changes substantially.

## Final frozen-model held-out evaluation

The persisted `paysim_fraud_model_v2_tuned_review` artifact and its unchanged
`0.49207180738449097` threshold were evaluated once on steps 577-743 (days 25-31).
Before the test partition was loaded, the evaluator verified the promoted run ID,
feature-contract digest, validation-selected threshold, and the original search and
visual manifests recording `test_partition_loaded=false`.

PR-AUC and ROC-AUC are threshold-independent ranking measures. Precision, recall, F1,
accuracy, alerts, and confusion-matrix counts below use the frozen threshold.

| Test cohort | Rows | Fraud | Fraud rate | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full held-out test, days 25-31 | 161,426 | 1,840 | 1.1398% | 0.539848 | 0.957729 | 0.289325 | 0.752717 | 0.417987 | 1,385 | 3,402 | 455 | 156,184 |
| Standard period, days 25-30 | 161,154 | 1,568 | 0.9730% | 0.519707 | 0.959165 | 0.258339 | 0.755740 | 0.385053 | 1,185 | 3,402 | 383 | 156,184 |
| Day 31 simulator artifact | 272 | 272 | 100% | Not informative | Undefined | 1.000000 | 0.735294 | 0.847458 | 200 | 0 | 72 | 0 |

Compared with validation, full-test PR-AUC increased from `0.487886` to `0.539848`,
ROC-AUC decreased slightly from `0.962215` to `0.957729`, and recall increased from
`70.65%` to `75.27%`. Precision fell from `34.40%` to `28.93%` and F1 from `0.462735`
to `0.417987` because the unchanged threshold generated more false-positive alerts in
the later period. This is mixed generalisation evidence rather than uniform degradation:
ranking and recall remained strong, while threshold-level precision and F1 weakened.
No model or threshold change was made after observing test results.

### Tuned XGBoost recall targets

| Policy | Threshold | Precision | Recall | F1 | TP | FP | FN | Alerts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| At least 50% recall | 0.600361 | 0.424888 | 0.500000 | 0.459394 | 758 | 1,026 | 758 | 1,784 |
| Maximum F1 | 0.492072 | 0.344041 | 0.706464 | 0.462735 | 1,071 | 2,042 | 445 | 3,113 |
| At least 75% recall | 0.162699 | 0.270630 | 0.750660 | 0.397833 | 1,138 | 3,067 | 378 | 4,205 |
| At least 80% recall | 0.061293 | 0.120925 | 0.800132 | 0.210098 | 1,213 | 8,818 | 303 | 10,031 |
| At least 90% recall | 0.049135 | 0.041391 | 0.904354 | 0.079159 | 1,371 | 31,752 | 145 | 33,123 |

### Daily alert capacity

Validation covers six simulation days, so total alerts equal six times the daily limit.

| Model | Alerts/day | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 100 | 0.498333 | 0.197230 | 0.282609 | 299 | 301 | 1,217 |
| Untuned XGBoost | 100 | 0.741667 | 0.293536 | 0.420605 | 445 | 155 | 1,071 |
| Tuned XGBoost | 100 | 0.746667 | 0.295515 | 0.423440 | 448 | 152 | 1,068 |
| Logistic Regression | 500 | 0.190333 | 0.376649 | 0.252879 | 571 | 2,429 | 945 |
| Untuned XGBoost | 500 | 0.344667 | 0.682058 | 0.457927 | 1,034 | 1,966 | 482 |
| Tuned XGBoost | 500 | 0.344000 | 0.680739 | 0.457042 | 1,032 | 1,968 | 484 |
| Logistic Regression | 1,000 | 0.127000 | 0.502639 | 0.202767 | 762 | 5,238 | 754 |
| Untuned XGBoost | 1,000 | 0.197500 | 0.781662 | 0.315327 | 1,185 | 4,815 | 331 |
| Tuned XGBoost | 1,000 | 0.197500 | 0.781662 | 0.315327 | 1,185 | 4,815 | 331 |
| Logistic Regression | 2,000 | 0.083417 | 0.660290 | 0.148121 | 1,001 | 10,999 | 515 |
| Untuned XGBoost | 2,000 | 0.103750 | 0.821240 | 0.184226 | 1,245 | 10,755 | 271 |
| Tuned XGBoost | 2,000 | 0.104750 | 0.829156 | 0.186002 | 1,257 | 10,743 | 259 |

## Reproduce locally

Generate the reviewed v1 dataset first, then run:

```powershell
python -m src.features.build_paysim_dataset_v2
python -m src.models.compare_paysim_features
python -m src.models.tune_xgboost --model-config config/models/paysim_fraud_v2.json --search-config config/models/paysim_xgboost_search_v2.json
```

Generated datasets, review outputs, and MLflow runs remain excluded from Git.

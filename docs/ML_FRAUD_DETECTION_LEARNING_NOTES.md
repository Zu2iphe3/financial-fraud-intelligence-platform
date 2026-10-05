# ML fraud detection learning notes

## What problem are we solving?

We want to rank PaySim transactions by how suspicious they appear. Fraud is rare, so
the model must find a small number of fraudulent transactions among millions of normal
ones. This is different from an ordinary balanced classification exercise: a model can
be more than 99% accurate by predicting almost everything as legitimate and still be
useless for fraud detection.

This phase is a controlled modelling study, not a deployed fraud decision system.
Earlier v1 PH7-02 work evaluated the test partition. That v1 test information was not
used to select or tune the behavioural v2 model. Final v2 feature selection and XGBoost
tuning used train and validation data only. After the v2 model and threshold were
frozen, one final held-out test evaluation was
performed without changing the selected model, features, tuning, or threshold.

## V1: the honest minimal baseline

V1 used only six predictive columns:

- the transaction amount after a `log1p` transformation;
- one yes/no column for each transaction type: `CASH_IN`, `CASH_OUT`, `DEBIT`,
  `PAYMENT`, and `TRANSFER`.

`log1p` compresses PaySim's very large amount range so extremely large transactions do
not dominate the numeric scale. One-hot type columns let models treat the five
transaction categories separately without pretending that one category is numerically
larger than another.

V1 was deliberately small and leakage-safe. Its best XGBoost validation PR-AUC was
`0.296950`, but its maximum-F1 review point found only about 21% of fraud. The main
problem was not simply the algorithm: amount and transaction type do not describe how
a recipient has behaved over time.

## Class imbalance

The training partition contains 6,008,416 transactions but only 4,857 fraud cases.
There are roughly 1,236 legitimate transactions for each fraud transaction.

We did not discard most legitimate rows and did not duplicate fraud rows. Logistic
Regression uses `class_weight="balanced"`. XGBoost experiments used
`scale_pos_weight` values to make fraud errors matter more during fitting. The v1
class-weight experiment demonstrated an important lesson: weighting can help a model
pay attention to rare fraud, but it cannot create behavioural information that is not
present in the features.

## Logistic Regression

Logistic Regression is our interpretable baseline. It learns a weighted linear
combination of the input features and converts that score into a fraud probability.
It provides a useful check: if a complicated tree model cannot beat a well-prepared
linear model, the extra complexity may not be justified.

The initial v2 Logistic Regression reached its iteration limit because behavioural
features have very different numeric ranges. We fixed this using `StandardScaler`
inside the fitted model pipeline and raised the reasonable iteration ceiling to 1,000.
The scaler learns means and standard deviations from training days only. V1 then
converged in 16 iterations and v2 in 34.

## XGBoost

XGBoost combines many decision trees. Each new tree concentrates on errors from earlier
trees. It is appropriate here because fraud patterns can involve non-linear
combinations—for example, a large amount may mean something different for a destination
with no history than for a frequently used destination.

Important settings include:

- tree depth: how complex each tree can become;
- learning rate: how strongly each new tree changes the model;
- minimum child weight: how much evidence is required before making a split;
- row/column sampling: controlled randomness that can reduce overfitting;
- L1/L2 regularization: penalties that discourage unnecessary complexity;
- `scale_pos_weight`: the relative importance assigned to rare fraud examples;
- early stopping: stop adding trees when validation PR-AUC stops improving.

## Hyperparameter tuning

Hyperparameters control how a model learns; they are not learned directly from the
rows. We ran one bounded 12-candidate v2 XGBoost search rather than an endless search.
Candidates varied the important settings above. All were trained on days 1-18 and
compared on days 19-24 using validation PR-AUC.

The tuned winner achieved validation PR-AUC `0.487886`, compared with `0.485619` for
untuned v2 XGBoost. This is a real but modest improvement. The tuned maximum-F1 review
point has slightly higher recall and more false positives, so tuning did not remove the
need for an operational threshold decision.

## Train, validation, and test

- **Train, days 1-18:** fit scalers and model parameters.
- **Validation, days 19-24:** compare models, stop XGBoost training, compare thresholds,
  and choose modelling direction.
- **Test, days 25-31:** evaluate the final frozen model once. Behavioural v2 test data
  was opened only after its model, features, hyperparameters, and threshold were locked.
  Day 31 remains a separately identified simulator-artifact cohort.

Keeping these roles separate prevents us from repeatedly checking test results and
quietly adapting the model to the test set.

## Why the split is chronological

In a real fraud system, a model learns from the past and predicts future transactions.
A random split would mix later transactions into training and earlier transactions into
validation. Chronological whole-day boundaries better represent how the system would
operate and prevent information from the same simulated day crossing partitions.

## Leakage and prohibited PaySim fields

Leakage happens when a predictive feature contains information that would not genuinely
be available at decision time or indirectly reveals the answer. PaySim documentation
warns that fraudulent transactions are cancelled. Therefore these balance fields are
analysis-only and prohibited from modelling:

- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

We also prohibit `isFlaggedFraud`, fraud labels, raw origin/destination IDs, source-row
identity, future activity, the current step, and the Day 31 indicator as predictive
inputs. Raw IDs may exist temporarily only to calculate past behaviour; they never enter
the model matrix.

## Metrics in plain language

- **Precision:** Of the transactions we alerted on, how many were actually fraud?
- **Recall:** Of all fraud transactions, how many did we find?
- **F1:** A single balance between precision and recall.
- **PR-AUC:** Measures fraud-ranking quality across many thresholds. It is especially
  useful when fraud is rare, but it does not by itself prove operational usefulness.
- **ROC-AUC:** Measures general class ranking and is secondary here because it can look
  optimistic with very rare fraud.
- **False positive:** A legitimate transaction sent for fraud review.
- **False negative:** A fraud transaction the selected operating point missed.

Accuracy is informational only because predicting nearly every row as legitimate would
produce very high accuracy.

## Thresholds

The model produces a probability-like score. A threshold converts that score into an
alert. Lower thresholds usually find more fraud but create more false positives. Higher
thresholds reduce alerts but miss more fraud.

We report maximum-F1 and recall targets of at least 50%, 75%, 80%, and 90%. These are
review points, not a chosen production threshold. A real choice needs business costs,
fraud-loss tolerance, and investigator capacity.

## Daily top-k evaluation

A fraud team can review only a limited number of alerts each day. Top-k evaluation ranks
each day's transactions and measures results if analysts review the top 100, 500, 1,000,
or 2,000 scores. This makes model comparison concrete even before a fixed threshold is
chosen.

At 500 alerts per validation day, final v2 XGBoost finds about 68% of fraud with roughly
34% precision. The tuned and untuned models are almost identical at this capacity.

## Behavioural v2 features

V2 keeps the six v1 features and adds 14 destination-history features:

- transaction counts over the previous 24 and 168 hours;
- transaction-amount totals over the previous 24 and 168 hours;
- unique origin counts over the previous 24 and 168 hours;
- hours since the destination's previous activity;
- a flag saying whether destination history exists;
- current amount compared with the destination's earlier 168-hour mean;
- previous-24-hour counts for each of the five transaction types.

For a transaction at hour `s`, a 24-hour feature uses hours `s-24` through `s-1`; a
168-hour feature uses `s-168` through `s-1`. It never includes hour `s`. PaySim does not
provide an order within an hour, so all transactions in hour `s` are calculated before
any of them update the historical state. This is the central past-only leakage guard.

V2 increased XGBoost validation PR-AUC from `0.296950` to `0.485619`, a 63.5% relative
improvement. At the maximum-F1 review point, recall increased from about 21% to 69%.

## Relationship features

We tested past counts for the exact origin-destination pair. PaySim did not repeat a
pair in training or validation, so one field was always zero and the new-relationship
flag was always one. Constant features contain no ranking information and were removed
from the selected contract. The finding remains documented as experiment evidence.

## Isolation Forest and anomaly detection

Isolation Forest is an unsupervised algorithm: it searches for unusual patterns without
using fraud labels. Our experiment trained it only on historical training data and froze
the day-18 detector for validation.

Adding its percentile score increased PR-AUC slightly, from `0.485619` to `0.489558`,
but reduced precision and recall at every daily top-k capacity. Because the operational
evidence did not improve, the anomaly score was rejected from the selected model. The
experiment was still useful because it showed that “unusual” does not automatically
mean “better fraud alert.”

## MLflow

MLflow records each model run's parameters, metrics, run ID, threshold tables, top-k
tables, fitted candidate, feature importance, and review figures. This makes it possible
to compare experiments and explain why a candidate was selected or rejected. Local
MLflow state is generated and ignored by Git; committed configurations and documentation
describe how to reproduce it.

## Results so far

| Model | Validation PR-AUC | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| V1 Logistic Regression | 0.074576 | 0.058627 | 0.729551 | 0.108532 | 17,759 | 410 |
| V1 XGBoost | 0.296950 | 0.791155 | 0.212401 | 0.334893 | 85 | 1,194 |
| V2 Logistic Regression | 0.222619 | 0.341801 | 0.292876 | 0.315453 | 855 | 1,072 |
| V2 XGBoost | 0.485619 | 0.350284 | 0.690633 | 0.464817 | 1,942 | 469 |
| Tuned V2 XGBoost | 0.487886 | 0.344041 | 0.706464 | 0.462735 | 2,042 | 445 |

These threshold-level values use maximum-F1 validation review points. They are not a
production threshold decision.

The final frozen v2 model then achieved held-out test PR-AUC `0.539848`, ROC-AUC
`0.957729`, precision `0.289325`, recall `0.752717`, and F1 `0.417987` at the unchanged
`0.49207180738449097` threshold. It found 1,385 of 1,840 fraud cases, with 3,402 false
positives and 455 false negatives. Test evidence was reported, not used to revise the
model.

## Did v2 obviously overfit?

We applied the tuned v2 model's unchanged validation review threshold to both training
and validation. Training recall was `69.18%` and validation recall was `70.65%`;
ROC-AUC changed only from `0.9669` to `0.9622`. Validation recall also stayed between
approximately 68% and 72% on every day from day 19 through day 24. This provides no
obvious train/validation overfitting signal.

Training has a much lower fraud rate than validation, so raw PR-AUC, precision, and F1
cannot be compared as though both partitions had the same class balance. The diagnostic
supports reasonable train-to-validation generalization. The later held-out result adds
independent evidence: ranking and recall remained strong, while precision and F1 fell
at the frozen threshold.

## How to explain this work in an interview

> I started with a deliberately leakage-safe baseline using only transaction amount and
> type. Because PaySim fraud is extremely imbalanced, I evaluated PR-AUC, precision,
> recall, false positives, false negatives, threshold trade-offs, and daily alert
> capacity rather than relying on accuracy. I used chronological train/validation/test
> splits and kept PaySim balance fields out because the simulator documentation warns
> that cancelled fraud makes them leak the outcome. The baseline revealed a feature
> ceiling, so I built past-only destination features with 24-hour and 168-hour windows.
> Every transaction uses only earlier hourly steps. This improved XGBoost validation
> PR-AUC from about 0.297 to 0.486 and recall at the maximum-F1 review point from about
> 21% to 69%. I tracked controlled class-weight and hyperparameter experiments in MLflow,
> rejected constant relationship features and an Isolation Forest score when operational
> top-k results did not improve. Earlier v1 work had evaluated test after locking its
> choice, and I kept behavioural v2 test-isolated throughout feature selection and
> tuning. After freezing the v2 model and threshold, I evaluated them once on future
> days 25-31 and reported the mixed generalisation result without changing the model.

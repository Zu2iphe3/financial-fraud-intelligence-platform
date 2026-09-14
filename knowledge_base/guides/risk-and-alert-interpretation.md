# Risk and Alert Interpretation Guide

> **FICTIONAL PORTFOLIO DOCUMENT:** This original guide describes Harborline
> Pay, a fictional digital-payments company. It is educational material, not a
> real organization's procedure, legal advice, or authority to act on an account.

**Source ID:** `risk-alert-interpretation-guide-v1`  
**Version:** `1.0.0`

## RISK-01 - Purpose and boundaries

This guide helps fictional analysts interpret model output consistently. The
model prioritizes review; it does not establish intent, guilt, or a required
customer action. The API scores a leakage-safe feature vector prepared upstream
and returns a probability and risk classification.

## RISK-02 - What the fraud score means

A higher fraud probability means the model found the supplied feature pattern
more similar to fraud examples learned during training. It is not a guarantee
that the transaction is fraudulent, and probabilities may not be perfectly
calibrated to real-world rates.

A risk classification is produced by comparing the probability with a configured
operating threshold. Moving that threshold changes the number of alerts and the
balance between detected fraud and false alerts. This corpus does not choose a
production threshold.

## RISK-03 - Precision recall and investigation workload

- **Precision** asks: of the alerts raised, what proportion were actually fraud
  in the evaluated labelled data? Low precision creates more false alerts and
  more analyst work.
- **Recall** asks: of all fraud cases, what proportion did the model alert on?
  Low recall means more fraud cases were missed.
- **False positives** are legitimate transactions sent for review.
- **False negatives** are fraud transactions the chosen threshold did not flag.

Accuracy can appear high when fraud is rare, even for an unhelpful model. Fraud
review therefore considers precision, recall, F1, PR-AUC, confusion matrices,
and alert capacity together rather than relying on accuracy alone.

## RISK-04 - Interpreting behavioural features

Behavioural features summarize activity strictly before the transaction being
scored. Examples include recent transaction counts and amounts, unique recent
origins for a destination, time since earlier destination activity, historical
amount deviation, recent transaction-type counts, and relationship newness.

Each feature is context, not proof. A new destination may be legitimate, a burst
may reflect ordinary demand, and a large deviation may be a valid one-off
payment. Several consistent signals generally provide a stronger reason for
review than one isolated value.

## RISK-05 - Human review and decision limits

The analyst checks feature timing and quality, reviews supporting and conflicting
evidence, considers reasonable alternatives, and records uncertainty. The
analyst should not reverse-engineer a conclusion from the score or claim the
model observed facts that were not supplied.

Raw origin and destination identifiers must not be predictive inputs. Balance
fields, balance-derived values, fraud labels, `isFlaggedFraud`, future events,
and Day 31 timing are prohibited predictive inputs under the project's leakage
contract.

## RISK-06 - PaySim and model limitations

PaySim is synthetic and cannot represent every real customer, fraud method, data
quality issue, or change over time. Its fraud events are simulator outcomes, and
some balance fields reveal those outcomes. Results from this project demonstrate
an engineering and modelling workflow, not production effectiveness.

The selected behavioural model showed useful validation improvement, but model
quality may change on another period or population. Earlier v1 experiments and
rejected features remain learning evidence, not justification for expanding what
an analyst may conclude from a current alert.

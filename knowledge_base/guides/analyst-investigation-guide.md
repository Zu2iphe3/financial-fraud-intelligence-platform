# Fraud Analyst Investigation Guide

> **FICTIONAL PORTFOLIO DOCUMENT:** This original guide describes Harborline
> Pay, a fictional digital-payments company. It is educational material, not a
> real organization's procedure, legal advice, or authority to act on an account.

**Source ID:** `analyst-investigation-guide-v1`  
**Version:** `1.0.0`

## GUIDE-01 - How to use this guide

Use the scenario closest to the observed alert, then apply the Fraud
Investigation Policy and Fraud Escalation Policy. More than one scenario may be
relevant. Record actual observations and uncertainty rather than forcing an
alert to match a scenario.

## GUIDE-02 - Unusual transfer to a new destination

**Responsible fictional role:** Fraud Operations Analyst

### Investigation actions

1. Confirm that the relationship-newness indicator used only earlier events.
2. Compare the amount with the origin's earlier transaction amounts.
3. Review the destination's earlier transaction count, total amount, distinct
   origins, and recent transaction types.
4. Check whether other past-only signals support or contradict concern.
5. Record ordinary explanations, data gaps, and a reasoned outcome.

### Escalation conditions

Request a Senior Fraud Reviewer when the new relationship is combined with a
substantial amount deviation, an unusual destination pattern, or material
uncertainty. Use critical review only when evidence also suggests continuing or
coordinated fictional harm. Relationship newness alone is not enough.

## GUIDE-03 - Rapid activity involving one destination

**Responsible fictional role:** Fraud Operations Analyst

### Investigation actions

1. Review past-only destination counts and amounts over each available window.
2. Compare distinct recent origins with the destination's longer history.
3. Identify whether the activity is a short burst or established behaviour.
4. Compare transaction types and amounts without using later transactions.
5. Document whether several independent indicators tell a consistent story.

### Escalation conditions

Request priority review when a clear burst involves several origins or repeated
high-impact activity and lacks a reasonable explanation. Request critical
review when the recorded pattern appears coordinated and potentially ongoing.
Route impossible counts or inconsistent windows to the Model Risk Steward.

## GUIDE-04 - Amount outside recent behaviour

**Responsible fictional role:** Fraud Operations Analyst

### Investigation actions

1. Compare the amount with the origin's historical mean or other available
   past-only reference value.
2. Check whether the transaction type is normal for the origin and destination.
3. Review whether destination activity and relationship history support concern.
4. Consider that legitimate transactions can be unusually large.
5. Record the evidence for and against escalation.

### Escalation conditions

Request senior review when the amount deviation is accompanied by new or bursty
behaviour, or when potential impact is material and evidence remains uncertain.
An amount deviation without supporting evidence should not automatically be
treated as fraud.

## GUIDE-05 - High model risk with conflicting evidence

**Responsible fictional role:** Senior Fraud Reviewer

### Investigation actions

1. Verify that the feature vector follows the approved feature contract.
2. Separate model probability and risk classification from observed behaviour.
3. Identify which indicators support the alert and which contradict it.
4. Check for missing, stale, impossible, or incorrectly timed feature values.
5. Record whether the case concern and any model-quality concern are separate.

### Escalation conditions

Escalate a credible case pattern under the Fraud Escalation Policy even when
some indicators conflict. Refer systematic scoring or data problems to the Model
Risk Steward. Do not escalate a transaction as fraud solely because its model
probability is high.

## GUIDE-06 - Case-note checklist

A complete fictional case note answers:

- What triggered the review?
- Which facts were available at the transaction time?
- Which investigation actions were completed?
- What evidence supports and contradicts concern?
- Which policy sections apply?
- What outcome is recommended, by which role, and why?
- What remains unknown and what follow-up is required?

Do not include credentials, raw personal identifiers, or invented facts.

# Fraud Escalation Policy

> **FICTIONAL PORTFOLIO DOCUMENT:** This original policy describes Harborline
> Pay, a fictional digital-payments company. It is educational material, not a
> real organization's policy, legal advice, or authority to act on an account.

**Source ID:** `fraud-escalation-policy-v1`  
**Version:** `1.0.0`

## ESC-01 - Purpose and boundaries

This policy explains how a fictional investigation moves to a role with greater
review responsibility. Escalation communicates evidence and uncertainty; it is
not a declaration that fraud occurred. Any real payment restriction, customer
contact, regulatory report, or law-enforcement referral is outside this project.

## ESC-02 - Escalation levels

- **Routine review:** the Fraud Operations Analyst can complete the documented
  checks and record an outcome.
- **Priority review:** a Senior Fraud Reviewer should assess stronger evidence,
  material uncertainty, repeated activity, or potentially significant impact.
- **Critical review:** the Fraud Duty Manager should coordinate a fictional
  response when credible evidence suggests continuing or coordinated harm.
- **Model or data review:** the Model Risk Steward should investigate suspected
  feature, scoring, drift, or data-quality problems separately from case guilt.

Levels express review urgency within this fictional exercise. They do not copy
or replace a real organization's service levels.

## ESC-03 - Escalation conditions

Priority review is appropriate when the evidence includes a meaningful
combination of new-relationship activity, unusual amount, repeated destination
activity, or several alerts that cannot be explained during routine review.

Critical review is appropriate when recorded evidence suggests that suspicious
activity may be continuing, coordinated across several origins, or capable of
causing material fictional harm. Model or data review is appropriate when
missing features, inconsistent scores, impossible values, or a systematic alert
pattern could affect multiple cases.

An analyst must state which evidence supports the chosen condition. A high score
or high amount alone is insufficient.

## ESC-04 - Decision and hand-off process

The current owner records the escalation level, reason, supporting observations,
alternative explanations, unresolved questions, and requested next decision.
The receiving role acknowledges ownership in the fictional case record and may
accept, downgrade, redirect, or close the escalation with a written reason.

If case evidence and model-quality concerns coexist, they are handed off as two
linked questions: the Senior Fraud Reviewer assesses the case, while the Model
Risk Steward assesses the system concern.

## ESC-05 - Information required for escalation

Every escalation should include:

- a non-sensitive transaction reference and event time;
- the alert reason, probability, and risk classification;
- relevant past-only behavioural indicators;
- investigation actions already completed;
- supporting and contradictory evidence;
- the proposed escalation level and responsible role;
- unanswered questions and a requested next action; and
- citations to the relevant fictional policy sections.

Credentials, raw personal identifiers, unsupported allegations, and future
information must not be included.

## ESC-06 - Closure and feedback

The receiving role records the final fictional decision, evidence relied upon,
remaining uncertainty, and any follow-up. If an alert exposed unclear guidance,
the team records a proposed policy improvement without silently changing past
case notes. If it exposed a model or data concern, that concern is documented
for the Model Risk Steward rather than presented as proof of fraud.

# Fraud Investigation Policy

> **FICTIONAL PORTFOLIO DOCUMENT:** This original policy describes Harborline
> Pay, a fictional digital-payments company. It is educational material, not a
> real organization's policy, legal advice, or authority to act on an account.

**Source ID:** `fraud-investigation-policy-v1`  
**Version:** `1.0.0`

## INV-01 - Purpose and boundaries

This policy gives Harborline Pay's fictional Fraud Operations team a consistent
way to review transaction alerts. Its purpose is to organize evidence and make
proportionate recommendations. A model score starts an investigation; it does
not prove that a person committed fraud.

The policy applies only to the synthetic scenarios in this portfolio. Analysts
must not infer facts that are absent from the available transaction history.
Actions affecting a real customer, account, or payment would require separate,
lawful production controls that are outside this project.

## INV-02 - Investigation principles

Analysts should:

1. separate observed facts from assumptions;
2. use only information available at the time of the reviewed transaction;
3. consider both suspicious and ordinary explanations;
4. record why each check was relevant;
5. escalate uncertainty when potential harm is material; and
6. avoid treating transaction amount or model output as proof on its own.

The investigation should be reproducible: another fictional analyst should be
able to read the case note and understand the evidence, decision, and remaining
uncertainty.

## INV-03 - Alert intake and triage

The Fraud Operations Analyst first confirms that the alert contains a valid
transaction reference, event time, transaction type, amount, model probability,
risk classification, and the behavioural indicators supplied to the scoring
service. Missing or contradictory information is recorded before analysis.

Triage considers the possible impact, the strength of the evidence, repeated or
linked activity, and whether immediate review is warranted. Priority is based
on the whole context rather than a single fixed amount or score.

## INV-04 - Investigation actions

For a supported alert, the analyst should:

1. confirm the transaction context and the time represented by each feature;
2. compare the amount with the origin's strictly earlier activity;
3. review strictly earlier destination activity, including recent counts,
   amounts, transaction types, and distinct origins;
4. check whether the origin-destination relationship appears new in the
   available history;
5. look for a burst, an unusual amount, or several mutually supporting signals;
6. record ordinary explanations and data-quality limitations; and
7. choose a documented outcome: close as not suspicious, continue monitoring,
   request senior review, or escalate.

PaySim balance fields, fraud labels, raw customer identifiers, and future events
must never be treated as model evidence. Raw identifiers may be used only to
join the fictional case context and must not be copied into portfolio reports.

## INV-05 - Escalation indicators

Escalation is appropriate when one or more of the following is supported by
recorded evidence:

- several behavioural indicators reinforce a credible fraud concern;
- a rapid pattern suggests continuing potential loss;
- multiple origins show unusual activity involving the same destination;
- the possible impact is material and the evidence remains uncertain;
- linked alerts suggest a coordinated pattern; or
- a data or model problem could affect more than one investigation.

The analyst uses the Fraud Escalation Policy to choose a level and recipient.
No single indicator automatically requires an adverse action.

## INV-06 - Evidence and case outcomes

A case note records the alert reason, event time, observed indicators, checks
performed, alternative explanations, relevant source sections, decision,
responsible role, and follow-up. It must distinguish model output from human
judgement.

Available fictional outcomes are:

- **Close:** evidence does not support continued investigation.
- **Monitor:** evidence is weak but a defined follow-up is reasonable.
- **Senior review:** evidence or uncertainty requires a second assessment.
- **Escalate:** documented conditions in the escalation policy are met.

## INV-07 - Fictional responsibilities

- The **Fraud Operations Analyst** performs triage, gathers evidence, records
  alternatives, and recommends an outcome.
- The **Senior Fraud Reviewer** challenges the evidence, handles ambiguous or
  higher-impact cases, and confirms or changes the recommendation.
- The **Fraud Duty Manager** receives critical fictional escalations and
  coordinates the documented response.
- The **Model Risk Steward** reviews suspected scoring, feature, or data-quality
  problems. This role does not decide whether an individual transaction is fraud.

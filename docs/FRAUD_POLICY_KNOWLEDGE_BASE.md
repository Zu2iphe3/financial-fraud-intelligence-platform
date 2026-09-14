# Fictional fraud-policy knowledge base

## Purpose

PH9-01 provides the original fictional document corpus for the fraud analyst
assistant planned in PH9-02. The main deliverable is readable, internally
consistent guidance that a reviewer can inspect without running an AI model.

The corpus describes Harborline Pay, a fictional digital-payments company. It
does not reproduce an employer, client, regulator, or real financial
institution's documents, processes, prompts, escalation paths, or business
rules.

## Document set

| Source ID | Document | Purpose |
|---|---|---|
| `fraud-investigation-policy-v1` | Fraud Investigation Policy | Defines triage, investigation actions, evidence, outcomes, and responsibilities. |
| `fraud-escalation-policy-v1` | Fraud Escalation Policy | Defines fictional escalation levels, conditions, hand-offs, and closure. |
| `analyst-investigation-guide-v1` | Fraud Analyst Investigation Guide | Applies the policies to supported fictional alert scenarios. |
| `risk-alert-interpretation-guide-v1` | Risk and Alert Interpretation Guide | Explains scores, thresholds, fraud metrics, behavioural features, and limitations. |

The source files live under `knowledge_base/`. The accompanying
`knowledge_base/manifest.json` records stable source IDs, section IDs, versions,
paths, policy domains, and supported scenarios.

## Coverage of Issue #13

| Required domain | Primary source | Evidence |
|---|---|---|
| Fraud investigation | `fraud-investigation-policy-v1` | Triage, past-only checks, evidence, escalation indicators, and case outcomes. |
| Escalation | `fraud-escalation-policy-v1` | Review levels, escalation conditions, hand-off information, and responsibilities. |
| Analyst guidance | `analyst-investigation-guide-v1` | Four scenarios with actions, escalation conditions, and a responsible fictional role. |
| Risk interpretation | `risk-alert-interpretation-guide-v1` | Probability, thresholds, precision, recall, alert workload, human review, and PaySim limitations. |

## Supported fictional scenarios

The first corpus version supports:

1. an unusual transfer to a new destination;
2. rapid activity involving one destination;
3. an amount outside recent behaviour; and
4. high model risk with conflicting evidence.

Each scenario identifies investigation actions, escalation conditions, and the
responsible fictional role. The guidance deliberately requires multiple pieces
of evidence and records alternative explanations. A model score, amount, or
relationship-newness indicator is never treated as proof of fraud on its own.

## Source-aware structure

Document and section IDs are stable rather than derived from absolute file
paths or chunk positions. This means PH9-02 can preserve readable citations such
as `analyst-investigation-guide-v1 / GUIDE-03` when it later performs chunking
and retrieval on another machine.

Paths in the manifest are relative to the `knowledge_base/` directory. Section
headings contain their IDs so both a person and the future ingestion process can
locate the cited guidance.

## Lightweight validation

`src/rag/corpus_validation.py` enforces only the document contract needed by
Issue #13 and PH9-02. It checks:

- stable, unique document and section identifiers;
- repository-relative document paths and declared files;
- the visible fictional-use notice and source ID;
- coverage of the four required policy domains;
- actions, escalation conditions, and ownership for supported scenarios; and
- unresolved placeholder content.

Run the focused checks on Windows with:

```powershell
python -m pytest tests/test_rag_corpus.py --basetemp=.pytest_tmp/ph9-01-focused
```

The validator does not parse embeddings, build chunks, run a vector database,
or judge the quality of an AI answer. Those capabilities belong to PH9-02.

## Originality and portfolio-safety review

The PH9-01 review confirms that:

- all policy prose was written specifically for this portfolio project;
- Harborline Pay and all named roles and processes are fictional;
- no employer, client, or real organization document was used;
- no customer record, personal identifier, credential, or secret is present;
- no production threshold, legal requirement, or real escalation rule is
  represented as authoritative;
- each file visibly states its fictional status; and
- no unresolved placeholder content remains.

Automated checks can catch structural mistakes and known placeholder markers,
but they cannot prove originality. The human content and Git diff review remains
an explicit part of accepting this corpus.

## PH9-02 boundary

PH9-02 may read the manifest and documents, create citation-aware chunks,
generate local embeddings, store them in the planned local vector store,
retrieve relevant sections, and use Ollama to produce grounded answers with
sources. PH9-01 does not implement any of those operations.

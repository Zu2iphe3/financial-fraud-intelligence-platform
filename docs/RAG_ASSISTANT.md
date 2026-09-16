# Local grounded RAG analyst assistant

## Purpose and boundary

PH9-02 makes the fictional PH9-01 fraud-policy corpus searchable and uses a
local Ollama model to explain retrieved guidance. It is a policy assistant, not
a fraud prediction service. A policy answer must never be presented as a model
score, and a model score must never be presented as policy evidence.

All inference, embeddings, and vector storage run locally. No hosted embedding
or language-model service is used. The documents remain explicitly fictional
portfolio material rather than operational or legal guidance.

## How the pipeline works

```text
PH9-01 manifest and Markdown documents
    -> validate declared sources and sections
    -> split within section boundaries
    -> create local sentence-transformer embeddings
    -> store deterministic chunks in local Chroma

Question
    -> semantic similarity search
    -> apply the documented relevance threshold
    -> decline immediately when evidence is insufficient
    -> otherwise pass escaped, untrusted passages to local Ollama
    -> return generated text plus application-verified sources
```

## Why Chroma was selected

Chroma fits this small, local, rebuildable corpus without adding a PostgreSQL
extension or coupling policy search to the transaction database. Its generated
state lives under `chroma_db/`, which Git already ignores. PostgreSQL continues
to serve transaction and analytics data; Chroma serves only local RAG evidence.

pgvector remains a reasonable future option for a shared or larger deployment,
but it is unnecessary for Issue #14.

## Configuration

Committed defaults live in `config/rag/policy_assistant.json`:

| Setting | Default | Purpose |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Creates local semantic vectors. |
| Ollama model | `llama3.2:3b` | Generates the grounded response locally. |
| Vector store | `chroma_db/` | Stores the generated local index. |
| Collection | `ffip_fraud_policy_v1` | Separates this corpus from other indexes. |
| Chunk size / overlap | 900 / 120 characters | Splits longer sections while retaining nearby context. |
| Retrieval top-k | 4 | Limits the context and review surface. |
| Minimum relevance | 0.35 | Rejects evidence below the reviewed boundary. |

Local overrides supported through `.env` or the current shell are:

```text
OLLAMA_BASE_URL
OLLAMA_CHAT_MODEL
RAG_EMBEDDING_MODEL
RAG_CHROMA_PATH
```

Only loopback Ollama URLs (`localhost`, `127.0.0.1`, or `::1`) are accepted.
The `.env` file is ignored and must not contain hosted API credentials.

## Fresh-clone setup on Windows

Create and activate the repository virtual environment, then install the
declared dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy the example configuration if `.env` does not already exist:

```powershell
Copy-Item .env.example .env
```

Install and launch Ollama, then download the configured local chat model once:

```powershell
ollama --version
ollama pull llama3.2:3b
ollama list
```

Ollama normally starts its local service with the Windows application. If it is
not running, start Ollama before using the `ask` command.

Hugging Face downloads the embedding model the first time indexing runs. A
checkout inside a long OneDrive path may exceed Windows path limits in the
default cache. Use a shorter local cache in that case:

```powershell
$env:HF_HOME="$env:LOCALAPPDATA\ffip-hf"
$env:HF_HUB_DISABLE_TELEMETRY="1"
```

After the model has downloaded, fully local/offline validation can use:

```powershell
$env:HF_HUB_OFFLINE="1"
```

No Hugging Face token is required for this public embedding model.

## Build and inspect the index

Build or rebuild the complete local index:

```powershell
python -m src.rag.cli index
```

The reviewed corpus produces:

```text
4 documents
25 declared sections
26 deterministic chunks
```

Chunks never cross section boundaries. Each chunk stores the corpus ID, source
ID, title, version, relative path, section ID, section heading, chunk position,
fictional status, and deterministic chunk ID.

Rebuilding removes the earlier generated collection contents and writes the
current deterministic set. An unchanged corpus therefore produces the same IDs
and count without duplicates. Because Chroma is generated state, an interrupted
rebuild can be safely rerun from the committed corpus.

Inspect retrieval without calling Ollama:

```powershell
python -m src.rag.cli retrieve "When should rapid destination activity be escalated?"
```

## Relevance-threshold review

The intentionally small evaluation set is committed at
`config/rag/retrieval_evaluation.json`. It contains four covered questions and
three clearly unrelated questions. Run it with:

```powershell
python -m src.rag.cli evaluate
```

Results for the reviewed embedding model and corpus were:

| Question category | Raw top relevance score | Result at 0.35 |
|---|---:|---|
| New-destination investigation | 0.5563 | Expected section found |
| Rapid-destination escalation | 0.4648 | Expected section found |
| Meaning of fraud probability | 0.6683 | Expected section found |
| Escalation hand-off information | 0.6665 | Expected section found |
| Employee annual leave | 0.1955 | Rejected |
| Replacing a flat tyre | 0.0372 | Rejected |
| Sourdough recipe | 0.0741 | Rejected |

All 4 covered questions found an expected source and all 3 unsupported questions
were rejected. The 0.35 boundary sits between the lowest covered top score
(0.4648) and highest unsupported top score (0.1955).

This is a transparent acceptance check, not a broad retrieval benchmark. The
threshold must be reviewed again if the corpus, embedding model, or chunking
configuration changes.

## Ask grounded questions

```powershell
python -m src.rag.cli ask "What should an analyst investigate when a transfer goes to a new destination?"
```

A successful response contains:

- `status: answered`;
- the locally generated answer; and
- verified document and section sources copied from retrieved metadata.

For an unsupported question, the assistant returns
`status: insufficient_evidence`, the documented decline message, and an empty
source list. Ollama is not called in that path.

## Grounding safeguards

- Retrieved passages are labelled as untrusted document content.
- Source text is escaped before it enters the prompt boundary.
- The system prompt says document instructions cannot override grounding.
- Ollama receives only passages that passed the relevance threshold.
- Ollama does not receive passage numbers, document identifiers, or section
  identifiers and is instructed to return natural-language prose without citations.
- The application builds the authoritative source attribution from stored metadata.
- Any model-authored document/section citation is rejected, including an identifier
  that happens to match the retrieved evidence.
- No evidence means a deterministic decline before generation.
- Ollama connection failures return a controlled availability error and do not
  present raw passages as a generated answer.

These controls reduce common grounding failures but do not make a local language
model infallible. Human review remains appropriate for portfolio demonstrations.

## Tests

Fast tests use deterministic local vectors and model doubles, so GitHub Actions
does not download AI models or require Ollama:

```powershell
python -m pytest tests/test_rag_config.py tests/test_rag_corpus.py tests/test_rag_ingestion.py tests/test_rag_retrieval.py tests/test_rag_assistant.py --basetemp=.pytest_tmp/rag-unit
```

Run the real local embedding and Ollama integration checks after completing the
setup above:

```powershell
$env:RUN_RAG_MODEL_TESTS="1"
$env:RUN_OLLAMA_TESTS="1"
python -m pytest tests/integration/test_rag_pipeline_integration.py --basetemp=.pytest_tmp/rag-live
```

The integration suite uses a temporary Chroma path. The normal `chroma_db/`
index, downloaded embedding cache, Ollama models, `.env`, and test temporary
directories must never be committed.

## PH10 boundary

The CLI is a transparent local testing interface. Streamlit will later call the
same retrieval and assistant logic, but dashboard layout, sessions, analytics
integration, and user-facing presentation remain in PH10-01.

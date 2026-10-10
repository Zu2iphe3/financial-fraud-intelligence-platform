# Dataset and approved model attribution

This repository keeps the licensing and attribution of third-party data and derived
artifacts separate from the evaluation-licensed original source code written for the
portfolio project.

## PaySim1 dataset

This project uses **PaySim1**, a synthetic mobile-money transaction dataset available
from the [original Kaggle listing](https://www.kaggle.com/datasets/ealaxi/paysim1).
PaySim1 is published under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).

The raw PaySim CSV is **not redistributed** in this repository. Users must obtain it
from Kaggle and comply with the dataset terms. The source should be cited as:

> Lopez-Rojas, E. A., Elmir, A., and Axelsson, S. (2016). “PaySim: A Financial
> Mobile Money Simulator for Fraud Detection.” *28th European Modeling and Simulation
> Symposium (EMSS 2016)*, pp. 249-255.

Dataset documentation and exploratory findings in this repository distinguish official
PaySim facts from this project's own analysis.

## Derived approved scoring artifact

The following files form the versioned approved scoring bundle:

- `artifacts/models/paysim_fraud_model_v2_scoring/model.joblib`;
- `artifacts/models/paysim_fraud_model_v2_scoring/model_manifest.json`.

The estimator was trained on PaySim1 and is distributed as a derived artifact under
CC BY-SA 4.0 with the attribution above. The repository's Evaluation License does not
apply to this model bundle. CC BY-SA permits reuse, including commercial reuse, subject
to its attribution and ShareAlike requirements. The manifest records the approved model
version, selected MLflow run, feature-contract identity, package versions, and SHA-256
of the binary. It is provided for reproducible portfolio inference, not as a
production-approved financial decision system.

Local MLflow state, candidate models, training/review outputs, processed datasets,
database contents, and vector-store state are generated evidence and are not
redistributed.

## Original project source code and documentation

The repository owner's original Python, SQL, configuration, fictional policy documents,
project documentation, and portfolio screenshots are provided under the repository's
[Evaluation License](../LICENSE). That license permits local portfolio and technical
evaluation but does not grant commercial-use or redistribution rights. It does not
replace, relicense, or weaken the PaySim1 CC BY-SA terms or the derived-model notice
above.

The local Policy Assistant uses
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
which is published under Apache License 2.0, and `llama3.2:3b`, obtained separately
through Ollama and governed by the
[Meta Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
and [Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md).
Neither model is redistributed by this repository. Other libraries, tools, and container
images remain governed by their own licenses; see [Third-party notices](../THIRD_PARTY_NOTICES.md).

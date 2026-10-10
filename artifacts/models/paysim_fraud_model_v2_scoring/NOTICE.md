# Approved scoring bundle notice

This notice applies to:

- `model.joblib`;
- `model_manifest.json`.

The XGBoost estimator in this bundle was trained on the PaySim1 synthetic mobile-money
dataset after this project applied chronological splitting, past-only behavioural feature
engineering, class weighting, validation-based model selection, and a validation-selected
decision threshold. These are modifications and derived modelling work; the raw PaySim CSV
is not included.

PaySim1 is available from the
[original Kaggle listing](https://www.kaggle.com/datasets/ealaxi/paysim1) under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).
Please cite:

> Lopez-Rojas, E. A., Elmir, A., and Axelsson, S. (2016). “PaySim: A Financial
> Mobile Money Simulator for Fraud Detection.” *28th European Modeling and Simulation
> Symposium (EMSS 2016)*, pp. 249–255.

The approved model bundle is distributed as a PaySim-derived artifact under CC BY-SA 4.0.
The repository's Evaluation License does not apply to these two files. XGBoost and the
software required to deserialize and run the estimator remain under their own licenses.

The bundle is supplied for reproducible portfolio inference and is not a
production-approved fraud decision system. Its manifest identifies the selected MLflow
run, feature contract, threshold, dependency versions, and model checksum.

# PH7-02 non-selected experiment evidence

This record preserves useful experiments without placing rejected features in the
normal v2 generation or modelling path. It contains behavioural v2 validation evidence
only: the v2 experiment did not load the test partition and selected no production
threshold. Earlier v1 PH7-02 work did evaluate the test partition, as documented in
`MODEL_TRAINING.md`. A later, separate final evaluation opened the v2 test partition
only after the selected artifact and validation threshold were frozen; it did not alter
the non-selected experiment evidence preserved here.

## Selected path

The selected feature contract is
[`config/features/paysim_v2.json`](../../config/features/paysim_v2.json). It contains
20 features: the six v1 amount/type fields and 14 past-only destination-history fields.

## Relationship feature feasibility result

Two candidate features counted earlier transactions between the exact same PaySim
origin and destination. Full generation found zero train or validation rows with an
earlier matching pair. Therefore `relationship_prior_count_log1p` was always zero and
`is_new_origin_dest_relationship` was always one. Constant fields cannot help a model
rank fraud, so both were rejected.

Their definitions and finding remain in
[`config/experiments/paysim_rejected_feature_candidates_v1.json`](../../config/experiments/paysim_rejected_feature_candidates_v1.json).

## Isolation Forest result

The experiment used a label-free Isolation Forest score. Training rows were scored
prequentially using earlier days, and the detector fitted on training days 1-18 was
frozen for validation. This prevented later activity and validation rows from fitting
the anomaly detector.

| Candidate | PR-AUC | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Behavioural XGBoost | 0.485619 | 0.350284 | 0.690633 | 0.464817 | 1,942 | 469 |
| Behavioural + Isolation Forest XGBoost | 0.489558 | 0.388817 | 0.591689 | 0.469265 | 1,410 | 619 |

Although PR-AUC increased by `0.003939`, the anomaly candidate performed worse at all
reviewed daily alert capacities:

| Alerts/day | Behavioural recall | Anomaly recall | Behavioural precision | Anomaly precision |
|---:|---:|---:|---:|---:|
| 100 | 0.293536 | 0.289578 | 0.741667 | 0.731667 |
| 500 | 0.682058 | 0.675462 | 0.344667 | 0.341333 |
| 1,000 | 0.781662 | 0.777045 | 0.197500 | 0.196333 |
| 2,000 | 0.821240 | 0.812665 | 0.103750 | 0.102667 |

The score is therefore preserved as rejected experiment evidence and is absent from
the selected predictive contract.

## V1 class-weight experiment

The bounded v1 class-weight pass compared weights `1`, `5`, `10`, `25`, `35`, `50`,
`100`, `200`, and approximately `309`. Weight 100 had the highest validation PR-AUC
(`0.296950`), but results were close across weights. This showed that class weighting
alone could not overcome v1's limited amount/type feature set.

The complete reviewed inputs remain in:

- [`config/models/paysim_xgboost_search_v1.json`](../../config/models/paysim_xgboost_search_v1.json)
- [`config/models/paysim_xgboost_final_baseline_v1.json`](../../config/models/paysim_xgboost_final_baseline_v1.json)

Local detailed evidence remains under ignored `artifacts/model_reviews/` and `mlruns/`.
Those directories are deliberately outside the normal committed/source path because
they contain generated experiment state. The committed configurations, documentation,
and reproduction commands retain the experiment decisions for a fresh clone.

# Exploratory calibration and ensemble audit

These methods were evaluated after test outcomes were already available. They are exploratory, not independent confirmation. Calibration fits validation labels only. EM and BBSE use the entire unlabeled target batch and therefore require a transductive evaluation protocol. Target labels score predictions only. All methods receive the same five seeds for both kernels.

| Study | Kernel | Method | Individual accuracy, mean +/- SD | Five-model ensemble accuracy |
|---|---|---|---:|---:|
| original | fidelity | default | 87.85% +/- 0.84% | 88.14% |
| original | fidelity | validation_threshold | 85.67% +/- 1.41% | 86.06% |
| original | fidelity | em | 88.37% +/- 1.10% | 88.94% |
| original | fidelity | bbse | 88.04% +/- 1.02% | 88.62% |
| original | fidelity | platt | 85.48% +/- 1.03% | 86.22% |
| original | fidelity | platt_em | 85.80% +/- 0.96% | 86.54% |
| original | dot | default | 86.89% +/- 1.27% | 87.50% |
| original | dot | validation_threshold | 85.71% +/- 1.21% | 86.22% |
| original | dot | em | 87.63% +/- 1.56% | 88.94% |
| original | dot | bbse | 87.47% +/- 1.41% | 88.78% |
| original | dot | platt | 85.48% +/- 0.69% | 85.58% |
| original | dot | platt_em | 85.67% +/- 0.92% | 86.06% |
| longer_training | fidelity | default | 86.67% +/- 1.28% | 86.70% |
| longer_training | fidelity | validation_threshold | 86.51% +/- 1.83% | 87.66% |
| longer_training | fidelity | em | 86.73% +/- 1.15% | 87.18% |
| longer_training | fidelity | bbse | 86.79% +/- 1.32% | 87.18% |
| longer_training | fidelity | platt | 86.38% +/- 1.28% | 86.54% |
| longer_training | fidelity | platt_em | 86.47% +/- 1.33% | 86.70% |
| longer_training | dot | default | 87.08% +/- 1.04% | 87.66% |
| longer_training | dot | validation_threshold | 86.54% +/- 0.57% | 87.66% |
| longer_training | dot | em | 87.28% +/- 0.99% | 87.50% |
| longer_training | dot | bbse | 87.18% +/- 1.14% | 87.50% |
| longer_training | dot | platt | 86.09% +/- 1.03% | 86.22% |
| longer_training | dot | platt_em | 86.15% +/- 1.15% | 86.22% |
| hybrid | fidelity | default | 85.54% +/- 1.34% | 86.54% |
| hybrid | fidelity | validation_threshold | 86.31% +/- 1.60% | 88.14% |
| hybrid | fidelity | em | 85.29% +/- 1.46% | 86.06% |
| hybrid | fidelity | bbse | 85.38% +/- 1.58% | 86.54% |
| hybrid | fidelity | platt | 85.77% +/- 1.08% | 87.18% |
| hybrid | fidelity | platt_em | 85.45% +/- 1.17% | 86.54% |
| hybrid | dot | default | 85.87% +/- 1.08% | 86.70% |
| hybrid | dot | validation_threshold | 86.60% +/- 1.27% | 87.02% |
| hybrid | dot | em | 85.71% +/- 1.04% | 86.38% |
| hybrid | dot | bbse | 85.87% +/- 1.05% | 86.70% |
| hybrid | dot | platt | 86.25% +/- 0.44% | 87.02% |
| hybrid | dot | platt_em | 86.09% +/- 0.54% | 87.02% |

An ensemble is one predictor costing five forward passes; it is not five independent replications. Per-seed and ensemble results must not be interchanged.

The audit includes the original validation-threshold and averaging checks; EM adjustment; hard-prediction BBSE; positive-slope logistic (Platt) probability calibration; and Platt followed by EM. Label-shift adjustment assumes stable class-conditional distributions; an observed class-prevalence change alone does not verify that assumption. Positive-slope calibration and binary prior adjustment preserve per-model score ordering, so they do not create an AUROC improvement.

References: [BBSE, Lipton et al. (2018)](https://proceedings.mlr.press/v80/lipton18a.html); [EM prior adjustment, Saerens et al. (2002), DOI](https://doi.org/10.1162/089976602753284446). These are established methods and do not establish algorithmic novelty for this project.

Earlier diagnostics that chose a threshold using test labels, or forced predictions to match the known test class counts, are oracle calculations. Any greater-than-90% values from those calculations are excluded from valid accuracy claims.

[Saved decision parameters and complete results](artifacts/accuracy_calibration/audit.json), [summary CSV](artifacts/accuracy_calibration/summary.csv). Reproduce with `python -m research.accuracy_calibration_audit`.

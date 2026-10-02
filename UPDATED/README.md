# Updated quantum fidelity attention project

This standalone folder contains the completed native-224 PneumoniaMNIST experiment: original-byte source files, frozen configurations, feature caches, pretrained weights, all 15 neural checkpoints, the linear control, saved predictions, verification evidence, and a detailed LaTeX report.

**Entangled quantum attention: 91.60% ± 0.58% across five models; 91.67% ensemble (572/624).** Product quantum ensemble: 91.19%. Matched classical dot ensemble: **92.15%**. These are exploratory ideal-simulation results with ImageNet pretraining; they do not establish quantum advantage, new algorithmic novelty, or clinical validity.

## Read the report

Open [DETAILED_PROJECT_REPORT.pdf](DETAILED_PROJECT_REPORT.pdf), or compile [DETAILED_PROJECT_REPORT.tex](DETAILED_PROJECT_REPORT.tex) from this folder. The report explains the data, encoder, gates, fidelity, attention, training, calibration, controls, statistics, verification, earlier unsuccessful attempts, and reproduction. Its source appendix contains the implementation used for the result.

## Run from this folder

```powershell
cd UPDATED
python -m pip install -r requirements.txt
python fetch_assets.py
python verify_package.py
python tests/run_suite.py
python -m research.quantum_native_accuracy analyze
```

The frozen feature caches and checkpoints are included, so analysis does not require retraining. The analyzer recomputes all validation/test image features and verifies every quantum test overlap with PennyLane; this can take several minutes on a CPU. `environment/requirements-lock.txt` records the original full environment, not a separately validated clean installation. The broad requirements are lower bounds; matching the recorded environment is preferable for exact reproduction.

`python -m research.quantum_native_accuracy run` reuses finished runs. For a genuinely fresh training experiment, use a separate copy and new experiment namespace rather than overwriting this evidence. Never refreeze the retained study or select settings using its test scores.

## What is local versus on GitHub

The local package contains the official 214 MB native dataset. That one file is excluded from ordinary Git because it exceeds GitHub's file limit; `fetch_assets.py` downloads the fixed public archive and verifies its checksum. Its precise hash and size remain in `PACKAGE_CONTENTS.json`. The smaller reference dataset, model weights, caches, checkpoints, predictions, and report are included on GitHub.

All frozen computational source files are copied without edits. Tests and support modules for earlier branches are retained for dependency completeness. The legacy dashboard and one original small-model checkpoint are included to keep its integration tests executable; that dashboard serves the old model, not the new native classifier. Other earlier runs remain outside this package. Earlier result documents provide context, and the original presentation has not been repurposed for this new model.

Primary method references: [hybrid transfer learning](https://arxiv.org/abs/1912.08278), [data re-uploading](https://arxiv.org/abs/1907.02085), [official MedMNIST+ data](https://zenodo.org/records/10519652), [official ResNet18 preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

# Reproducibility

Run commands from the repository root. Executed environment: Windows, Python 3.14.2, CPU PyTorch 2.14.0+cpu and PennyLane 0.45.1. Exact installed distributions are recorded in `environment/installed-packages.json` and `environment/requirements-lock.txt`; each run retains its environment, source ZIP/hashes, seed, configuration, dataset hash and commit. This is an installed-environment snapshot, not a verified clean-install lock. Hardware, CPU contention and library differences can change timing and numerical reproducibility.

## Environment and official data

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements-lock.txt
python -m research.fetch_data pneumoniamnist breastmnist dermamnist bloodmnist
```

The clean-environment installation command has not been tested. For a smaller dependency installation, the project also provides `requirements.txt`; its lower-bound constraints are not an exact reproduction environment. Fetching requires internet. Respect official licenses, including DermaMNIST's noncommercial restriction. Dataset hashes and complete indices are saved in each run. No external clinical images are required.

## Tests and frozen experiments

```powershell
python tests/run_suite.py
python -m research.suite configs/central.json --output runs/central
python -m research.suite configs/additional.json --output runs/additional
python -m research.ablations configs/ablations.json --output runs/ablations
python -m research.measure --runs runs/central --output runs/measurements
python -m research.noise_probe --output runs/circuits
python -m research.noise_sweep
python verification/check_evidence.py
```

These commands reuse completed configurations and resume incomplete training. Existing `configs/*.frozen.json` records validate manifest hashes; do not freeze them again. To rerun training independently, give a new output directory and point subsequent analysis at it. Ablations intentionally reuse identical central runs. `noise_probe` reuses existing main probes but regenerates its deterministic resource/entangling summaries. Exact interrupted-training equivalence is tested on a small synthetic dataset; resume source changes are explicitly logged.

The calibration input `calibrations/props_guadalupe.json` is a retained historical April 2021 public Qiskit fake-backend snapshot. Its origin/hash and conversion are recorded in `runs/circuits/calibration_profile.json`. It is not a present-backend model. The central measurement command is a probability-distribution emulator with zero circuit calls. Real circuit inference can be exercised with the dashboard or `python verification/circuit_demo.py`; that saved demo uses one validation image and does not estimate test accuracy.

## Every paper table and figure

```powershell
python -m research.analyze --runs runs/central runs/additional --measurements runs/measurements --output artifacts
python -m research.literature
```

Analysis regenerates central tables, paired tests, measurement tables, ablation summary, baseline/shot/reliability/noise figures and TeX result macros. Figures are saved as PDF, SVG and PNG. Raw results are never synthesized from paper values. The literature command formats the manually verified registry; it does not perform a new search.

| Output | Saved evidence |
|---|---|
| central_table.tex / central_summary.csv / matched_baselines | 100 baseline runs, test_metrics.json and per-example test_predictions.npz |
| measurement_table.tex / measurement_summary.csv / shot_tradeoff | 140 measurement runs, complete validation/test predictions and allocation arrays |
| primary_hypotheses.json / paired_classification.json | Paired seed metrics and aligned sample IDs; 10,000-resample seed bootstrap; exact sign-flip, Wilcoxon, Holm; per-seed binary DeLong and McNemar |
| reliability | Disclosed seed-0 calibrated dot/fidelity test reliability bins |
| ablation_summary.csv | 140 contrast rows in runs/ablations/index.json; repeated configurations reference existing runs |
| circuit_noise / circuit_probes.csv | runs/circuits/probes.jsonl and retained per-shot outcomes on three angle pairs |
| noise_strengths / noise_strengths_per_pair.csv | 216 retained isolated-channel circuit logs and raw outcomes in runs/noise_strengths |
| mitigation_summary.csv | Measured known-channel readout inversion on initial ancilla probes; calibration-shot overhead excluded |
| result_macros.tex | Generated primary numerical results used by paper and slides |

## Paper, presentation and dashboard

The locally downloaded official Tectonic compiler has its release URL and verified archive digest in `.tools/tectonic/download.json`. A first build may download TeX packages. With Tectonic available:

```powershell
.\.tools\tectonic\tectonic.exe -o artifacts PROJECT_METHODOLOGY_REPORT.tex
.\.tools\tectonic\tectonic.exe -o artifacts RESEARCH_PRESENTATION.tex
python dashboard/app.py
```

For a workspace-local TeX package cache, first set $env:TECTONIC_CACHE_DIR = Join-Path (Get-Location) '.tools\tectonic\cache' in PowerShell. This setting follows the [official Tectonic documentation](https://tectonic-typesetting.github.io/book/latest/getting-started/first-document.html).

The dashboard binds only to http://127.0.0.1:5050. Select a saved model/seed and validation index. Analytical, probability-emulator, ideal-statevector and actual finite-shot modes are explicitly distinguished; classifier circuits support standard and destructive SWAP. Noise and mitigation are currently evaluated on a separate four-feature overlap probe, not whole-classifier noisy inference. No QPU credentials or cloud service are needed.

## Evidence preservation

The pretrained quantum branches reuse the four-qubit attention code and keep separate image-feature caches:

```powershell
python -m research.quantum_transfer_experiment run
python -m research.quantum_transfer_analyze --verify-circuits
python -m research.quantum_native_accuracy run
python -m research.quantum_native_accuracy analyze
```

Each branch requires its already-frozen manifest. The upsampled-28 branch uses `configs/pretrained_quantum_accuracy.frozen.json`; the native-224 branch uses `configs/native_quantum_accuracy.frozen.json`. On a fresh reconstruction, restore official ResNet-18 ImageNet-1K V1 weights with SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, run `python -m research.quantum_transfer` for the 28-pixel feature cache or `python -m research.quantum_native_accuracy prepare` for native features, then freeze once before model training/test prediction. Native data comes from the official MedMNIST+ file `pneumoniamnist_224.npz`, retained as `data/native_224/pneumoniamnist.npz` and checked against published MD5 `d6a3c71de1b945ea11211b03746c1fe1`. `python -m research.fetch_native_dataset` downloads/resumes this fixed public file. Do not refreeze existing experiments. The native adapter substitutes the manifest, artifact directories and data resolution while reusing the earlier frozen learning code; its source hash is included in the native freeze.

The neural heads share frozen ResNet features, training-only normalization, two training views and five seeds. The classical linear control uses the same features and validation-only regularization selection. Both quantum kernels are independently evaluated on every test attention overlap through PennyLane. The verifier also re-encodes all validation/test images from original pixels and checks full checkpoint predictions. A 224-pixel pretrained encoder applied to upsampled 28-pixel images is distinct from one applied to native-224 inputs. External ImageNet pretraining is disclosed in both branches; neither supplies an independent external confirmation cohort.

The quantum-preserving accuracy experiment has its own manifest, resumable checkpoints and matched entangled/product/dot conditions:

```powershell
python -m research.quantum_robust_experiment pilot
python -m research.quantum_robust_experiment freeze
python -m research.quantum_robust_experiment run
python -m research.quantum_robust_analyze --verify-quantum
```

On a fresh reconstruction, complete the validation pilot before freezing. Once `configs/quantum_robust_accuracy.frozen.json` exists, reuse it and run only the last two commands. Do not refreeze after inspecting test scores. The runner retains model, EMA, optimizer, scheduler and all random-generator states for exact continuation; a test verifies bit-exact resumed weights and predictions for both optimizer recipes. One shared recipe is chosen by mean class-balanced validation NLL across both pilot seeds and all three kernels. All final kernels use five seeds, ordinary argmax and an equal-probability ensemble without target-batch adaptation. Quantum attention training uses four-qubit gate simulation; the independent verifier replaces every overlap with a PennyLane compute-uncompute circuit on all 624 test images for all five entangled and all five product-state models. It records QNode calls separately from broadcast circuit settings; ideal probabilities have zero measurement shots. These exploratory experiments share a previously inspected official test cohort and do not create an independent confirmation set.

The later accuracy comparisons and calibration audit have separate commands:

```powershell
python -m research.suite configs/accuracy_followup.json --output runs/accuracy_followup
python -m research.accuracy_analyze
python -m research.hybrid_accuracy run
python -m research.hybrid_analyze --verify-inference
python -m research.accuracy_calibration_audit
```

The 40-epoch and hybrid manifests are already frozen; do not refreeze them. The hybrid runner checks its four frozen source dependencies, skips completed test conditions, and retrains an incomplete condition deterministically rather than restoring optimizer state. The hybrid analysis checks both complete splits, recomputes metrics and optionally reproduces inference from every best checkpoint. The calibration audit is post hoc and retains six fixed methods for both kernels and all three studies. Its methods accept validation labels and target probabilities; they never receive target labels or the known target class counts. Those labels enter scoring only. EM/BBSE use the unlabeled target cohort and therefore require an explicitly transductive protocol. All methods, including unsuccessful ones, are retained. Test-label-selected threshold and known-test-prior oracle diagnostics are excluded from achieved-accuracy claims.

The October 2 follow-up is reproduced independently of the original analysis:

```powershell
python -m research.bias_experiment run
python -m research.bias_analyze
python verification/check_bias_aware.py
```

The frozen manifest is `configs/bias_aware.frozen.json` (2026-10-02 09:41:55 UTC); it locks thirteen computational/configuration dependencies. Do not run the freeze action again. The prior-art gate and validation pilot precede the freeze. The run action reuses completed conditions and trains missing seeds 5-9 with validation-only checkpoint selection. It checks all frozen hashes before using the official test set. Complete outcomes/ledgers are under `runs/bias_aware/test`; statistics and vector figures are under `artifacts/bias_aware`; readable results are in `BIAS_AWARE_RESULTS.md`. Five sampling repetitions are averaged within each of ten model seeds before paired tests. The main sweep uses factorized noisy density-matrix simulation and parity sampling, while the independent verifier executes twelve full eight-wire PennyLane calls (six exact and six finite, 24,576 total finite shots). The verifier casts the saved float32 angles to float64 to match production simulator gate precision. Only final-block CLS attention is replaced by noisy estimates.

`verification/check_evidence.py` checks manifest hashes, expected seed/model counts, complete sample IDs, matched parameter counts, raw shot budgets and preservation of original checkpoints/results. It writes an evidence manifest, not fabricated experimental output. Tests produce timestamped logs. Original materials remain in the archive and old checkpoints/results remain untouched. The user-modified original PPTX is not overwritten; the new presentation uses a separate filename.

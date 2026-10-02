# Quantum circuit attention: robust accuracy experiment

This experiment retains a quantum feature circuit and compares it with product-state fidelity and matched dot attention. It uses four-qubit gate simulation for differentiable training and an independent PennyLane compute-uncompute backend for quantum inference verification. All results are exploratory because the official test set was examined in earlier experiments.

The validation pilot selected the shared **robust_adamw** recipe. Every kernel receives the same training settings and five seeds. The previous 88.94% result used an ensemble plus target-prior adjustment; this experiment uses ordinary argmax and an equal-weight ensemble, without target adaptation.

| Kernel | Single-model accuracy, mean +/- seed SD | AUROC, mean +/- seed SD | Five-model ensemble accuracy |
|---|---:|---:|---:|
| entangled | 88.56% +/- 0.99% | 0.95173 +/- 0.00307 | 89.10% |
| product | 88.75% +/- 0.58% | 0.94963 +/- 0.00520 | 89.74% |
| dot | 87.98% +/- 0.44% | 0.95068 +/- 0.00326 | 88.94% |

The entangled ensemble does not exceed 90% on all 624 test images: 89.10%. The single-model mean is 88.56%. Ensemble accuracy is one prediction set costing five model inferences, not five independent repetitions.

## Quantum algorithms and execution

- Four RY encoding gates, three CNOT entanglers, then data-dependent RZ and RY(theta/2) re-uploading per wire. Fidelity is the zero-state probability after compute-uncompute U(k)^dagger U(q). The product ablation uses four RY gates without entanglement.
- Gate-based statevector simulation trains query/key projections by backpropagation. A classical computer simulates the quantum algorithm. No physical QPU or quantum speedup is implied.
- Parameter counts are matched: 16,226 per kernel. The local patch encoder, classical value aggregation and classification head are shared.
- Complete-test PennyLane verification for both quantum maps: 2,340 batched QNode calls, 7,213,440 four-qubit overlap settings, zero measurement shots (ideal probabilities). All saved statevector and PennyLane logits agree within numerical tolerance.
- Independently executed entangled quantum-circuit ensemble accuracy: 89.10%.
- Independently executed product quantum-circuit ensemble accuracy: 89.74%.

## Validation recipe selection

| Recipe | Entangled balanced NLL | Product balanced NLL | Dot balanced NLL | Shared mean |
|---|---:|---:|---:|---:|
| robust_adamw | 0.18673 | 0.19046 | 0.20308 | 0.19343 |
| robust_sam | 0.22487 | 0.22681 | 0.23292 | 0.22820 |

These are validation losses averaged over the two pilot seeds. Lower is better. All twelve pilot conditions are retained in `runs/quantum_robust_accuracy/pilot_summary.json`; the shared mean selects the recipe. The additional planned AdamW seeds were trained on training/validation data in parallel while the pilot finished and did not enter recipe selection.

## Changes and selection

- Training-only per-image rotation up to 10 degrees, scale and translation, border padding, horizontal flip, brightness/contrast jitter and MixUp(alpha=0.2).
- Twenty-four training epochs, AdamW with weight decay 0.001, warmup and cosine learning rate, and EMA(decay=0.98). The SAM candidate uses radius 0.05 and two gradient passes.
- The best EMA checkpoint minimizes class-balanced validation NLL. One shared optimizer recipe was selected by validation scores across two pilot seeds and all three kernels, then frozen before this experiment's test inference.
- SAM, augmentation, EMA and data re-uploading are established methods. Higher accuracy alone is not proof of algorithmic novelty. A fidelity-specific contribution requires outperforming the equally treated controls.

## Paired comparisons

| Entangled minus control | Metric | Mean difference | Seed-bootstrap 95% interval | Holm p |
|---|---|---:|---|---:|
| product | accuracy | -0.19231 | [-0.51282, +0.16026] | 1.00000 |
| product | auroc | +0.00210 | [-0.00075, +0.00494] | 1.00000 |
| dot | accuracy | +0.57692 | [-0.51282, +1.31410] | 1.00000 |
| dot | auroc | +0.00105 | [-0.00322, +0.00552] | 1.00000 |

Accuracy differences are percentage points. Bootstrap intervals cover model-seed variability on this fixed cohort. With five seeds, the smallest nonzero two-sided exact sign-flip p is 0.0625. Four seed comparisons use Holm correction. Ensemble paired-example tests are reported separately and are exploratory.

## Evidence and reproduction

[Design and prior art](QUANTUM_ACCURACY_PLAN.md), [frozen manifest](configs/quantum_robust_accuracy.frozen.json), [per-seed results](artifacts/quantum_robust_accuracy/per_seed.csv), [statistics](artifacts/quantum_robust_accuracy/summary.json), [verification](artifacts/quantum_robust_accuracy/verification.json), [figure](artifacts/quantum_robust_accuracy/accuracy.pdf).

```powershell
python -m research.quantum_robust_experiment run
python -m research.quantum_robust_analyze --verify-quantum
python tests/run_suite.py
```

Do not refreeze or change the selected settings based on these test outcomes. The original paper and slide PDFs document the earlier study. This file records the later quantum-preserving accuracy experiment.

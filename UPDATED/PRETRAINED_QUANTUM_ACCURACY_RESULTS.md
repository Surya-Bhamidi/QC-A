# Pretrained image encoder with quantum fidelity attention

Frozen ImageNet-1K ResNet-18 features feed four-qubit entangled or product-fidelity attention. The input uses the official 28-pixel PneumoniaMNIST images with the weights' resize-to-256/center-crop-to-224 preprocessing. These results are exploratory because the official test cohort was examined earlier.

| Kernel | Single-model accuracy, mean +/- SD | Mean AUROC +/- SD | Five-model ensemble accuracy |
|---|---:|---:|---:|
| entangled | 86.19% +/- 1.11% | 0.94818 +/- 0.00261 | 86.70% |
| product | 86.89% +/- 1.10% | 0.94760 +/- 0.00327 | 86.70% |
| dot | 86.25% +/- 0.92% | 0.94652 +/- 0.00574 | 86.06% |
| Classical linear control | 85.90% (one deterministic classifier) | 0.94610 | Not an ensemble |

The >90% quantum-ensemble target is not achieved on all 624 official test images. All fifteen neural checkpoints and the linear control are retained. Ensembles average five validation-calibrated probability arrays with equal weights; they are one prediction set, not five repetitions.

## Quantum algorithms and evidence

- Entangled attention prepares four RY-encoded qubits, applies three CNOTs, then RZ(theta) and RY(theta/2) re-uploading per wire. Product attention prepares four independent RY-encoded qubits. Every attention overlap is a quantum-state fidelity.
- Training executes differentiable gates on four-qubit statevectors. Independent inference obtains the zero-state probability of the PennyLane compute-uncompute circuit U(k)^dagger U(q). Neither execution mode uses a physical QPU.
- The encoder, feature projection, value aggregation, classifier and optimizer are classical. The quantum circuits supply query/key attention similarities. Gradients through the quantum path and agreement with the PennyLane classifier are tested.
- Frozen encoder: 11,176,512 parameters. All three neural heads have 30,658 trainable parameters. The linear control fits 513 coefficients on the same frozen features.
- Complete-test verification: 780 batched QNode calls and 624,000 four-qubit circuit settings across ten quantum classifiers. Shots: zero, since these are ideal probabilities. Every saved classifier prediction is reproduced from original pixels, and circuit decisions match the gate-statevector decisions.
- Independently executed entangled quantum ensemble: 86.70% (541/624 correct), AUROC 0.95238.
- Independently executed product quantum ensemble: 86.70% (541/624 correct), AUROC 0.95295.

## Training, selection and attribution

- Four spatial ResNet tokens, a learned projection to 32 dimensions, two attention blocks and two heads. All neural kernels share the same 16-epoch AdamW recipe, class-weighted cross-entropy, feature MixUp(alpha=0.1), training-only flip views, warmup/cosine learning rate and EMA(decay=0.95).
- Feature normalization fits training views only. The frozen encoder stays in evaluation mode. All 4,708 training examples, 524 validation examples and 624 test examples are retained; the test set supplies scoring labels only.
- Best checkpoints minimize class-balanced validation NLL. Temperatures fit validation labels only. The linear control selects C from 0.01/0.1/1/10 by class-balanced validation NLL; every candidate score is preserved.
- The previous 88.94% reference was a small model ensemble with target-prior adjustment. The previous robust product ensemble reached 89.74%. Neither used ImageNet pretraining. The new experiment adds external pretraining and uses ordinary argmax without target adaptation; its comparison to the older models changes the model/data regime.
- Transfer learning, entangling attention and re-uploading have prior art. An accuracy increase is not proof of algorithmic novelty, quantum speedup, clinical validity or publication readiness. A quantum-specific accuracy benefit must be assessed against the equally treated dot and linear controls.
- The product ablation removes both the CNOTs and the second encoding stage. This contrast does not isolate entanglement alone. No finite-shot or noisy classifier accuracy is claimed here.
- Upsampling the 28-pixel source to 224 does not recover higher-resolution source-image information.

## Paired seed comparisons

| Entangled minus control | Metric | Mean difference | Seed-bootstrap 95% interval | Holm p |
|---|---|---:|---|---:|
| product | accuracy | -0.70513 | [-1.08974, -0.44872] | 0.25000 |
| product | auroc | +0.00059 | [-0.00189, +0.00307] | 1.00000 |
| dot | accuracy | -0.06410 | [-0.54487, +0.41667] | 1.00000 |
| dot | auroc | +0.00167 | [-0.00266, +0.00600] | 1.00000 |

Accuracy effects are percentage points. Intervals cover five-seed variability on this fixed cohort. The minimum nonzero two-sided exact sign-flip p with five seeds is 0.0625. Four seed comparisons use Holm correction. Ensemble paired-example McNemar/DeLong tests are exploratory and retained separately.

## Reproduction

[Prior-art gate](PRETRAINED_QUANTUM_NOVELTY_GATE.md). The manifest, weights, feature caches, checkpoints and saved predictions have retained hashes. `verification.json` independently checks full image features, training-only normalization, every checkpoint, metrics and circuit inference. The original compiled paper and slides describe the earlier study.

Evidence: `artifacts/pretrained_quantum_accuracy/summary.json`, `per_seed.csv`, `verification.json` and `accuracy.pdf`.

```powershell
python -m research.quantum_transfer_experiment run
python -m research.quantum_transfer_analyze --verify-circuits
python tests/run_suite.py
```

Reuse the existing frozen manifest; do not refreeze or tune settings from these test outcomes.

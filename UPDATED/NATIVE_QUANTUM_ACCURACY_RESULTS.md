# Pretrained image encoder with quantum fidelity attention

Frozen ImageNet-1K ResNet-18 features feed four-qubit entangled or product-fidelity attention. The input uses the official 224-pixel PneumoniaMNIST images with the weights' resize-to-256/center-crop-to-224 preprocessing. These results are exploratory because the official test cohort was examined earlier.

| Kernel | Single-model accuracy, mean +/- SD | Mean AUROC +/- SD | Five-model ensemble accuracy |
|---|---:|---:|---:|
| entangled | 91.60% +/- 0.58% | 0.98394 +/- 0.00394 | 91.67% |
| product | 91.22% +/- 0.29% | 0.98443 +/- 0.00142 | 91.19% |
| dot | 91.15% +/- 0.13% | 0.98336 +/- 0.00226 | 92.15% |
| Classical linear control | 91.03% (one deterministic classifier) | 0.98581 | Not an ensemble |

The >90% quantum-ensemble target is achieved on all 624 official test images. All fifteen neural checkpoints and the linear control are retained. Ensembles average five validation-calibrated probability arrays with equal weights; they are one prediction set, not five repetitions.

## Quantum algorithms and evidence

- Entangled attention prepares four RY-encoded qubits, applies three CNOTs, then RZ(theta) and RY(theta/2) re-uploading per wire. Product attention prepares four independent RY-encoded qubits. Every attention overlap is a quantum-state fidelity.
- Training executes differentiable gates on four-qubit statevectors. Independent inference obtains the zero-state probability of the PennyLane compute-uncompute circuit U(k)^dagger U(q). Neither execution mode uses a physical QPU.
- The encoder, feature projection, value aggregation, classifier and optimizer are classical. The quantum circuits supply query/key attention similarities. Gradients through the quantum path and agreement with the PennyLane classifier are tested.
- Frozen encoder: 11,176,512 parameters. All three neural heads have 30,658 trainable parameters. The linear control fits 513 coefficients on the same frozen features.
- Complete-test verification: 780 batched QNode calls and 624,000 four-qubit circuit settings across ten quantum classifiers. Shots: zero, since these are ideal probabilities. Every saved classifier prediction is reproduced from original pixels, and circuit decisions match the gate-statevector decisions.
- Independently executed entangled quantum ensemble: 91.67% (572/624 correct), AUROC 0.98585.
- Independently executed product quantum ensemble: 91.19% (569/624 correct), AUROC 0.98597.

## Training, selection and attribution

- Four spatial ResNet tokens, a learned projection to 32 dimensions, two attention blocks and two heads. All neural kernels share the same 16-epoch AdamW recipe, class-weighted cross-entropy, feature MixUp(alpha=0.1), training-only flip views, warmup/cosine learning rate and EMA(decay=0.95).
- Feature normalization fits training views only. The frozen encoder stays in evaluation mode. All 4,708 training examples, 524 validation examples and 624 test examples are retained; the test set supplies scoring labels only.
- Best checkpoints minimize class-balanced validation NLL. Temperatures fit validation labels only. The linear control selects C from 0.01/0.1/1/10 by class-balanced validation NLL; every candidate score is preserved.
- The previous 88.94% reference was a small model ensemble with target-prior adjustment. The previous robust product ensemble reached 89.74%. Neither used ImageNet pretraining. The new experiment adds external pretraining and uses ordinary argmax without target adaptation; its comparison to the older models changes the model/data regime.
- Transfer learning, entangling attention and re-uploading have prior art. An accuracy increase is not proof of algorithmic novelty, quantum speedup, clinical validity or publication readiness. A quantum-specific accuracy benefit must be assessed against the equally treated dot and linear controls.
- The product ablation removes both the CNOTs and the second encoding stage. This contrast does not isolate entanglement alone. No finite-shot or noisy classifier accuracy is claimed here.
- Native 224-pixel images are a separate resolution regime from the old 28-pixel benchmark. The official checksum, split labels and same-index image correspondence are audited; results must be labeled by resolution.

## Paired seed comparisons

| Entangled minus control | Metric | Mean difference | Seed-bootstrap 95% interval | Holm p |
|---|---|---:|---|---:|
| product | accuracy | +0.38462 | [-0.16026, +0.92949] | 1.00000 |
| product | auroc | -0.00049 | [-0.00353, +0.00192] | 1.00000 |
| dot | accuracy | +0.44872 | [+0.00000, +0.89744] | 1.00000 |
| dot | auroc | +0.00058 | [-0.00185, +0.00333] | 1.00000 |

Accuracy effects are percentage points. Intervals cover five-seed variability on this fixed cohort. The minimum nonzero two-sided exact sign-flip p with five seeds is 0.0625. Four seed comparisons use Holm correction. Ensemble paired-example McNemar/DeLong tests are exploratory and retained separately.

## Reproduction

[Prior-art gate](PRETRAINED_QUANTUM_NOVELTY_GATE.md). The manifest, weights, feature caches, checkpoints and saved predictions have retained hashes. `verification.json` independently checks full image features, training-only normalization, every checkpoint, metrics and circuit inference. The original compiled paper and slides describe the earlier study.

Evidence: `artifacts/native_quantum_accuracy/summary.json`, `per_seed.csv`, `verification.json` and `accuracy.pdf`.

```powershell
python -m research.quantum_native_accuracy run
python -m research.quantum_native_accuracy analyze
python tests/run_suite.py
```

Reuse the existing frozen manifest; do not refreeze or tune settings from these test outcomes.

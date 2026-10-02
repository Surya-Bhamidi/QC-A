# Hybrid accuracy experiment: completed results

This frozen exploratory comparison tests a convolutional image encoder followed by fidelity or matched dot attention. Every model uses the same encoder, dimensions, training recipe and complete official splits. The target was greater than 90% test accuracy. The official test split was previously inspected, so this is not an independent confirmatory experiment.

## Result

| Architecture | Test accuracy, mean +/- seed SD | Test AUROC, mean +/- seed SD | Validation accuracy, mean +/- seed SD |
|---|---:|---:|---:|
| Convolutional encoder + fidelity attention | 85.54% +/- 1.34% | 0.94963 +/- 0.00380 | 96.26% +/- 0.86% |
| Convolutional encoder + dot attention | 85.87% +/- 1.08% | 0.95082 +/- 0.00538 | 96.37% +/- 0.73% |

The fidelity hybrid's five-seed mean test accuracy is 85.54%, with seed-bootstrap 95% interval [84.36%, 86.47%]. The original smaller fidelity model scored 87.85% +/- 0.84%. This architecture change does not demonstrate the requested accuracy leap.

Fidelity minus matched dot, test_accuracy: -0.32051 percentage points; seed-bootstrap 95% interval [-2.01923, +1.12179]; exact paired sign-flip p=0.75000, Holm p across two metrics=1.00000.

Fidelity minus matched dot, test_auroc: -0.00120; seed-bootstrap 95% interval [-0.00404, +0.00192]; exact paired sign-flip p=0.50000, Holm p across two metrics=1.00000.

The large validation-to-test gap is observed in both architectures. Label proportions also differ: 74.2% class 1 in training/validation and 62.5% in test. These observations do not by themselves establish the cause of the gap or prove that all future methods must fail. Greater than 90% remains a research target, not an achieved result.

## Protocol and verification

- Five seeds per kernel, ten complete 624-image test predictions. Both kernels have 50,786 trainable parameters.
- Three convolutional layers (16/32/48 channels), GroupNorm/GELU, 4x4 pooled tokens, CLS token, two attention blocks, three heads and four query/key features per head.
- AdamW, 30 epochs, learning rate 0.0006, cosine schedule, class-weighted training cross entropy; checkpoint selection by unweighted validation NLL. Temperature fits only validation labels and does not change argmax accuracy.
- The validation pilot used seeds 0-2. The design was frozen before hybrid test inference. No hyperparameters or decision thresholds changed after hybrid test outcomes.
- Inference uses analytical product fidelity or dot attention; quantum circuit calls and measurement shots are zero. No quantum advantage or methodological novelty is claimed.
- The interrupted run was continued using the existing frozen source. This runner skips completed test conditions but retrains an incomplete condition deterministically; it does not restore partial optimizer state.
- Saved logits/probabilities, labels, sample IDs, metrics, parameter matching and frozen hashes are checked. Full inference from all ten best checkpoints was reproduced successfully.

## Reproduction

```powershell
python -m research.hybrid_accuracy run
python -m research.hybrid_analyze --verify-inference
python tests/run_suite.py
```

[Frozen design](configs/hybrid_accuracy.frozen.json), [per-seed results](artifacts/hybrid_accuracy/per_seed.csv), [paired statistics](artifacts/hybrid_accuracy/summary.json), [evidence audit](artifacts/hybrid_accuracy/verification.json).

The original paper and presentation retain their earlier experimental scope. This report records the later accuracy experiment.

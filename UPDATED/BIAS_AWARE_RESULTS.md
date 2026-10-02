# Bias-aware follow-up: executed results

**The later accuracy target is now achieved with quantum attention retained:** [native-224 results](NATIVE_QUANTUM_ACCURACY_RESULTS.md) give **91.60% +/- 0.58%** entangled single-model mean and **91.67%** ensemble accuracy on all 624 test images, independently reproduced with PennyLane circuits. The matched dot ensemble scores 92.15%, so no quantum-specific advantage is established. This change uses native higher-resolution inputs and ImageNet pretraining. The [robust-training attempt](QUANTUM_ROBUST_ACCURACY_RESULTS.md) (89.74% product ensemble) and [pretrained 28-pixel attempt](PRETRAINED_QUANTUM_ACCURACY_RESULTS.md) (86.70% quantum ensembles) are retained. These are separate classifier experiments; the bias-aware measurement outcomes below are preserved.

The novelty check preceded implementation. General joint mitigation and shot allocation already have close prior art; this is an attention-specific experimental specialization. These results do not establish algorithmic novelty, quantum advantage or publication readiness.

**Decision: the proposed joint method does not beat the strongest controls.** At the primary setting its output error is 55.1% higher than raw combined allocation and 22.0% higher than pooled calibrated uniform. Every one of the ten model seeds has the same direction for these contrasts (Holm p=0.00977). Accuracy is 86.99% versus 87.11% for raw combined. This supports a negative empirical finding, not a successful new-method claim.

Completed 2,400 conditions: 10 trained model seeds x 5 measurement repetitions x 2 noise models x 2 budgets x 12 policies, each on all 624 official PneumoniaMNIST test images. Repetitions are averaged within each seed before inference.

## Frozen primary comparison

Historical noise, 64 nominal shots per overlap. Lower conditional attention-output MSE is better. Changes compare the joint selective method against each baseline.

| Baseline | Baseline MSE | Joint selective MSE | Relative change | Paired difference 95% seed CI | Holm p |
|---|---:|---:|---:|---|---:|
| Raw uniform | 0.017208 | 0.021835 | +26.9% | [+0.004030, +0.005199] | 0.00977 |
| Raw combined | 0.014081 | 0.021835 | +55.1% | [+0.007077, +0.008391] | 0.00977 |
| Calibrated uniform | 0.026724 | 0.021835 | -18.3% | [-0.005487, -0.004267] | 0.00977 |
| Calibrated adaptive | 0.028191 | 0.021835 | -22.5% | [-0.007009, -0.005696] | 0.00977 |
| Pooled calibrated uniform | 0.017898 | 0.021835 | +22.0% | [+0.003198, +0.004661] | 0.00977 |

The proposed joint method shows corrected significant improvement against 2 of the five prespecified primary baselines. Superiority requires comparison with the strong controls, not just a weaker baseline.

Intervals are percentile bootstraps over ten model seeds. Exact two-sided sign-flip tests enumerate 1,024 sign patterns under the symmetric/exchangeable paired-null assumption; Holm correction covers the five frozen primary contrasts. These intervals condition on this fixed dataset and do not measure external-patient uncertainty.

## All methods and conditions

Accuracy is mean +/- SD across model seeds after averaging sampling repeats. AUROC and MSE are means. All secondary configurations are retained.

| Noise | Budget | Method | Output MSE | Accuracy (%) | AUROC |
|---|---:|---|---:|---:|---:|
| historical | 64 | Raw uniform | 0.017208 | 86.88 +/- 1.19 | 0.94797 |
| historical | 64 | Raw variance | 0.018319 | 87.01 +/- 1.11 | 0.94822 |
| historical | 64 | Raw sensitivity | 0.014213 | 87.11 +/- 1.14 | 0.94831 |
| historical | 64 | Raw combined | 0.014081 | 87.11 +/- 1.10 | 0.94819 |
| historical | 64 | Calibrated uniform | 0.026724 | 87.25 +/- 1.15 | 0.94782 |
| historical | 64 | Pooled calibrated uniform | 0.017898 | 87.17 +/- 1.25 | 0.94813 |
| historical | 64 | Calibrated adaptive | 0.028191 | 87.07 +/- 1.22 | 0.94789 |
| historical | 64 | Selective / uniform shots | 0.021975 | 87.06 +/- 1.10 | 0.94803 |
| historical | 64 | Selective / joint shots | 0.021835 | 86.99 +/- 1.07 | 0.94806 |
| historical | 64 | Selective / pooled calibration | 0.019550 | 86.99 +/- 1.13 | 0.94806 |
| historical | 64 | Selective / diagonal risk | 0.021857 | 86.95 +/- 1.19 | 0.94798 |
| historical | 64 | Selective / no calibration covariance | 0.022839 | 87.04 +/- 1.10 | 0.94789 |
| historical | 256 | Raw uniform | 0.006980 | 86.93 +/- 1.05 | 0.94819 |
| historical | 256 | Raw variance | 0.006989 | 87.00 +/- 1.11 | 0.94820 |
| historical | 256 | Raw sensitivity | 0.005899 | 87.04 +/- 1.18 | 0.94829 |
| historical | 256 | Raw combined | 0.005834 | 87.01 +/- 1.13 | 0.94829 |
| historical | 256 | Calibrated uniform | 0.006466 | 87.28 +/- 1.15 | 0.94835 |
| historical | 256 | Pooled calibrated uniform | 0.004522 | 87.27 +/- 1.15 | 0.94837 |
| historical | 256 | Calibrated adaptive | 0.006259 | 87.27 +/- 1.14 | 0.94829 |
| historical | 256 | Selective / uniform shots | 0.005845 | 87.20 +/- 1.14 | 0.94830 |
| historical | 256 | Selective / joint shots | 0.005630 | 87.15 +/- 1.09 | 0.94831 |
| historical | 256 | Selective / pooled calibration | 0.004837 | 87.20 +/- 1.18 | 0.94826 |
| historical | 256 | Selective / diagonal risk | 0.005246 | 87.18 +/- 1.20 | 0.94824 |
| historical | 256 | Selective / no calibration covariance | 0.005722 | 87.12 +/- 1.13 | 0.94826 |
| stress | 64 | Raw uniform | 0.042717 | 86.21 +/- 0.99 | 0.94774 |
| stress | 64 | Raw variance | 0.041159 | 86.30 +/- 1.05 | 0.94757 |
| stress | 64 | Raw sensitivity | 0.035789 | 86.43 +/- 0.98 | 0.94767 |
| stress | 64 | Raw combined | 0.034822 | 86.35 +/- 1.00 | 0.94779 |
| stress | 64 | Calibrated uniform | 0.047338 | 87.08 +/- 1.13 | 0.94730 |
| stress | 64 | Pooled calibrated uniform | 0.029773 | 87.10 +/- 1.07 | 0.94768 |
| stress | 64 | Calibrated adaptive | 0.051286 | 86.96 +/- 1.10 | 0.94737 |
| stress | 64 | Selective / uniform shots | 0.038520 | 86.55 +/- 1.10 | 0.94759 |
| stress | 64 | Selective / joint shots | 0.040042 | 86.45 +/- 1.03 | 0.94756 |
| stress | 64 | Selective / pooled calibration | 0.036246 | 86.50 +/- 0.99 | 0.94759 |
| stress | 64 | Selective / diagonal risk | 0.036691 | 86.61 +/- 1.10 | 0.94789 |
| stress | 64 | Selective / no calibration covariance | 0.039810 | 86.54 +/- 1.11 | 0.94751 |
| stress | 256 | Raw uniform | 0.031226 | 86.34 +/- 0.99 | 0.94782 |
| stress | 256 | Raw variance | 0.030550 | 86.30 +/- 1.00 | 0.94787 |
| stress | 256 | Raw sensitivity | 0.029015 | 86.35 +/- 0.98 | 0.94785 |
| stress | 256 | Raw combined | 0.028813 | 86.38 +/- 0.99 | 0.94785 |
| stress | 256 | Calibrated uniform | 0.011205 | 87.24 +/- 1.12 | 0.94812 |
| stress | 256 | Pooled calibrated uniform | 0.007754 | 87.21 +/- 1.13 | 0.94816 |
| stress | 256 | Calibrated adaptive | 0.011192 | 87.31 +/- 1.17 | 0.94827 |
| stress | 256 | Selective / uniform shots | 0.016110 | 86.80 +/- 1.09 | 0.94794 |
| stress | 256 | Selective / joint shots | 0.015743 | 86.84 +/- 1.14 | 0.94795 |
| stress | 256 | Selective / pooled calibration | 0.014423 | 86.79 +/- 1.10 | 0.94797 |
| stress | 256 | Selective / diagonal risk | 0.010735 | 86.98 +/- 1.06 | 0.94809 |
| stress | 256 | Selective / no calibration covariance | 0.015448 | 86.80 +/- 1.05 | 0.94798 |

Analytical reference for these ten seeds: accuracy 87.44% +/- 1.19%; AUROC 0.94839. This is a different seed count and measurement scope from the original five-seed, full-attention experiment.

## Interpretation and publication judgment

Selective correction beats the weaker per-image fully calibrated baselines at the primary setting, but that improvement disappears against strong raw or pooled-calibration controls. At 256 shots under historical noise, joint selective correction slightly improves output MSE versus raw combined (0.005630 versus 0.005834), while pooled calibrated uniform reaches 0.004522. Under stress at 256 shots, selective correction improves over raw combined (0.015743 versus 0.028813), but pooled calibrated uniform is again better (0.007754). The pooled version of the proposed method also loses to pooled calibrated uniform in all four noise/budget settings.

The full covariance surrogate does not consistently beat its diagonal ablation; the diagonal version has lower output error in three of four settings. Correct local geometry alone therefore does not establish an effective finite-shot policy. No causal attribution to one approximation is proven: pilot uncertainty, affine-response mismatch, nonlinear propagation, calibration cost and finite optimization can all matter. These results do not justify tuning on the same test set until the proposed method wins.

A methodological novelty claim is unsupported because general decision-aware mitigation and joint coefficient/allocation design have close prior art, and this specialization fails against strong baselines. A negative-results or benchmarking paper could be developed, but acceptance is not established; it would need a distinct generalizable lesson, wider tasks/circuit families, independent confirmation and a venue-specific assessment. Adding more runs of this configuration alone would not establish novelty.

## Costs, execution and limitations

- Each image has 2 heads x 17 final-CLS overlaps. Total budget is 2,176 or 8,704 shots per image, including calibration, pilots and production; respectively 1,357,824 or 5,431,296 for 624 images per condition.
- Raw policies spend the full budget on targets. Corrected policies spend 12.5% on four known references. Adaptive corrected policies also spend 272 pilot shots per image. Independent production outcomes exclude pilots. Calibrated uniform avoids this pilot cost.
- Pooled controls share all paid reference outcomes across test images under the stationary-noise assumption. They use no labels, target fidelities or true noise parameters. Their online/latency requirements differ from per-image calibration.
- Simulation applies gates and local noise channels to exact disjoint two-qubit density matrices, then samples the destructive-SWAP parity distribution. No full-register PennyLane calls or physical QPU runs are counted in the 2,400-condition sweep. Independent full-register checks are recorded separately.
- Only final-block CLS attention is noisy. The earlier backbone is analytical. This is not end-to-end noisy quantum-transformer inference. Product feature states and independent local noise enable efficient classical simulation.
- Historical noise uses a retained April 2021 device calibration proxy, not a current device. The second noise model is a synthetic stress mixture. Affine calibration is an approximation and can mismatch target circuits.
- The primary endpoint is attention-output reconstruction, not clinical accuracy. Better reconstruction need not improve classification.
- The official test set was examined in the earlier study. This disclosed frozen follow-up is not fresh external confirmation. No settings changed after its test freeze.

## Evidence and reproduction

- [Novelty gate and primary sources](BIAS_AWARE_NOVELTY_GATE.md)
- [Mathematical method and assumptions](BIAS_AWARE_METHOD.md)
- [Frozen manifest](configs/bias_aware.frozen.json)
- [Per-condition results](artifacts/bias_aware/conditions.csv)
- [Per-seed results](artifacts/bias_aware/per_seed.csv)
- [Primary paired tests](artifacts/bias_aware/primary_tests.json)
- [All secondary contrasts](artifacts/bias_aware/secondary_tests.json)
- [Output-error figure](artifacts/bias_aware/output_error.pdf)
- [Complete prediction and shot-ledger audit](verification/bias_aware/evidence_manifest.json)
- [Independent full-register circuit checks](verification/bias_aware/full_register_checks.json)
- Saved predictions, measured outcomes and allocation ledgers: runs/bias_aware/test.

```powershell
python -m research.bias_experiment run
python -m research.bias_analyze
python verification/check_bias_aware.py
python tests/run_suite.py
```

The run command validates frozen source/configuration hashes and resumes completed conditions. Do not refreeze or tune on these test outcomes.

# Claim-to-evidence audit

| Claim | Evidence | Verdict / boundary |
|---|---|---|
| Original standard forward computes product fidelity classically | RESEARCH_AUDIT.md, archived src/quantum/swap_test.py | Verified; circuit construction is not circuit execution |
| New low-dimensional kernels are parameter/dimension matched | research/models.py; resources.json; tests; evidence_manifest.json | Verified; full-dimensional dot has more parameters |
| Four datasets, five seeds, complete official tests | 100 saved metric/prediction pairs; checked sample IDs | Verified; no patient-ID audit or clinical validity implied |
| Adaptive measurement study uses full validation/test | 140 result directories and 1,400 test-allocation files | Verified; classical probability emulation, modeled shots |
| Actual circuit inference works | circuit_demo.json; native_quantum_accuracy/verification.json; numerical circuit tests | Original single-image demo and later full-test ideal quantum branches verified; no complete-test noisy or physical-hardware accuracy claim |
| Hardware-calibration-derived noise was evaluated | saved Guadalupe JSON/hash; circuits/calibration_profile.json | Historical homogeneous proxy only; not current backend/native compilation |
| Noise type/strength was varied | noise_strengths/protocol.json, 216 per-circuit logs/outcomes | Synthetic pairs only; no classifier robustness claim |
| Combined allocation is better than uniform | primary_hypotheses.json | Unsupported; primary output error is numerically worse |
| Product fidelity is generally better than matched dot | central_summary.csv, paired_classification.json | Unsupported; three secondary datasets favor dot in mean AUROC |
| Readout mitigation improves results | mitigation_summary.csv | Unsupported in the initial three-pair realization |
| Generic adaptive SWAP attention is new | verified bibliography and NOVELTY_SEARCH.md | Unsupported; substantial prior-art overlap |
| Constant depth, quantum advantage, clinical explanations or physical QPU results | None | Not claimed in active research documents/UI |
| Original results and user presentation are preserved | original-manifest.json, archive and evidence checker | Verified unchanged |
| All broad acceptance criteria are met | LIMITATIONS.md and implementation report | False; research judgment remains qualified |
| Convolutional-token hybrid exceeded 90% test accuracy | HYBRID_ACCURACY_RESULTS.md; hybrid_accuracy/verification.json | False: fidelity mean 85.54%; matched dot 85.87%; every full checkpoint prediction was reproduced |
| Validation/test class prevalence differs | Official pneumoniamnist.npz labels: train 3494/4708, validation 389/524, test 390/624 | Verified; prevalence change alone does not establish the cause of the accuracy gap |
| Ensembles and label-free prior correction solve the 90% target | ACCURACY_CALIBRATION_AUDIT.md and complete method arrays | False: best audited ensemble is 88.94% for both original kernels; five forward passes and transductive target predictions |
| Test-optimal thresholds or known test-class counts validate an accuracy leap | Earlier exploratory oracle calculations | Invalid as deployable accuracy claims; explicitly excluded from reported achieved results |
| Robust quantum training exceeds 90% | QUANTUM_ROBUST_ACCURACY_RESULTS.md | Not achieved: product ensemble 89.74%, entangled 89.10%, dot 88.94%; all fifteen full checkpoints reproduced |
| Trained entangling attention and full-test ideal circuit inference exist | quantum_robust_accuracy/verification.json; pretrained_quantum_accuracy/verification.json | Verified for the new accuracy branches: all twenty quantum classifiers run every test overlap through PennyLane; not a noisy or hardware accuracy claim |
| Upsampled-28 ImageNet features solve the target | PRETRAINED_QUANTUM_ACCURACY_RESULTS.md | Not achieved: both quantum ensembles 86.70%, matched dot 86.06%, linear control 85.90%; all complete original-pixel predictions verified |
| Quantum attention exceeds 90% on the full official test cohort | NATIVE_QUANTUM_ACCURACY_RESULTS.md; native_quantum_accuracy/verification.json | Verified in ideal simulation: entangled mean 91.60% +/- 0.58%, ensemble 91.67%; all quantum seeds >90%, full PennyLane inference reproduced |
| Native-224 improvement establishes quantum superiority or novelty | native_quantum_accuracy/summary.json; PRETRAINED_QUANTUM_NOVELTY_GATE.md | Unsupported: dot ensemble 92.15%, seed-contrast Holm p=1; gain changes resolution and adds external pretraining; cohort previously inspected |

Historical course documents and source snapshots are deliberately preserved as historical records. They must not be cited as evidence for the new claims. The active paper, README, notebook, dashboard and report entry points have been updated; the original modified PPTX remains untouched as requested.

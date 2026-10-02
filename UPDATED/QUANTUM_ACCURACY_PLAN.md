# Quantum-preserving accuracy improvement

The earlier 88.94% score is a five-model ensemble with post hoc, unlabeled-target prior adjustment. It uses analytical product fidelity, not actual full-test quantum circuit calls. It has the same accuracy as the matched dot ensemble and does not establish a quantum-specific advantage.

## Plausible changes and their limits

| Change | Reason to try it | Quantum requirement / claim boundary |
|---|---|---|
| SAM optimizer | Penalizes sharp parameter neighborhoods to target generalization | Classical optimizer trains the quantum circuit attention; it is established prior art |
| Affine and intensity augmentation, MixUp | Training/validation performance has not transferred to test; augmentations discourage fragile image features | Applied only during training, equally to all kernels |
| EMA and learning-rate warmup | Stabilize training and checkpoint estimates | Classical training utilities; gains cannot be credited to quantum novelty alone |
| Entangling feature map with data re-uploading | Product states miss multi-feature interactions | Four-qubit quantum feature circuit, gate-simulated during training and checked with PennyLane compute-uncompute inference |
| Equal-weight ensemble | Existing seeds disagree; averaging can improve predictions | Costs five model inferences; report separately from the five-seed single-model mean |
| Pretraining / higher-resolution original images | Could improve feature transfer more substantially | Requires a new data/protocol assessment; not part of the present bounded experiment |

The implemented experiment uses the original small patch backbone and compares entangled fidelity, product fidelity and matched dot under identical parameter counts, training data and optimization recipes. It first runs two seeds of two fixed recipes, then selects one shared recipe on class-balanced validation NLL and freezes it before new test inference. All three kernels and all five final seeds are retained. Argmax predictions use no known target class counts or test-selected cutoff. This new experiment is exploratory because the test split was examined earlier.

The quantum feature preparation is RY encoding on four qubits, a chain of three CNOTs, then RZ(theta) and RY(theta/2) re-encoding per wire. The second data-dependent rotations prevent a common final unitary from cancelling out of the overlap. Each fidelity is the probability of returning to zero in U(k)^dagger U(q)|0000>. Full-test PennyLane inference is ideal statevector simulation, not physical hardware or a finite-shot accuracy claim. A circuit simulator is classical execution of a specified quantum algorithm; it is not evidence of quantum computational advantage.

During the validation pilot, the verification scope was expanded to all five product-state models as well as all five entangled models. The learning recipes, recipe-selection rule and primary entangled comparisons are unchanged. This checks complete-test circuit execution for both quantum variants, including the original product-state feature map.

References checked before implementation: [SAM, Foret et al.](https://arxiv.org/abs/2010.01412), [official SAM implementation](https://github.com/google-research/sam), and [data re-uploading, Perez-Salinas et al.](https://arxiv.org/abs/1907.02085). These components are prior art. The question is whether this concrete combination improves accuracy against strong matched controls, not whether it constitutes a newly invented algorithm. Crossing 90% alone would not establish novelty or publication readiness.

An additional primary-source search during the validation pilot found [Higher-Order Token Interactions via Quantum Attention, Xu et al., June 2026](https://arxiv.org/abs/2606.11673). Its attention design also combines data re-uploading and entanglers, although its multi-token encoding and local readout differ from our pairwise overlap map. We therefore make no generic novelty claim for entangling attention with re-uploading, and do not borrow its expressivity or trainability guarantees for this implementation.

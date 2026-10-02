# Fresh novelty search — 2026-10-01

**Decision: the broad novelty claim does not survive.** SWAP attention, quantum medical classifiers, destructive SWAP attention and sensitivity/variance shot allocation already have prior art. The smaller candidate contribution is a controlled softmax-output allocation evaluation and reproducibility study. An absence of an exact search hit is not proof of novelty. The primary 64-shot allocation result is negative, so no efficiency breakthrough is claimed.

## October 2 extension assessment

Before implementing the proposed bias-aware extension, a fresh primary-source check found still closer work: [Scavino's Decision Kernels for Quantum Error Mitigation](https://arxiv.org/html/2607.02888v1) includes shared-noise covariance, downstream objectives and joint coefficient/allocation design. Generic joint mitigation/allocation therefore cannot be claimed as new. A fixed bias penalty also does not change optimal shots; correction strength must be an optimization variable. These corrections and the bounded experimental decision were recorded first in [BIAS_AWARE_NOVELTY_GATE.md](BIAS_AWARE_NOVELTY_GATE.md).

The attention-specific extension was implemented and frozen after a validation-only pilot. [The completed 2,400-condition follow-up](BIAS_AWARE_RESULTS.md) does not beat strong controls. It supports a disclosed negative empirical result, not established algorithmic novelty or publication readiness. The remaining sections record the original October 1 review.

## Closest overlaps

* [HQViT, Zhang et al.](https://arxiv.org/abs/2504.02730) already computes attention coefficients with a SWAP-based design. Our product RY feature map and classical value aggregation differ, but these do not establish novelty alone.
* [AQKA, Xu et al.](https://arxiv.org/abs/2605.14672v2) derives allocation from task sensitivity and Bernoulli variance, with kernel-learning experiments. The generic square-root allocation rule is prior art. Our derivative `beta*a_j*(v_j-y)` targets a softmax attention output; this is a specialization requiring evidence, not a new general optimization principle.
* [Bouakba and Belhadef](https://doi.org/10.21203/rs.3.rs-8946631/v1) already propose destructive SWAP self-attention. Their primary Research Square PDF was found; the landing-page browser request failed. No novelty is claimed for that circuit choice.
* [QiT, Patro and Agneeswaran](https://arxiv.org/abs/2609.17789), submitted September 2026, explicitly evaluates classical trigonometric, quantum-inspired transformer operations. This reinforces the need to identify analytical fidelity as classical computation.
* [Boucher, Whittle and Mazomenos](https://arxiv.org/abs/2503.07294v2) study medical QViTs, parameter comparisons and distillation. Note the **v2 title and three authors**: search results still expose an earlier title and author list.

## Search log and coverage limits

Queries were issued through public web search and primary-page opens on this date. Domain-scoped queries are **not equivalent to exhaustive authenticated database searches**. No claim of complete recall is made.

| Index / publisher | Queries or follow-up | Outcome |
|---|---|---|
| arXiv | quantum vision transformer; HQViT; SWAP-test attention; shot attention adaptive quantum transformer; adaptive shot quantum kernel; named required papers | Primary abstracts and available HTML inspected; AQKA and QiT are material overlaps |
| IEEE Xplore | quantum attention shot; QSAN; quantum self-attention | Public index results, including shot optimization and QSAN-related records; no authenticated exhaustive search |
| ACM Digital Library | quantum attention shot; quantum self-attention | Low-relevance / incomplete public-index retrieval; no affirmative absence conclusion |
| SpringerLink | quantum adaptive shot kernel; QSANN / quanvolution primary records | Relevant primary publication records and simulation discussions found |
| ScienceDirect | quantum shot attention; Quantum Mixed-State Self-Attention Network; image-classification surveys | QMSAN primary publication page inspected; survey records identified |
| Quantum journal | Quantum Vision Transformers; adaptive measurement-frugal optimizer | Cherrat and iCANS primary publisher records inspected |
| Nature / Scientific Reports | quantum shot allocation attention; MedMNIST v2 | Official benchmark PDF and related methods; incomplete novelty coverage |
| OpenReview | quantum adaptive shot attention; quantum vision transformers | QViT review PDF found; exact adaptive-attention absence not established |
| Semantic Scholar | quantum shot attention; arXiv citation links | Poor general query results; source reference lists used for citation tracking; exhaustive citation graph remains outstanding |
| Research Square | Quantum Destructive Self-Attention | Primary preprint PDF located |

Additional terms covered through named-paper and related-reference searches: quantum fidelity attention, shot-frugal kernel classification, finite-shot quantum transformers, NISQ attention, biomedical QViT, noise-aware QViT, quantum-inspired kernels. Initial remote repository open failed; audit relies on the local repository.

## Literature table and verification

`literature/table.csv` contains full titles/authors/source identifiers and task, data, encoding, circuit, attention, execution, qubits, shots, noise, baselines, metrics, contribution, limitations and relationship fields. `NE` transparently marks extraction gaps; it must not be read as “not reported” without checking the full paper. The readable companion is `literature/TABLE.md`. Numerical comparison to published performance is not attempted because protocols differ.

Specific original errors were confirmed from primary records: arXiv [2209.06588](https://arxiv.org/abs/2209.06588) is a thermal-inertial odometry paper, and [2107.03926](https://arxiv.org/abs/2107.03926) concerns financial time-series similarity. They cannot substantiate the original quantum-attention citations. Cherrat's paper is [Quantum Vision Transformers](https://quantum-journal.org/papers/q-2024-02-22-1265/), distinct from Comajoan Cara's [quark–gluon study](https://arxiv.org/abs/2405.10284).

## Submission judgment

The later [native-224 quantum accuracy experiment](NATIVE_QUANTUM_ACCURACY_RESULTS.md) reaches 91.60% mean / 91.67% ensemble accuracy with full-test ideal PennyLane execution. It adds external ImageNet pretraining and native higher-resolution inputs. Matched dot has 92.15% ensemble accuracy, and the paired seed comparisons do not establish quantum superiority. [The transfer-learning prior-art gate](PRETRAINED_QUANTUM_NOVELTY_GATE.md) precedes implementation; hybrid quantum transfer learning is already established by [Mari et al.](https://quantum-journal.org/papers/q-2020-10-09-340/). This meets the accuracy target for the new regime and preserves quantum algorithms, while algorithmic novelty and publication readiness remain unproven.

The October 2 quantum-preserving accuracy experiment is described in [QUANTUM_ACCURACY_PLAN.md](QUANTUM_ACCURACY_PLAN.md). Its SAM optimizer and data re-uploading are established components ([Foret et al.](https://arxiv.org/abs/2010.01412), [Perez-Salinas et al.](https://arxiv.org/abs/1907.02085)). An additional search during its validation pilot confirmed [Xu et al., Higher-Order Token Interactions via Quantum Attention](https://arxiv.org/abs/2606.11673), which uses entanglers and re-uploading with a different readout. The present four-qubit pairwise overlap map does not inherit that paper's theory, and the generic combination is not asserted as novel. A validation-selected training improvement or a score above 90% would be an empirical result; publication suitability still requires a specific contribution and evidence against equally treated controls.

The current defensible framing is a reproducible matched-baseline and measurement-budget study, including negative results. A QML benchmarking/reproducibility workshop or an applied quantum-computing workshop is a more plausible target than a main-track claim of a novel hardware-ready architecture. Venue selection and publishability remain unconfirmed. Before submission, finish the database/citation-graph review and extraction gaps, strengthen noise evaluation beyond a three-pair proxy, and assess whether the narrower empirical contribution meets a specific venue's standards. No physical hardware execution, clinical validation or quantum advantage is established.

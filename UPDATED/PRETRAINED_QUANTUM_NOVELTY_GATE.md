# Pretrained image features with quantum attention: prior-art gate

This gate was written before implementation of the pretrained branch. The earlier robust-training experiment is retained; its first fixed test runs remain below 90%. The next hypothesis is that stronger image representations can improve accuracy while preserving quantum fidelity attention.

## Primary sources checked

- [Mari et al., Transfer learning in hybrid classical-quantum neural networks, Quantum 4, 340 (2020)](https://quantum-journal.org/papers/q-2020-10-09-340/): pretrained classical image networks feeding quantum circuits are prior art. The [authors' implementation](https://github.com/XanaduAI/quantum-transfer-learning) is also public.
- [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): official ImageNet-1K V1 pretrained weights and their documented preprocessing are available. This is external pretraining, unlike our original small model trained from scratch.
- [Perez-Salinas et al., data re-uploading](https://arxiv.org/abs/1907.02085) and [Xu et al., quantum higher-order attention](https://arxiv.org/abs/2606.11673): neither re-uploading nor entangling quantum attention is generically new.

## Decision before coding

Proceed as a disclosed accuracy experiment, not a claim of a newly invented algorithm or guaranteed publication. Freeze the pretrained feature extractor, retain the four-qubit entangled/product attention circuits, and compare matched dot attention plus a classical linear classifier on the same cached features. All neural heads use the same five seeds and optimization recipe. Hyperparameter and checkpoint choices use validation labels only. Every final test condition is retained, including failures.

The official 28-pixel MedMNIST images will be upsampled using the weights' preprocessing; this does not add original image resolution. Pretraining weights and feature caches will have retained hashes. Training-only horizontal-flip views use training images. Feature normalization fits training features only. The ResNet remains a classical frozen encoder; the attention overlaps remain quantum gate simulation, with independent complete-test PennyLane compute-uncompute verification. No physical QPU, finite-shot classifier accuracy, clinical validation, quantum speedup or algorithmic novelty is implied.

The official test cohort has already been inspected. This is exploratory, and reaching 90% would not convert it into independent confirmation. A numerical improvement over the older model must be distinguished from a quantum-specific improvement over equally treated controls.

## Native-resolution extension gate

After the fixed 28-pixel pretrained runs also failed to show the required improvement, the [official MedMNIST+ distribution](https://zenodo.org/records/10519652) was checked. It provides `pneumoniamnist_224.npz`, 214.4 MB, official MD5 `d6a3c71de1b945ea11211b03746c1fe1`, with the same official split sizes. A separate extension will use these native 224-pixel images and preserve the pretrained encoder, quantum/classical heads, seeds, optimization and validation-selection rule. This gate precedes implementation of the native-resolution adapter.

This is a resolution change, not a new quantum algorithm. Its results must be labeled as native-224 with external ImageNet pretraining, and must not be compared to 28-pixel results as if only the quantum algorithm changed. Original dataset files, caches and failed conditions are preserved. The native file checksum, full split labels and same-index downsampled image correspondence will be audited. Every matched control receives native images, and all quantum models will again have full-test PennyLane verification. No publishability or 90% outcome is guaranteed.

## Executed outcome

The fixed native-224 extension achieves 91.60% +/- 0.58% entangled single-model mean and 91.67% ensemble accuracy. Every quantum seed exceeds 90%, and complete-test PennyLane inference reproduces all predictions. Matched dot achieves 91.15% mean and 92.15% ensemble; four paired seed contrasts have Holm p=1. The original quantum-ensemble reference was 88.94% on 28-pixel inputs without ImageNet pretraining. The accuracy target is met in a changed regime, while the gate's novelty/publication judgment remains unestablished. Full results and preserved failed branches are linked in [NATIVE_QUANTUM_ACCURACY_RESULTS.md](NATIVE_QUANTUM_ACCURACY_RESULTS.md).

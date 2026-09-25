# Quantum Vision Transformer (QViT) vs. Classical ViT
### University Sem 5 Quantum Computing & Algorithms Course Project

A hybrid quantum-classical deep learning research project comparing a standard **Classical Vision Transformer (ViT)** against a **Quantum Vision Transformer (QViT)** on medical imaging (Chest X-Ray Pneumonia / HAM10000 Skin Lesion datasets).

In this project, classical dot-product self-attention $\text{softmax}(QK^T / \sqrt{d})V$ is replaced by **Quantum Self-Attention (QSA)**, where the similarity between patch embeddings is computed using a **Quantum SWAP Test Circuit** on a statevector simulator with PennyLane and PyTorch.

---

## Architecture Overview

```
Input Medical Image (128x128)
       │
       ▼
Non-Overlapping Patches (e.g., 16 patches of 32x32 or 64 patches of 16x16)
       │
       ▼
Patch Linear Projection Layer (embed_dim = 32)
       │
       ▼
Prepend [CLS] Token + 1D Learnable Positional Embeddings
       │
       ├─────────────────────────────────────────┬─────────────────────────────────────────┐
       │                                         │                                         │
       ▼ [Classical ViT Branch]                  ▼ [Hybrid QViT Branch]                    │
Linear Projections: Q, K, V (embed_dim -> 32)    Linear Projections: Q, K (embed_dim -> 4), V (-> 32)
       │                                         │                                         │
Scaled Dot-Product Attention:                    Angle Embedding (RY Rotations) on Registers Q & K
  Attention = softmax(Q K^T / sqrt(d)) * V       │                                         │
       │                                         Quantum SWAP Test Circuit (9 Wires):      │
       │                                           - 1 Ancilla Qubit                       │
       │                                           - 4 Query Qubits                        │
       │                                           - 4 Key Qubits                          │
       │                                           - Hadamard + CSWAP (Fredkin) + Hadamard │
       │                                           - <Z_ancilla> = Fidelity(|q>, |k>)      │
       │                                         │                                         │
       │                                         Fidelity-Weighted Softmax Attention * V   │
       │                                         │                                         │
       └─────────────────────────────────────────┴─────────────────────────────────────────┘
                               │
                               ▼
            Transformer Encoder MLP Block + Residuals
                               │
                               ▼
                   Layer Normalization Layer
                               │
                               ▼
             Classification Head (from [CLS] Token)
                               │
                               ▼
                  Pneumonia vs. Normal (Logits)
```

---

## Key Features

1. **Classical ViT Baseline First**: Built as a fully functional benchmark model with standard multi-head scaled dot-product attention, trained and validated to establish reliable ground-truth performance.
2. **Quantum SWAP Test Self-Attention**:
   - Query ($Q$) and Key ($K$) vectors are projected into 4-dimensional registers.
   - Encoded via **Angle Embedding** ($RY$ rotations).
   - Pairwise attention weights computed via the **Quantum SWAP Test circuit**, measuring state fidelity $|\langle q | k \rangle|^2$.
   - Fully integrated with PyTorch autograd through PennyLane's `interface="torch"` for end-to-end backpropagation.
3. **Reproducible Medical Dataset**:
   - Defaults to the Kermany Pediatric Chest X-Ray benchmark (~5,856 images, subsetted to ~2,000 images at $128 \times 128$) via `medmnist` (`pneumoniamnist`).
   - Supports HAM10000 (`dermamnist`) or custom local image folders (`--data_dir`).
4. **NISQ Noise Resilience Study**:
   - Tests trained QViT performance under noisy intermediate-scale quantum conditions using PennyLane's `default.mixed` density matrix simulator.
   - Sweeps depolarizing noise rates $p \in [0.0, 0.20]$ to quantify error vulnerability.
5. **Authentic Metrics**:
   - Zero hardcoded or fabricated numbers.
   - Exact parameter counts and percentage reductions computed directly from `model.parameters()`.
   - Automated generation of training curves, blended attention heatmaps, and noise degradation curves.

---

## Mathematical Derivation of Quantum SWAP Test

### 1. Register Allocation (Total 9 Qubits)
- **Ancilla wire**: Wire `0`
- **Query register ($Q$)**: Wires `1, 2, 3, 4` ($d = 4$ qubits)
- **Key register ($K$)**: Wires `5, 6, 7, 8` ($d = 4$ qubits)

### 2. State Preparation (Angle Embedding)
Features $x_i \in \mathbb{R}$ are normalized to rotation angles $\theta_i = \frac{\pi}{2}(\tanh(x_i) + 1.0) \in [0, \pi]$.
Single-qubit rotation gates $RY(\theta_i)$ prepare the pure states:
$$|q\rangle = \bigotimes_{i=1}^d \left(\cos\frac{q_i}{2}|0\rangle + \sin\frac{q_i}{2}|1\rangle\right), \quad |k\rangle = \bigotimes_{i=1}^d \left(\cos\frac{k_i}{2}|0\rangle + \sin\frac{k_i}{2}|1\rangle\right)$$

### 3. Circuit Operations
1. **Initial state**:
   $$|\Psi_0\rangle = |0\rangle_a \otimes |q\rangle_Q \otimes |k\rangle_K$$
2. **First Hadamard on Ancilla**:
   $$|\Psi_1\rangle = \frac{1}{\sqrt{2}} |0\rangle_a |q\rangle |k\rangle + \frac{1}{\sqrt{2}} |1\rangle_a |q\rangle |k\rangle$$
3. **Controlled-SWAP (Fredkin) Gates**:
   For each feature index $i \in \{1, \dots, d\}$, apply $\text{CSWAP}$ controlled on ancilla wire 0 between $Q_i$ and $K_i$:
   $$|\Psi_2\rangle = \frac{1}{\sqrt{2}} |0\rangle_a |q\rangle |k\rangle + \frac{1}{\sqrt{2}} |1\rangle_a |k\rangle |q\rangle$$
4. **Second Hadamard on Ancilla**:
   Interferes the two branches:
   $$|\Psi_3\rangle = \frac{1}{2} |0\rangle_a (|q\rangle|k\rangle + |k\rangle|q\rangle) + \frac{1}{2} |1\rangle_a (|q\rangle|k\rangle - |k\rangle|q\rangle)$$
5. **Measurement & State Fidelity**:
   The expectation value of the Pauli-$Z$ operator on the ancilla wire is:
   $$\langle Z_a \rangle = P(|0\rangle_a) - P(|1\rangle_a) = |\langle q | k \rangle|^2 = \text{Fidelity}(|q\rangle, |k\rangle) \in [0, 1]$$
   - If $|q\rangle = |k\rangle \implies \langle Z_a \rangle = 1.0$ (Maximum Attention).
   - If $|q\rangle \perp |k\rangle \implies \langle Z_a \rangle = 0.0$ (Zero Attention).

---

## Project Structure

```
QC&A PROJECT/
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation & viva guide
├── COMPARISON_REPORT.md         # Generated academic benchmark report
├── train_classical_vit.py       # Classical ViT training script
├── train_qvit.py                # Quantum Vision Transformer training script
├── evaluate_and_analyze.py      # Comprehensive evaluation, plots & report generator
├── experiments_results.ipynb    # Interactive Jupyter notebook
├── checkpoints/                 # Saved model weights (*.pt)
│   ├── classical_vit_best.pt
│   └── qvit_best.pt
├── results/                     # Experimental metrics, plots & tables
│   ├── classical_vit_metrics.json
│   ├── qvit_metrics.json
│   ├── noise_resilience.json
│   ├── parameter_comparison.txt
│   ├── training_curves.png
│   ├── attention_heatmaps.png
│   └── noise_degradation_curve.png
└── src/
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py           # Medical image loaders (Chest X-ray / HAM10000)
    ├── models/
    │   ├── __init__.py
    │   ├── classical_vit.py     # Classical ViT baseline
    │   └── qvit.py              # Hybrid Quantum Vision Transformer
    ├── quantum/
    │   ├── __init__.py
    │   ├── swap_test.py         # Quantum SWAP test circuit with PennyLane
    │   └── noise.py             # NISQ depolarizing noise analysis
    └── utils/
        ├── __init__.py
        └── visualization.py     # Heatmaps, curves, and table formatting
```

---

## Installation & Setup

1. **Clone or Navigate to the Workspace**:
   ```bash
   cd "c:\Users\bhara\OneDrive\Desktop\QC&A PROJECT"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Quantum Circuit Unit Test**:
   ```bash
   python -m src.quantum.swap_test
   ```
   *Expected Output*:
   - Identical vector overlap: `1.00000`
   - Gradient flow verification: `torch.Size([1, 1, 4])`
   - Batched Attention Matrix: `[2, 4, 4]`

---

## Interactive Presentation Dashboard (For University Viva / Endsem Demo)

Launch the full interactive dark-mode demonstration dashboard:
```bash
python dashboard/app.py
```
Then open your browser at **`http://localhost:5050/`**

### Features in the Dashboard:
1. **Live Clinical Diagnosis Workstation**: Click any sample patient X-ray (Normal vs. Pneumonia) or upload custom images to see real-time inference, confidence meters, and the **Quantum Self-Attention Heatmap** overlaid on lung infiltrates.
2. **Interactive 9-Wire Quantum SWAP Test Circuit Explorer**: Drag sliders for Query/Key features to watch the state fidelity $|\langle\psi_q|\phi_k\rangle|^2$ compute live with step-by-step gate explanations.
3. **NISQ Depolarizing Noise Stress-Tester**: Adjust noise rate $p \in [0.0, 0.20]$ to see how quantum decoherence alters attention sharpness in real time.
4. **Academic Benchmark Metrics**: View the comparative table and high-res loss/accuracy plots.
5. **Viva Defense FAQ Drawer**: Clickable answers to top examiner presentation questions.

---

## How to Run Experiments

### 1. Train Classical ViT Baseline
```bash
python train_classical_vit.py --epochs 10 --lr 8e-4 --max_train_samples 4708 --max_val_samples 524 --max_test_samples 624
```

### 2. Train Hybrid QViT Model
```bash
python train_qvit.py --epochs 10 --lr 8e-4 --num_heads 2 --max_train_samples 4708 --max_val_samples 524 --max_test_samples 624
```

### 3. Generate Complete Evaluation Suite & Visualizations
```bash
python evaluate_and_analyze.py --num_heads 2
```
This produces:
- `results/training_curves.png`: Overlaid Loss and Accuracy curves
- `results/attention_heatmaps.png`: Quantum attention overlaid on medical lesions
- `results/noise_degradation_curve.png`: Accuracy degradation under depolarizing noise
- `results/parameter_comparison.txt`: Parameter comparison table
- `COMPARISON_REPORT.md`: Comprehensive academic report

---

## Viva Defense FAQ & Technical Explanations

### Q1: Why use Angle Embedding ($RY$ rotations) instead of Amplitude Embedding?
- **Angle Embedding**: Requires $d$ qubits for $d$ features, with circuit depth $O(1)$. It maps continuous features smoothly into single-qubit Bloch rotations, is trivially differentiable, and does not require complex state preparation circuits.
- **Amplitude Embedding**: Encodes $2^d$ features into $d$ qubits, but preparing arbitrary amplitude states requires $O(2^d)$ CNOT gates, which causes substantial decoherence on NISQ devices and introduces severe gradient vanishing (barren plateaus).

### Q2: Why use the SWAP Test rather than a Variational Quantum Circuit (VQC)?
- A parameterized VQC requires optimizing circuit rotation gates via classical optimizers, suffering from barren plateaus and trainability issues.
- The **SWAP test** is a deterministic, non-parametric quantum algorithm that computes the inner product / fidelity $|\langle \psi | \phi \rangle|^2$ in Hilbert space directly. It is the direct quantum analog of classical dot-product attention.

### Q3: How do gradients flow through the Quantum Circuit during training?
Using PennyLane's `interface="torch"`, the expectation value $\langle Z_a \rangle$ is evaluated with PyTorch-compatible backpropagation (`diff_method="backprop"` on statevector simulator or parameter-shift rule on physical quantum processors):
$$\frac{\partial \langle Z_a \rangle}{\partial \theta_i} = \frac{\langle Z_a(\theta_i + s) \rangle - \langle Z_a(\theta_i - s) \rangle}{2 \sin(s)}$$
where $s = \frac{\pi}{2}$. This allows the standard PyTorch `optimizer.step()` to update projection matrices $W_Q, W_K, W_V$ alongside classical MLP weights.

### Q4: Why is QViT slower to train on CPU than Classical ViT?
Pairwise attention over $N$ patches requires $O(N^2)$ circuit evaluations. Classical CPUs simulate quantum statevectors by multiplying $2^n \times 2^n$ matrices (for $n=9$ wires, vectors of length 512), which is computationally heavier than highly optimized BLAS matrix multiplication. On native quantum hardware, quantum state overlap is evaluated natively in $O(1)$ depth.

### Q5: What is the physical significance of the NISQ Noise Analysis?
Real quantum processors have gate error rates and decoherence. The depolarizing noise model $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}\sum_{\sigma \in \{X, Y, Z\}} \sigma \rho \sigma$ demonstrates how attention matrices lose contrast and revert to random uniform attention as error rate $p$ increases, underscoring the necessity of Quantum Error Correction (QEC) for future quantum deep learning.

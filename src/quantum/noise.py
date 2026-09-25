"""
noise.py - NISQ Noise Resilience Analysis for Quantum Vision Transformer

Simulates noisy quantum hardware using PennyLane's `default.mixed` density matrix simulator
with depolarizing noise channels applied to the ancilla and register wires during SWAP test.

Theoretical Background for Viva:
--------------------------------
In current Noisy Intermediate-Scale Quantum (NISQ) devices, physical qubits suffer from
decoherence (loss of quantum phase/energy) and gate infidelities.
The depolarizing channel model is defined as:
    E(rho) = (1 - p) * rho + (p / 3) * (X rho X + Y rho Y + Z rho Z)
where p is the error probability per gate/wire.
As p increases, the quantum state decays toward the maximally mixed state I / 2^n,
blurring the overlap / fidelity measured on the ancilla toward 0.5.
This module empirically quantifies the degradation in classification accuracy of the
trained QViT as physical qubit error rate p increases.
"""

import os
import json
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def evaluate_noisy_qvit(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    noise_prob: float,
    max_batches: int = 10,
) -> Tuple[float, float]:
    """
    Evaluates the QViT model under a specified depolarizing noise rate.

    Args:
        model: Trained QViT instance
        dataloader: Test or validation DataLoader
        criterion: Loss function (CrossEntropyLoss)
        device: CPU or CUDA device
        noise_prob: Depolarizing error rate p in [0.0, 1.0]
        max_batches: Maximum number of batches to evaluate for fast simulation

    Returns:
        (test_loss, test_accuracy)
    """
    model.eval()
    model.set_noise(noise_prob)

    total_loss = 0.0
    correct = 0
    total = 0

    batch_count = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, acc


def run_noise_study(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_levels: List[float] = [0.0, 0.02, 0.05, 0.10, 0.20],
    results_path: str = "./results/noise_resilience.json",
    max_batches: int = 10,
) -> Dict:
    """
    Sweeps through multiple depolarizing noise rates and saves the resulting accuracy degradation curve.
    """
    criterion = nn.CrossEntropyLoss()
    results = {
        "noise_levels": noise_levels,
        "accuracies": [],
        "losses": [],
    }

    print(f"\n=== Starting NISQ Noise Resilience Sweep on QViT ===")
    print(f"Testing depolarizing noise rates p in: {noise_levels}")

    for p in noise_levels:
        print(f"Running inference with depolarizing noise p = {p:.3f} ...", end=" ", flush=True)
        loss, acc = evaluate_noisy_qvit(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            noise_prob=p,
            max_batches=max_batches,
        )
        results["accuracies"].append(acc)
        results["losses"].append(loss)
        print(f"-> Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}")

    # Reset model to noiseless default
    model.set_noise(0.0)

    # Save results to JSON
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Noise resilience analysis saved to {results_path}\n")

    return results

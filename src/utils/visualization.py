"""
visualization.py - Plotting and Visualization Utilities for QViT Project

Generates:
1. Overlaid Training/Validation Loss and Accuracy Curves (Classical ViT vs QViT)
2. Attention Heatmaps overlaid on Medical Images with alpha blending
3. NISQ Noise Resilience Degradation Plots
4. Formatted Parameter Comparison Tables
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F


def set_plot_style():
    """Sets a clean, publication-quality aesthetic."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_training_curves(
    classical_metrics: Dict,
    qvit_metrics: Dict,
    save_path: str = "./results/training_curves.png",
):
    """
    Plots overlaid training and validation loss and accuracy curves.
    """
    set_plot_style()
    epochs_c = range(1, len(classical_metrics["train_loss"]) + 1)
    epochs_q = range(1, len(qvit_metrics["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Loss Curves
    axes[0].plot(epochs_c, classical_metrics["train_loss"], "o--", color="#1f77b4", label="Classical ViT (Train)", alpha=0.7)
    axes[0].plot(epochs_c, classical_metrics["val_loss"], "s-", color="#0d47a1", label="Classical ViT (Val)", linewidth=2)
    axes[0].plot(epochs_q, qvit_metrics["train_loss"], "o--", color="#2ca02c", label="QViT Hybrid (Train)", alpha=0.7)
    axes[0].plot(epochs_q, qvit_metrics["val_loss"], "s-", color="#1b5e20", label="QViT Hybrid (Val)", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none")

    # 2. Accuracy Curves
    c_train_acc = [a * 100 for a in classical_metrics["train_acc"]]
    c_val_acc = [a * 100 for a in classical_metrics["val_acc"]]
    q_train_acc = [a * 100 for a in qvit_metrics["train_acc"]]
    q_val_acc = [a * 100 for a in qvit_metrics["val_acc"]]

    axes[1].plot(epochs_c, c_train_acc, "o--", color="#1f77b4", label="Classical ViT (Train)", alpha=0.7)
    axes[1].plot(epochs_c, c_val_acc, "s-", color="#0d47a1", label="Classical ViT (Val)", linewidth=2)
    axes[1].plot(epochs_q, q_train_acc, "o--", color="#2ca02c", label="QViT Hybrid (Train)", alpha=0.7)
    axes[1].plot(epochs_q, q_val_acc, "s-", color="#1b5e20", label="QViT Hybrid (Val)", linewidth=2)
    axes[1].set_title("Classification Accuracy (%) vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_attention_heatmaps(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    num_samples: int = 4,
    save_path: str = "./results/attention_heatmaps.png",
):
    """
    Visualizes learned quantum attention matrix overlaid on sample medical images.
    Extracts attention from [CLS] token to spatial patches, upsamples to image size,
    and blends with matplotlib alpha transparency.
    """
    set_plot_style()
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples), dpi=300)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    # Class names for PneumoniaMNIST
    label_map = {0: "Normal", 1: "Pneumonia"}

    with torch.no_grad():
        for idx in range(num_samples):
            img_tensor, label = dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)

            # Forward pass returning attention weights
            logits, attns = model(input_tensor, return_attn=True)
            pred_idx = logits.argmax(dim=-1).item()
            gt_label = label_map.get(int(label), f"Class {label}")
            pred_label = label_map.get(pred_idx, f"Class {pred_idx}")

            # Prepare image for display: unnormalize from [-1, 1] to [0, 1]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 0.5) + 0.5
            img_np = np.clip(img_np, 0.0, 1.0)

            # Extract attention map from last block
            # Shape of attns[-1]: (1, N_tokens, N_tokens) where N_tokens = 1 + N_patches
            last_attn = attns[-1][0].cpu()  # (N_tokens, N_tokens)
            # Attention from [CLS] token (index 0) to all spatial patches (indices 1:)
            cls_attn = last_attn[0, 1:]  # (N_patches,)

            # Calculate patch grid dimension
            n_patches = cls_attn.shape[0]
            grid_dim = int(np.sqrt(n_patches))
            attn_grid = cls_attn.view(1, 1, grid_dim, grid_dim)

            # Interpolate to original image size (128x128)
            attn_upsampled = F.interpolate(
                attn_grid,
                size=(img_tensor.shape[1], img_tensor.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()

            # Normalize attention to [0, 1] for visualization
            attn_norm = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

            # Col 1: Original Image
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Sample #{idx+1} | True: {gt_label}\nPred: {pred_label}", fontsize=10, fontweight="bold")
            axes[idx, 0].axis("off")

            # Col 2: Quantum Attention Heatmap
            im_heat = axes[idx, 1].imshow(attn_norm, cmap="magma")
            axes[idx, 1].set_title(f"Quantum SWAP-Test Attention\n(Grid: {grid_dim}x{grid_dim})", fontsize=10, fontweight="bold")
            axes[idx, 1].axis("off")
            plt.colorbar(im_heat, ax=axes[idx, 1], fraction=0.046, pad=0.04)

            # Col 3: Blended Overlay
            axes[idx, 2].imshow(img_np)
            axes[idx, 2].imshow(attn_norm, cmap="magma", alpha=0.55)
            axes[idx, 2].set_title("Attention Blended on Lesion Grid", fontsize=10, fontweight="bold")
            axes[idx, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved quantum attention heatmaps to {save_path}")


def plot_noise_resilience(
    noise_data: Dict,
    save_path: str = "./results/noise_degradation_curve.png",
):
    """
    Plots test accuracy degradation under increasing NISQ depolarizing noise levels.
    """
    set_plot_style()
    p_values = [p * 100 for p in noise_data["noise_levels"]]
    accuracies = [a * 100 for a in noise_data["accuracies"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(p_values, accuracies, "o-", color="#d32f2f", linewidth=2.5, markersize=8, label="QViT Test Accuracy")
    ax.axhline(accuracies[0], color="navy", linestyle="--", alpha=0.6, label=f"Noiseless Baseline ({accuracies[0]:.1f}%)")

    # Annotate points
    for p, acc in zip(p_values, accuracies):
        ax.annotate(
            f"{acc:.1f}%",
            (p, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title("NISQ Noise Resilience Analysis (Depolarizing Channel)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Depolarizing Noise Rate p (%)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved NISQ noise degradation curve to {save_path}")


def format_parameter_table(classical_model: torch.nn.Module, qvit_model: torch.nn.Module) -> str:
    """
    Computes parameter counts directly from actual models and formats comparison table.
    """
    c_tot, c_train = classical_model.count_parameters()
    q_tot, q_train = qvit_model.count_parameters()
    diff = c_train - q_train
    pct_reduction = (diff / c_train) * 100

    table = []
    table.append("==========================================================================")
    table.append("      AUTHENTIC PARAMETER COUNT COMPARISON (COMPUTED FROM MODEL DEF)      ")
    table.append("==========================================================================")
    table.append(f"{'Metric':<35} | {'Classical ViT':<16} | {'QViT Hybrid':<16}")
    table.append("--------------------------------------------------------------------------")
    table.append(f"{'Total Parameters':<35} | {c_tot:<16,} | {q_tot:<16,}")
    table.append(f"{'Trainable Parameters':<35} | {c_train:<16,} | {q_train:<16,}")
    table.append("--------------------------------------------------------------------------")
    table.append(f"Trainable Parameter Reduction: {diff:,} parameters ({pct_reduction:.2f}%)")
    table.append("==========================================================================")
    table_str = "\n".join(table)
    return table_str

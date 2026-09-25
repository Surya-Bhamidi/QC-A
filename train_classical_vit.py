"""
train_classical_vit.py - Training Pipeline for Classical Vision Transformer

Trains the classical ViT baseline on the medical imaging dataset (e.g. Chest X-Ray Pneumonia),
evaluating after each epoch, saving the best checkpoint and logging metrics to JSON.
"""

import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import load_medical_dataset, compute_class_weights
from src.models.classical_vit import ClassicalViT


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    eval_loss = total_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc


def main():
    parser = argparse.ArgumentParser(description="Train Classical ViT Baseline")
    parser.add_argument("--dataset", type=str, default="pneumonia", choices=["pneumonia", "derma", "custom"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=1600)
    parser.add_argument("--max_val_samples", type=int, default=400)
    parser.add_argument("--max_test_samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Enable inverse class frequency weighting")
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Classical ViT on device: {device} ===")

    # Load dataset
    print(f"Loading dataset: {args.dataset} (max train: {args.max_train_samples}, max val: {args.max_val_samples})...")
    train_loader, val_loader, test_loader, in_channels, num_classes = load_medical_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    # Initialize model
    model = ClassicalViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"Model Summary: Classical ViT | Total Params: {total_params:,} | Trainable Params: {trainable_params:,}")

    if args.use_class_weights:
        weights = compute_class_weights(train_loader, num_classes=num_classes).to(device)
        print(f"Class weighting enabled: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        "model_name": "ClassicalViT",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": args.epochs,
        "training_time_sec": 0.0,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
    }

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "classical_vit_best.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
            print(f"  -> Saved new best checkpoint to {best_checkpoint_path}")

    total_training_time = time.time() - start_time
    history["training_time_sec"] = total_training_time

    # Final Test evaluation with best checkpoint
    print(f"\nLoading best checkpoint for final evaluation on test set...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    print(f"Final Test Evaluation: Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")
    print(f"Total Classical ViT Training Time: {total_training_time:.2f}s")

    # Save metrics JSON
    metrics_path = os.path.join(args.results_dir, "classical_vit_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved classical metrics to {metrics_path}")


if __name__ == "__main__":
    main()

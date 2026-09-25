"""
dataset.py - Medical Imaging Dataset Loader for QViT and Classical ViT

Supports:
1. Chest X-Ray Pneumonia (via MedMNIST PneumoniaMNIST Kermany et al. benchmark)
2. HAM10000 Skin Lesion (via MedMNIST DermaMNIST)
3. Local ImageFolder datasets (custom Kaggle Chest X-ray / HAM10000 folders)

All images are resized to 128x128 with configurable batch size and sample limits
for rapid, reproducible training on CPU/laptops.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import medmnist
from medmnist import INFO


def get_transforms(image_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns training and validation transforms.
    Includes gentle data augmentation for medical images (rotation, translation, horizontal flip).
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return train_transform, eval_transform


def compute_class_weights(dataloader: DataLoader, num_classes: int = 2) -> torch.Tensor:
    """
    Computes inverse class frequency weights from a dataloader to balance loss against class imbalance.
    """
    counts = torch.zeros(num_classes)
    for _, targets in dataloader:
        for t in targets:
            counts[t] += 1
    total = counts.sum()
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights


class SqueezeTargetDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset to ensure the target is a 1D scalar LongTensor,
    compatible with nn.CrossEntropyLoss.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if isinstance(target, torch.Tensor):
            target = target.squeeze().long()
        elif hasattr(target, "__len__"):
            target = torch.tensor(target[0], dtype=torch.long)
        else:
            target = torch.tensor(target, dtype=torch.long)
        return img, target


def load_medical_dataset(
    dataset_name: str = "pneumonia",
    data_dir: str = "./data",
    image_size: int = 128,
    batch_size: int = 32,
    max_train_samples: Optional[int] = 2000,
    max_val_samples: Optional[int] = 400,
    max_test_samples: Optional[int] = 400,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Loads train, validation, and test dataloaders for the chosen medical imaging dataset.

    Args:
        dataset_name: 'pneumonia' (Chest X-ray Kermany et al.), 'derma' (HAM10000), or 'custom'
        data_dir: Directory where dataset is stored or downloaded
        image_size: Resized image dimension (default 128x128)
        batch_size: Mini-batch size for DataLoader
        max_train_samples: Maximum training images to load (caps size for fast training)
        max_val_samples: Maximum validation images to load
        max_test_samples: Maximum test images to load
        num_workers: PyTorch DataLoader worker count

    Returns:
        train_loader, val_loader, test_loader, in_channels (3), num_classes
    """
    os.makedirs(data_dir, exist_ok=True)
    train_tf, eval_tf = get_transforms(image_size=image_size)

    if dataset_name.lower() in ["pneumonia", "pneumoniamnist", "chest_xray"]:
        DataClass = medmnist.PneumoniaMNIST
        info = INFO["pneumoniamnist"]
        num_classes = len(info["label"])  # 2: normal vs pneumonia

        raw_train = DataClass(split="train", transform=train_tf, download=True, root=data_dir, as_rgb=True)
        raw_val = DataClass(split="val", transform=eval_tf, download=True, root=data_dir, as_rgb=True)
        raw_test = DataClass(split="test", transform=eval_tf, download=True, root=data_dir, as_rgb=True)

    elif dataset_name.lower() in ["derma", "dermamnist", "ham10000"]:
        DataClass = medmnist.DermaMNIST
        info = INFO["dermamnist"]
        num_classes = len(info["label"])  # 7: skin lesion categories

        raw_train = DataClass(split="train", transform=train_tf, download=True, root=data_dir, as_rgb=True)
        raw_val = DataClass(split="val", transform=eval_tf, download=True, root=data_dir, as_rgb=True)
        raw_test = DataClass(split="test", transform=eval_tf, download=True, root=data_dir, as_rgb=True)

    elif dataset_name.lower() == "custom" or os.path.exists(data_dir):
        # Local ImageFolder directory
        train_path = os.path.join(data_dir, "train") if os.path.exists(os.path.join(data_dir, "train")) else data_dir
        val_path = os.path.join(data_dir, "val") if os.path.exists(os.path.join(data_dir, "val")) else data_dir
        test_path = os.path.join(data_dir, "test") if os.path.exists(os.path.join(data_dir, "test")) else val_path

        raw_train = ImageFolder(root=train_path, transform=train_tf)
        raw_val = ImageFolder(root=val_path, transform=eval_tf)
        raw_test = ImageFolder(root=test_path, transform=eval_tf)
        num_classes = len(raw_train.classes)
    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Choose 'pneumonia', 'derma', or 'custom'.")

    # Optional subsetting for fast CPU training
    if max_train_samples and max_train_samples < len(raw_train):
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(raw_train), generator=generator)[:max_train_samples].tolist()
        train_set = Subset(raw_train, indices)
    else:
        train_set = raw_train

    if max_val_samples and max_val_samples < len(raw_val):
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(raw_val), generator=generator)[:max_val_samples].tolist()
        val_set = Subset(raw_val, indices)
    else:
        val_set = raw_val

    if max_test_samples and max_test_samples < len(raw_test):
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(raw_test), generator=generator)[:max_test_samples].tolist()
        test_set = Subset(raw_test, indices)
    else:
        test_set = raw_test

    # Wrap to guarantee tensor targets
    train_wrapped = SqueezeTargetDataset(train_set)
    val_wrapped = SqueezeTargetDataset(val_set)
    test_wrapped = SqueezeTargetDataset(test_set)

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, 3, num_classes

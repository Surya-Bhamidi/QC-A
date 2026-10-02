"""Official splits only; no test subsetting and no implicit downloads."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import TensorDataset


def stratified_indices(labels, fraction, seed):
    # Classwise permutations make subsets nested across fractions for a seed.
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    return np.sort(np.concatenate([g[:max(1, int(np.ceil(len(g) * fraction)))] for g in groups]))


def load_splits(config, data_dir='data'):
    path = Path(data_dir) / f'{config.dataset}.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path}: run python -m research.fetch_data {config.dataset}')
    raw = np.load(path, allow_pickle=False)
    datasets, counts, indices = {}, {}, {}
    for split in ('train', 'val', 'test'):
        x = raw[f'{split}_images']
        y = raw[f'{split}_labels'].reshape(-1).astype(np.int64)
        idx = stratified_indices(y, config.fraction, config.seed) if split == 'train' else np.arange(len(y))
        indices[split] = idx.tolist()
        x = x[idx]
        if x.ndim == 3:
            x = x[..., None]
        x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float().div_(255)
        if x.shape[-2:] != (config.image_size, config.image_size):
            x = torch.nn.functional.interpolate(x, (config.image_size, config.image_size), mode='bilinear', align_corners=False)
        x = x.mul_(2).sub_(1)
        y = torch.from_numpy(y[idx].copy())
        datasets[split] = TensorDataset(x, y)
        counts[split] = {'n': len(y), 'class_counts': torch.bincount(y).tolist()}
    metadata = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'splits': counts, 'indices': indices}
    return datasets, int(datasets['train'].tensors[0].shape[1]), len(np.unique(raw['train_labels'])), metadata

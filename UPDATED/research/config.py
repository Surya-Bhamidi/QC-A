from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Config:
    dataset: str = 'pneumoniamnist'
    model: str = 'fidelity'
    seed: int = 0
    image_size: int = 28
    patch_size: int = 7
    embed_dim: int = 32
    heads: int = 2
    qk_dim: int = 4
    depth: int = 2
    mlp_dim: int = 64
    dropout: float = 0.1
    beta: float = 2.0
    epochs: int = 10
    batch_size: int = 64
    lr: float = 0.0008
    weight_decay: float = 0.0001
    fraction: float = 1.0
    class_weight: bool = True
    augment: bool = False
    threads: int = 4

    def __post_init__(self):
        if self.dataset not in {'pneumoniamnist', 'breastmnist', 'dermamnist', 'bloodmnist', 'retinamnist'}:
            raise ValueError('Use an official supported MedMNIST split archive')
        if self.model not in {'dot_full', 'dot', 'cosine', 'rbf', 'fidelity'}:
            raise ValueError('Unknown attention model')
        if min(self.heads, self.depth, self.qk_dim, self.epochs, self.patch_size, self.batch_size, self.threads) < 1:
            raise ValueError('Dimensions and training counts must be positive')
        if self.embed_dim % self.heads or self.image_size % self.patch_size:
            raise ValueError('Embedding/head and image/patch dimensions must divide exactly')
        if not 0 < self.fraction <= 1 or self.beta <= 0:
            raise ValueError('Invalid training fraction or logit multiplier')

    def dict(self):
        return asdict(self)

    @property
    def experiment_id(self):
        digest = hashlib.sha256(json.dumps(self.dict(), sort_keys=True).encode()).hexdigest()[:10]
        return f'{self.dataset}-{self.model}-s{self.seed}-{digest}'

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

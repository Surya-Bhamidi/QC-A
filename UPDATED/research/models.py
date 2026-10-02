"""Shared backbone, with exactly parameter-matched low-dimensional kernels."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def angles(x):
    return (x.tanh() + 1) * (math.pi / 2)


def product_fidelity(q, k):
    return ((angles(q).unsqueeze(-2) - angles(k).unsqueeze(-3)).mul(.5).cos().square()).prod(-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.kind, self.heads, self.beta = c.model, c.heads, c.beta
        self.dk = c.embed_dim // c.heads if c.model == 'dot_full' else c.qk_dim
        self.dv = c.embed_dim // c.heads
        self.q = nn.Linear(c.embed_dim, c.heads * self.dk)
        self.k = nn.Linear(c.embed_dim, c.heads * self.dk)
        self.v = nn.Linear(c.embed_dim, c.embed_dim)
        self.out = nn.Linear(c.embed_dim, c.embed_dim)
        self.drop = nn.Dropout(c.dropout)
        self.estimator = None
        self.capture = False
        self.last = None

    def forward(self, x):
        b, n, e = x.shape
        q = self.q(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        k = self.k(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, self.dv).transpose(1, 2)
        if self.kind.startswith('dot'):
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        elif self.kind == 'cosine':
            s = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
        elif self.kind == 'rbf':
            s = (-(q.unsqueeze(-2) - k.unsqueeze(-3)).square().sum(-1) / (2 * self.dk)).exp()
        else:
            s = product_fidelity(q, k)
        ideal = s
        audit = None
        if self.estimator is not None:
            if self.kind != 'fidelity' or self.training:
                raise ValueError('Measurement estimators are inference-only fidelity backends')
            s, audit = self.estimator(q, k, v, s, self.beta)
        a = (self.beta * s).softmax(-1)
        if self.capture:
            self.last = {'q': q.detach(), 'k': k.detach(), 'v': v.detach(), 'ideal': ideal.detach(),
                         'estimated': s.detach(), 'attention': a.detach(), 'audit': audit}
        y = (self.drop(a) @ v).transpose(1, 2).reshape(b, n, e)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(c.embed_dim), nn.LayerNorm(c.embed_dim)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.embed_dim, c.mlp_dim), nn.GELU(), nn.Dropout(c.dropout),
                                 nn.Linear(c.mlp_dim, c.embed_dim), nn.Dropout(c.dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MatchedViT(nn.Module):
    def __init__(self, c, channels=1, classes=2):
        super().__init__()
        self.config = c
        self.patch = nn.Conv2d(channels, c.embed_dim, c.patch_size, stride=c.patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, c.embed_dim))
        self.position = nn.Parameter(torch.randn(1, (c.image_size // c.patch_size)**2 + 1, c.embed_dim) * .02)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.depth)])
        self.norm = nn.LayerNorm(c.embed_dim)
        self.head = nn.Linear(c.embed_dim, classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.drop(torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.position)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])

    def set_estimator(self, estimator=None, capture=False):
        for block in self.blocks:
            block.attn.estimator = estimator
            block.attn.capture = capture

    def resources(self):
        c = self.config
        n = (c.image_size // c.patch_size)**2 + 1
        d = c.embed_dim // c.heads if c.model == 'dot_full' else c.qk_dim
        e = c.embed_dim
        channels = self.patch.in_channels
        # Dense multiply-accumulates only; kernel elementwise/softmax/nonlinear costs excluded.
        macs = (n - 1) * e * channels * c.patch_size**2
        macs += c.depth * (2*n*e*c.heads*d + 2*n*e*e + 2*n*e*c.mlp_dim + n*n*e)
        if c.model.startswith('dot') or c.model == 'cosine':
            macs += c.depth*n*n*c.heads*d
        macs += e*self.head.out_features
        return {'parameters': sum(p.numel() for p in self.parameters()), 'dense_macs_per_image': macs,
                'macs_exclusions': 'elementwise kernel, normalization, activation, softmax, dropout',
                'pairs_per_image': c.depth*c.heads*n*n if c.model == 'fidelity' else 0}

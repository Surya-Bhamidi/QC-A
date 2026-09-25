"""
classical_vit.py - Classical Vision Transformer (ViT) Baseline

Architecture:
1. PatchEmbedding: extracts patches from 128x128 image, projects linearly to embed_dim.
2. Positional Embedding + CLS Token: standard learnable 1D spatial embeddings.
3. Multi-Head Self-Attention: standard scaled dot-product softmax(QK^T / sqrt(d)) * V.
4. TransformerEncoderBlock: LayerNorm -> MHSA -> Residual -> LayerNorm -> MLP -> Residual.
5. Classification Head: maps CLS token (or pooled tokens) to target classes.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects them to embed_dim.
    For an image of size (C, H, W) and patch size P:
    Number of patches N = (H // P) * (W // P).
    """
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Conv2d with kernel=P and stride=P is equivalent to flattening patches + linear projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, C, H, W)
        # Proj output: (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class ClassicalMultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, embed_dim: int = 32, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape

        # Compute Q, K, V: shape (B, num_heads, N, head_dim)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product: (B, num_heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Weighted values: (B, num_heads, N, head_dim) -> (B, N, embed_dim)
        out = torch.matmul(attn_weights_drop, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        if return_attn:
            return out, attn_weights
        return out, None


class MLPBlock(nn.Module):
    """Feed-forward MLP network with GELU activation and dropout."""
    def __init__(self, embed_dim: int = 32, mlp_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Standard pre-LayerNorm Transformer Encoder block.
    x = x + MHSA(LN(x))
    x = x + MLP(LN(x))
    """
    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 2,
        mlp_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ClassicalMultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ClassicalViT(nn.Module):
    """
    Classical Vision Transformer Baseline.

    Args:
        image_size: Input image dimension (e.g. 128)
        patch_size: Square patch size (e.g. 16 or 32)
        in_channels: Input image channels (3 for RGB)
        num_classes: Classification categories (2 for Pneumonia vs Normal)
        embed_dim: Token embedding dimension (default 32)
        depth: Number of TransformerEncoderBlocks (default 2)
        num_heads: Number of attention heads (default 2)
        mlp_dim: Hidden dimension in MLP block (default 64)
        dropout: Dropout rate (default 0.1)
    """
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 2,
        mlp_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # Learnable CLS token and 1D positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B = x.shape[0]

        # Patch embedding: (B, N, embed_dim)
        x = self.patch_embed(x)

        # Prepend [CLS] token: (B, 1 + N, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        all_attns = []
        for block in self.blocks:
            x, attn_weights = block(x, return_attn=return_attn)
            if return_attn:
                all_attns.append(attn_weights)

        # Final LayerNorm
        x = self.norm(x)

        # Classification logits from [CLS] token (index 0)
        cls_out = x[:, 0]
        logits = self.head(cls_out)

        if return_attn:
            return logits, all_attns
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

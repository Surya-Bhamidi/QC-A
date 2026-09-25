"""
qvit.py - Quantum Vision Transformer (QViT) Model Architecture

Replaces the classical scaled dot-product attention in Transformer encoder blocks
with a Quantum Self-Attention (QSA) block powered by the Quantum SWAP Test Circuit.

Features:
1. Linear projection of Q and K down to quantum register size (default 4 qubits).
2. Quantum SWAP test circuit computing state fidelity between each patch pair.
3. Temperature-scaled attention weighting over classical Value (V) representations.
4. Identical classical backbone (patch embedding, positional embeddings, MLP, classification head)
   for fair, rigorous academic comparison.
5. Dynamic noise injection support for NISQ resilience testing on PennyLane's default.mixed.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classical_vit import PatchEmbedding, MLPBlock
from src.quantum.swap_test import QuantumSwapTestModule


class QuantumSelfAttention(nn.Module):
    """
    Quantum Self-Attention (QSA) Block using SWAP Test State Overlap.

    Mathematical flow:
    1. Q = Linear(embed_dim, n_qubits)(x)
    2. K = Linear(embed_dim, n_qubits)(x)
    3. V = Linear(embed_dim, embed_dim)(x)
    4. A_quantum[i, j] = Fidelity(|psi_{Q_i}>, |phi_{K_j}>) via PennyLane SWAP Test
    5. Attention Weights = Softmax(A_quantum / tau, dim=-1)
    6. Output = Attention Weights * V -> Out_Linear(embed_dim, embed_dim)
    """
    def __init__(
        self,
        embed_dim: int = 32,
        n_qubits: int = 4,
        num_heads: int = 1,
        temperature: float = 0.5,
        dropout: float = 0.0,
        device_type: str = "default.qubit",
        noise_prob: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        # Linear projections for Query, Key, and Value
        # Q and K project from embed_dim down to num_heads * n_qubits (parameter efficient!)
        self.q_proj = nn.Linear(embed_dim, num_heads * n_qubits)
        self.k_proj = nn.Linear(embed_dim, num_heads * n_qubits)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Quantum SWAP Test Kernel
        self.quantum_kernel = QuantumSwapTestModule(
            n_qubits=n_qubits,
            device_type=device_type,
            noise_prob=noise_prob,
        )

    def set_noise(self, noise_prob: float):
        """Sets depolarizing noise rate for NISQ simulation."""
        self.quantum_kernel.set_noise(noise_prob)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape

        if self.num_heads == 1:
            # Single-head QSA
            q = self.q_proj(x)  # (B, N, n_qubits)
            k = self.k_proj(x)  # (B, N, n_qubits)
            v = self.v_proj(x)  # (B, N, embed_dim)

            fidelity_matrix = self.quantum_kernel(q, k)  # (B, N, N)

            # Learnable calibration & temperature-scaled softmax
            scaled_fidelity = (self.scale * fidelity_matrix + self.bias) / torch.clamp(self.temperature, min=0.05)
            attn_weights = F.softmax(scaled_fidelity, dim=-1)
            attn_weights_drop = self.dropout(attn_weights)

            out = torch.matmul(attn_weights_drop, v)
            out = self.out_proj(out)
            disp_attn = attn_weights
        else:
            # Multi-head QSA
            q = self.q_proj(x).view(B, N, self.num_heads, self.n_qubits).transpose(1, 2)  # (B, H, N, d)
            k = self.k_proj(x).view(B, N, self.num_heads, self.n_qubits).transpose(1, 2)  # (B, H, N, d)
            v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)

            fidelity_matrix = self.quantum_kernel(q, k)  # (B, H, N, N)

            scaled_fidelity = (self.scale * fidelity_matrix + self.bias) / torch.clamp(self.temperature, min=0.05)
            attn_weights = F.softmax(scaled_fidelity, dim=-1)
            attn_weights_drop = self.dropout(attn_weights)

            out = torch.matmul(attn_weights_drop, v)  # (B, H, N, head_dim)
            out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
            out = self.out_proj(out)
            disp_attn = attn_weights.mean(dim=1)

        if return_attn:
            return out, disp_attn
        return out, None


class QuantumTransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder block equipped with Quantum Self-Attention (QSA).
    x = x + QSA(LN(x))
    x = x + MLP(LN(x))
    """
    def __init__(
        self,
        embed_dim: int = 32,
        n_qubits: int = 4,
        num_heads: int = 1,
        mlp_dim: int = 64,
        dropout: float = 0.0,
        device_type: str = "default.qubit",
        noise_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = QuantumSelfAttention(
            embed_dim=embed_dim,
            n_qubits=n_qubits,
            num_heads=num_heads,
            dropout=dropout,
            device_type=device_type,
            noise_prob=noise_prob,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout)

    def set_noise(self, noise_prob: float):
        self.attn.set_noise(noise_prob)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class QViT(nn.Module):
    """
    Quantum Vision Transformer (QViT) Hybrid Model.

    Args:
        image_size: Input image dimension (e.g. 128)
        patch_size: Square patch size (e.g. 16 or 32)
        in_channels: Input image channels (3 for RGB)
        num_classes: Number of target classes (e.g. 2 for Pneumonia vs Normal)
        embed_dim: Patch embedding dimension (default 32)
        n_qubits: Number of qubits per Query/Key register (default 4, total wires = 9)
        depth: Number of TransformerEncoderBlocks (default 2)
        mlp_dim: Hidden dimension in classical MLP block (default 64)
        dropout: Dropout rate (default 0.1)
        device_type: PennyLane simulator device ('default.qubit' or 'default.mixed')
        noise_prob: Depolarizing noise probability for NISQ analysis (default 0.0)
    """
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 32,
        n_qubits: int = 4,
        num_heads: int = 1,
        depth: int = 2,
        mlp_dim: int = 64,
        dropout: float = 0.1,
        device_type: str = "default.qubit",
        noise_prob: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.num_heads = num_heads
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

        # Quantum Transformer encoder blocks
        self.blocks = nn.ModuleList([
            QuantumTransformerEncoderBlock(
                embed_dim=embed_dim,
                n_qubits=n_qubits,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                device_type=device_type,
                noise_prob=noise_prob,
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

    def set_noise(self, noise_prob: float):
        """Sets depolarizing noise across all quantum attention blocks."""
        for block in self.blocks:
            block.set_noise(noise_prob)

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

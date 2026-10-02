# Adapted from FD-loss (https://github.com/Jiawei-Yang/FD-loss, MIT license).
# MiT is the pMF backbone; MiTEncoder / MiTDecoder reuse its blocks for the VAEs.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .commons import (
    TorchLinear, RMSNorm, SwiGLUMlp, PatchEmbedder, BottleneckPatchEmbed,
    apply_rotary_pos_emb, apply_rotary_pos_emb_partial,
    precompute_rope_freqs, precompute_rope_freqs_2d,
    TimestepEmbedder, LabelEmbedder,
)

logger = logging.getLogger("nv")

# "sdpa" uses F.scaled_dot_product_attention; "einsum" is the original explicit
# softmax, which forward-mode AD (torch.func.jvp) always supports.
ATTN_IMPL = {"value": "sdpa"}


class RoPEAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, weight_init="scaled_variance",
                 weight_init_constant=1.0, rope_func=apply_rotary_pos_emb):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_func = rope_func

        init_kwargs = dict(
            in_features=hidden_size, out_features=hidden_size,
            bias=False, weight_init=weight_init, init_constant=weight_init_constant,
        )
        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_freqs):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.rope_func(self.q_norm(q), rope_freqs)
        k = self.rope_func(self.k_norm(k), rope_freqs)

        if ATTN_IMPL["value"] == "sdpa":
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            ).transpose(1, 2)
        else:
            query = q / math.sqrt(self.head_dim)
            attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
            attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        return self.out_proj(attn.reshape(batch, seq_len, self.hidden_size))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=8 / 3, weight_init="scaled_variance",
                 weight_init_constant=1.0, rope_func=apply_rotary_pos_emb):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(
            hidden_size, num_heads=num_heads,
            weight_init=weight_init, weight_init_constant=weight_init_constant, rope_func=rope_func,
        )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if hidden_size > 1024:  # match upstream pMF rounding for H/XL models
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            hidden_size, mlp_hidden_dim,
            weight_init=weight_init, weight_init_constant=weight_init_constant,
        )

        # zero-initialized vector gates
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_features):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(
            hidden_size, out_features,
            bias=True, weight_init="zeros", bias_init="zeros",
        )

    def forward(self, x):
        return self.linear(self.norm(x))


def unpatchify(x, patch_size, channels):
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    x = x.reshape((x.shape[0], h, w, patch_size, patch_size, channels))
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape((x.shape[0], channels, h * patch_size, w * patch_size))


class _TokenTransformer(nn.Module):
    """Shared plumbing: rope / learned positional embedding and checkpointed blocks."""

    grad_checkpointing = False

    def _setup_positions(self, num_patches, prefix_tokens, hidden_size, num_heads, rope_2d, learned_pe):
        head_dim = hidden_size // num_heads
        total_tokens = num_patches + prefix_tokens
        if rope_2d:
            cos, sin = precompute_rope_freqs_2d(head_dim, num_patches)
            rope_func = apply_rotary_pos_emb_partial
        else:
            cos, sin = precompute_rope_freqs(head_dim, total_tokens)
            rope_func = apply_rotary_pos_emb
        # buffers so they live on the model's device (no per-block host->device copy)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if learned_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, hidden_size) * 0.02)
        else:
            self.pos_embed = None
        return rope_func

    @property
    def rope_freqs(self):
        return self.rope_cos, self.rope_sin

    def _add_pos(self, seq):
        return seq + self.pos_embed if self.pos_embed is not None else seq

    def _run_block(self, block, seq):
        if self.grad_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(block, seq, self.rope_freqs, use_reentrant=False)
        return block(seq, self.rope_freqs)


class MiT(_TokenTransformer):
    """meanflow improved transformer with shared backbone and dual u/v heads (pMF backbone)."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        bottleneck_dim: int = -1,
        aux_head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        rope_2d: bool = False,
        learned_pe: bool = False,
        disable_v_head: bool = False,
        output_type: str = "v",
        t_eps: float = 0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.out_channels = in_channels
        self.output_type = output_type
        self.t_eps = t_eps
        self.aux_head_depth = aux_head_depth
        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens

        if bottleneck_dim > 0:
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True,
            )
        else:
            self.x_embedder = PatchEmbedder(input_size, patch_size, in_channels, hidden_size, bias=True)

        embed_kwargs = dict(hidden_size=hidden_size, weight_init="scaled_variance", init_constant=embedding_init_constant)
        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        if num_cfg_tokens > 0 and num_interval_tokens > 0:
            self.omega_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(num_classes, **embed_kwargs)

        # learnable type tokens
        token_init = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
        self.time_tokens = nn.Parameter(token_init(torch.empty(num_time_tokens, hidden_size)))
        self.class_tokens = nn.Parameter(token_init(torch.empty(num_class_tokens, hidden_size)))
        if num_cfg_tokens > 0 and num_interval_tokens > 0:
            self.omega_tokens = nn.Parameter(token_init(torch.empty(num_cfg_tokens, hidden_size)))
            self.t_min_tokens = nn.Parameter(token_init(torch.empty(num_interval_tokens, hidden_size)))
            self.t_max_tokens = nn.Parameter(token_init(torch.empty(num_interval_tokens, hidden_size)))

        self.prefix_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens + num_time_tokens
        rope_func = self._setup_positions(
            self.x_embedder.num_patches, self.prefix_tokens, hidden_size, num_heads, rope_2d, learned_pe,
        )

        shared_depth = depth - aux_head_depth
        block_kwargs = dict(
            hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            weight_init="scaled_variance", weight_init_constant=weight_init_constant, rope_func=rope_func,
        )
        self.shared_blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(shared_depth)])
        self.u_heads = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
        self.u_final_layer = FinalLayer(hidden_size, patch_size * patch_size * self.out_channels)

        self.disable_v_head = disable_v_head
        if not disable_v_head:
            self.v_heads = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
            self.v_final_layer = FinalLayer(hidden_size, patch_size * patch_size * self.out_channels)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"[MiT] params: {n_params:.2f}M, depth: {depth}, hidden_size: {hidden_size}, "
                    f"prefix_tokens: {self.prefix_tokens}, num_patches: {self.x_embedder.num_patches}")

    def output_conversion(self, z, x, t):
        if self.output_type == "v":
            return x
        t = t.reshape(x.shape[0], 1, 1, 1)
        return (z - x) / torch.clamp(t, self.t_eps, 1.0)

    def _build_sequence(self, x, h, omega=None, t_min=None, t_max=None, y=None):
        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        y_embed = self.y_embedder(y)
        if self.num_cfg_tokens > 0 and self.num_interval_tokens > 0:
            omega_embed = self.omega_embedder(1 - 1 / omega)
            t_min_embed = self.cfg_t_start_embedder(t_min)
            t_max_embed = self.cfg_t_end_embedder(t_max)
            seq = torch.cat([
                self.class_tokens + y_embed.unsqueeze(1),
                self.omega_tokens + omega_embed.unsqueeze(1),
                self.t_min_tokens + t_min_embed.unsqueeze(1),
                self.t_max_tokens + t_max_embed.unsqueeze(1),
                self.time_tokens + h_embed.unsqueeze(1),
                x_embed,
            ], dim=1)
        else:
            seq = torch.cat([
                self.class_tokens + y_embed.unsqueeze(1),
                self.time_tokens + h_embed.unsqueeze(1),
                x_embed,
            ], dim=1)
        return self._add_pos(seq)

    def forward(self, x, t, h, omega=None, t_min=None, t_max=None, y=None):
        # we don't condition on t, only on h = t - r (https://arxiv.org/abs/2502.13129)
        seq = self._build_sequence(x, h, omega=omega, t_min=t_min, t_max=t_max, y=y)
        for block in self.shared_blocks:
            seq = self._run_block(block, seq)

        u_seq = seq
        for block in self.u_heads:
            u_seq = self._run_block(block, u_seq)
        u = unpatchify(self.u_final_layer(u_seq[:, self.prefix_tokens:]), self.patch_size, self.out_channels)

        if self.disable_v_head:
            u_out = self.output_conversion(x, u, t)
            return u_out, torch.zeros_like(u_out)

        v_seq = seq
        for block in self.v_heads:
            v_seq = self._run_block(block, v_seq)
        v = unpatchify(self.v_final_layer(v_seq[:, self.prefix_tokens:]), self.patch_size, self.out_channels)
        return self.output_conversion(x, u, t), self.output_conversion(x, v, t)


MiT_T = partial(MiT, depth=4, hidden_size=512, num_heads=8)
MiT_B = partial(MiT, depth=12, hidden_size=768, num_heads=12)
MiT_B2 = partial(MiT, depth=16, hidden_size=768, num_heads=12)
MiT_M = partial(MiT, depth=24, hidden_size=768, num_heads=12)
MiT_L = partial(MiT, depth=32, hidden_size=1024, num_heads=16)
MiT_XL = partial(MiT, depth=48, hidden_size=1024, num_heads=16)
MiT_H = partial(MiT, depth=48, hidden_size=1280, num_heads=16)

MiT_models = {
    "MiT_T": MiT_T, "MiT_B": MiT_B, "MiT_B2": MiT_B2, "MiT_M": MiT_M,
    "MiT_L": MiT_L, "MiT_XL": MiT_XL, "MiT_H": MiT_H,
}


# ---------------------------------------------------------------------------
# VAE encoder / decoder: same blocks; prefix = class tokens (+ register tokens)
# Sequence layout: [class tokens | register tokens | patch tokens]. Register tokens are
# part of the latent: the encoder emits a posterior for them and the decoder reads them
# as extra non-spatial tokens (rope only rotates the patch tokens).
# ---------------------------------------------------------------------------

class _ClassCondTransformer(_TokenTransformer):
    def __init__(self, num_patches, hidden_size, depth, num_heads, mlp_ratio, num_classes,
                 num_class_tokens, num_registers, token_init_constant, embedding_init_constant,
                 weight_init_constant, rope_2d, learned_pe):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_class_tokens = num_class_tokens
        self.num_registers = num_registers
        self.num_patches = num_patches
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, weight_init="scaled_variance",
                                        init_constant=embedding_init_constant)
        token_init = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
        self.class_tokens = nn.Parameter(token_init(torch.empty(num_class_tokens, hidden_size)))
        rope_func = self._setup_positions(num_patches, num_class_tokens + num_registers, hidden_size,
                                          num_heads, rope_2d, learned_pe)
        block_kwargs = dict(
            hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            weight_init="scaled_variance", weight_init_constant=weight_init_constant, rope_func=rope_func,
        )
        self.blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(depth)])

    def _trunk(self, tokens, y, taps=(), stop_after=None):
        """tokens: (B, num_registers + num_patches, hidden). Returns the same positions, plus
        {k: features after block k} for k in taps (1-indexed); stops early after stop_after."""
        seq = torch.cat([self.class_tokens + self.y_embedder(y).unsqueeze(1), tokens], dim=1)
        seq = self._add_pos(seq)
        feats = {}
        for i, block in enumerate(self.blocks):
            seq = self._run_block(block, seq)
            if i + 1 in taps:
                feats[i + 1] = seq[:, self.num_class_tokens:]
            if stop_after is not None and i + 1 >= stop_after:
                break
        return seq[:, self.num_class_tokens:], feats


class MiTEncoder(_ClassCondTransformer):
    """image -> per-token posterior (mu, logvar), each (B, num_registers + num_patches, latent_dim).
    With latent_dim=0 there is no head and the token features (B, R + T, hidden) are returned."""

    def __init__(self, input_size=128, patch_size=16, in_channels=3, hidden_size=768, depth=8,
                 num_heads=12, mlp_ratio=8 / 3, latent_dim=768, num_classes=1000, num_class_tokens=8,
                 num_registers=0, token_init_constant=1.0, embedding_init_constant=1.0,
                 weight_init_constant=0.32, rope_2d=True, learned_pe=True):
        # full-rank patch embedding: no bottleneck anywhere on the encoder path
        x_embedder = PatchEmbedder(input_size, patch_size, in_channels, hidden_size, bias=True)
        super().__init__(x_embedder.num_patches, hidden_size, depth, num_heads, mlp_ratio, num_classes,
                         num_class_tokens, num_registers, token_init_constant, embedding_init_constant,
                         weight_init_constant, rope_2d, learned_pe)
        self.x_embedder = x_embedder
        self.latent_dim = latent_dim
        if num_registers > 0:
            token_init = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
            self.register_tokens = nn.Parameter(token_init(torch.empty(num_registers, hidden_size)))
        self.head = FinalLayer(hidden_size, 2 * latent_dim) if latent_dim > 0 else None

    def forward(self, x, y):
        tokens = self.x_embedder(x)
        if self.num_registers > 0:
            regs = self.register_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            tokens = torch.cat([regs, tokens], dim=1)
        feats = self._trunk(tokens, y)[0]
        if self.head is None:
            return feats
        out = self.head(feats).float()
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class MiTDecoder(_ClassCondTransformer):
    """latent tokens (B, num_registers + num_patches, latent_dim) -> image (x-prediction)."""

    def __init__(self, input_size=128, patch_size=16, out_channels=3, hidden_size=768, depth=16,
                 num_heads=12, mlp_ratio=8 / 3, latent_dim=768, num_classes=1000, num_class_tokens=8,
                 num_registers=0, token_init_constant=1.0, embedding_init_constant=1.0,
                 weight_init_constant=0.32, rope_2d=True, learned_pe=True):
        num_patches = (input_size // patch_size) ** 2
        super().__init__(num_patches, hidden_size, depth, num_heads, mlp_ratio, num_classes,
                         num_class_tokens, num_registers, token_init_constant, embedding_init_constant,
                         weight_init_constant, rope_2d, learned_pe)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.z_embedder = TorchLinear(latent_dim, hidden_size, bias=True, weight_init="scaled_variance")
        self.final_layer = FinalLayer(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, z, y, taps=(), stop_after=None):
        """Image (x-prediction); with taps also {block: token features}; with stop_after only features."""
        tokens, feats = self._trunk(self.z_embedder(z), y, taps=taps, stop_after=stop_after)
        if stop_after is not None:
            return feats
        img = unpatchify(self.final_layer(tokens[:, self.num_registers:]), self.patch_size, self.out_channels)
        return (img, feats) if taps else img

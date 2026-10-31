"""Tensor-Var ERA5 deep kernel feature operators.

These modules form the *physics surrogate* of the task: a learned encoder /
decoder pair (`ERA5_K_S`, `ERA5_K_S_preimage`) that maps weather state fields
to a finite-dimensional feature space, a transformer-based inverse-observation
model (`ERA5_K_O`) that maps a window of observations to a state field, and
the linear forward operator `C_forward` that propagates features one step in
time.

Module/parameter names exactly mirror the upstream Tensor-Var repo so the
released `forward_model.pt` / `inverse_model.pt` checkpoints load without any
key remapping.
"""

from __future__ import annotations

import numbers
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Module dimensions (fixed by the released checkpoints)
# ---------------------------------------------------------------------------
ERA5_SETTINGS = {
    "obs_dim":                  [5, 64, 32],
    "history_len":              10,
    "state_dim":                [5, 64, 32],
    "seq_length":               10,
    "obs_feature_dim":          [512, 128, 64, 32, 16, 8],
    "state_filter_feature_dim": [32, 64, 128, 256],
}
_filter = ERA5_SETTINGS["state_filter_feature_dim"]
_state = ERA5_SETTINGS["state_dim"]
ERA5_SETTINGS["state_feature_dim"] = [
    int(_filter[-1] * (_state[1] * _state[2]) / 256),
    512,
]


# ---------------------------------------------------------------------------
# Restormer-style transformer building blocks (from Tensor-Var)
# ---------------------------------------------------------------------------
def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Upsample_Flex(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.body(x)


class Transformer_Based_Inv_Obs_Model(nn.Module):
    def __init__(
        self,
        in_channel: int = 50,
        out_channel: int = 5,
        LayerNorm_type: str = "WithBias",
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        num_blocks=(2, 2, 2, 2),
    ):
        super().__init__()
        dim_list = [in_channel * 2, in_channel * 4, in_channel * 2, out_channel]
        num_heads = [5, 10, 5, 1]
        self.patch_embed = OverlapPatchEmbed(in_channel, embed_dim=dim_list[0])
        self.Upsample_1 = Upsample_Flex(dim_list[0], dim_list[1])
        self.Upsample_2 = Upsample_Flex(dim_list[1], dim_list[2])
        self.Upsample_3 = Upsample_Flex(dim_list[2], dim_list[3])
        self.block1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim_list[0],
                    num_heads=num_heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.block2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim_list[1],
                    num_heads=num_heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.block3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim_list[2],
                    num_heads=num_heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.block4 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim_list[3],
                    num_heads=num_heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.block1(x)
        x = self.Upsample_1(x)
        x = self.block2(x)
        x = self.Upsample_2(x)
        x = self.block3(x)
        x = self.Upsample_3(x)
        x = self.block4(x)
        return x


# ---------------------------------------------------------------------------
# Encoder K_S, decoder K_S_preimage, observation network K_O
# ---------------------------------------------------------------------------
class ERA5_K_S(nn.Module):
    """Convolutional encoder mapping (B, 5, 64, 32) state -> (B, 512) feature."""

    def __init__(self):
        super().__init__()
        self.input_dim, self.w, self.h = ERA5_SETTINGS["state_dim"]
        self.filter_dims = ERA5_SETTINGS["state_filter_feature_dim"]
        self.hidden_dims = ERA5_SETTINGS["state_feature_dim"]

        self.Conv2D_size5_1 = nn.Conv2d(self.input_dim, self.filter_dims[0], 9, 1, 4)
        self.Conv2D_size5_2 = nn.Conv2d(self.filter_dims[0], self.filter_dims[0], 5, 1, 2)
        self.Conv2D_size3_1 = nn.Conv2d(self.filter_dims[0], self.filter_dims[1], 3, 1, 1)
        self.Conv2D_size3_2 = nn.Conv2d(self.filter_dims[1], self.filter_dims[2], 3, 1, 1)
        self.Conv2D_size3_3 = nn.Conv2d(self.filter_dims[2], self.filter_dims[3], 3, 1, 1)

        self.flatten = nn.Flatten()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])

    def forward(self, state: torch.Tensor, return_encode_list: bool = False):
        encode_list: Optional[list] = [state.clone()] if return_encode_list else None

        x = self.Conv2D_size5_1(state)
        if return_encode_list:
            encode_list.append(x.clone())
        x = self.relu(self.pooling(x))

        x = self.Conv2D_size5_2(x)
        if return_encode_list:
            encode_list.append(x.clone())
        x = self.pooling(self.relu(x))

        x = self.Conv2D_size3_1(x)
        if return_encode_list:
            encode_list.append(x.clone())
        x = self.relu(self.pooling(x))

        x = self.Conv2D_size3_2(x)
        if return_encode_list:
            encode_list.append(x.clone())
        x = self.relu(self.pooling(x))

        x = self.Conv2D_size3_3(x)
        if return_encode_list:
            encode_list.append(x.clone())
        x = self.relu(x)

        z = self.linear(self.flatten(x))
        if return_encode_list:
            return z, encode_list
        return z


class ERA5_K_S_preimage(nn.Module):
    """Convolutional decoder mapping latent (B, 512) -> state (B, 5, 64, 32).

    When `encode_list` is supplied, the matching encoder feature maps are added
    as skip connections at every spatial scale.
    """

    def __init__(self):
        super().__init__()
        self.input_dim, self.w, self.h = ERA5_SETTINGS["state_dim"]
        self.filter_dims = ERA5_SETTINGS["state_filter_feature_dim"]
        self.hidden_dims = ERA5_SETTINGS["state_feature_dim"]

        self.linear = nn.Linear(self.hidden_dims[1], self.hidden_dims[0])
        self.ConvTranspose2D_size3_1 = nn.ConvTranspose2d(self.filter_dims[3], self.filter_dims[2], 3, 1, 1)
        self.ConvTranspose2D_size3_2 = nn.ConvTranspose2d(self.filter_dims[2], self.filter_dims[1], 3, 1, 1)
        self.ConvTranspose2D_size3_3 = nn.ConvTranspose2d(self.filter_dims[1], self.filter_dims[0], 3, 1, 1)
        self.Upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ConvTranspose2D_size5_1 = nn.ConvTranspose2d(self.filter_dims[0], self.filter_dims[0], 5, 1, 2)
        self.ConvTranspose2D_size5_2 = nn.ConvTranspose2d(self.filter_dims[0], self.input_dim, 9, 1, 4)
        self.relu = nn.ReLU()
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 128, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, self.input_dim, 1, 1),
        )

    def forward(self, z: torch.Tensor, encode_list=None):
        x = self.linear(z)
        x = self.relu(x)
        x = x.view(-1, self.filter_dims[3], self.w // 16, self.h // 16)

        if encode_list is not None:
            x = self.ConvTranspose2D_size3_1(x + encode_list[-1])
            x = self.relu(x)

            x = self.Upsampling(x)
            x = self.ConvTranspose2D_size3_2(x + encode_list[-2])
            x = self.relu(x)

            x = self.Upsampling(x)
            x = self.ConvTranspose2D_size3_3(x + encode_list[-3])
            x = self.relu(x)

            x = self.Upsampling(x)
            x = self.ConvTranspose2D_size5_1(x + encode_list[-4])
            x = self.relu(x)

            x = self.Upsampling(x)
            x = self.ConvTranspose2D_size5_2(x + encode_list[-5])
            return self.output_conv(x)

        # No-skip path (kept for completeness; matches the upstream code).
        x = self.ConvTranspose2D_size3_1(x)
        x = self.relu(x)
        x = self.Upsampling(x)
        x = self.ConvTranspose2D_size3_2(x)
        x = self.relu(x)
        x = self.Upsampling(x)
        x = self.ConvTranspose2D_size3_3(x)
        x = self.relu(x)
        x = self.Upsampling(x)
        x = self.ConvTranspose2D_size5_1(x)
        x = self.relu(x)
        x = self.Upsampling(x)
        x = self.ConvTranspose2D_size5_2(x)
        return self.output_conv(x)


class ERA5_K_O(nn.Module):
    """Transformer-based inverse observation model: obs window -> state field."""

    def __init__(self):
        super().__init__()
        self.input_dim = ERA5_SETTINGS["obs_dim"][0] * ERA5_SETTINGS["history_len"]
        self.output_dim = ERA5_SETTINGS["state_dim"][0]
        self.features = Transformer_Based_Inv_Obs_Model(
            in_channel=self.input_dim, out_channel=self.output_dim
        )

    def forward(self, obs: torch.Tensor):
        return self.features(obs)


# ---------------------------------------------------------------------------
# High-level wrappers used by the cleaned solver
# ---------------------------------------------------------------------------
class ERA5ForwardModel(nn.Module):
    """Encoder + decoder + linear feature-space forward operator C_forward.

    State dict layout matches the upstream `ERA5_C_FORWARD` so the released
    `forward_model.pt` checkpoint loads directly. `C_forward` is loaded
    separately because the upstream code stores it as a free attribute.
    """

    def __init__(self):
        super().__init__()
        self.K_S = ERA5_K_S()
        self.K_S_preimage = ERA5_K_S_preimage()
        self.hidden_dim = self.K_S.hidden_dims[-1]
        self.seq_length = ERA5_SETTINGS["seq_length"]
        self.C_forward: Optional[torch.Tensor] = None

    def encode(self, state: torch.Tensor, return_encode_list: bool = True):
        return self.K_S(state, return_encode_list=return_encode_list)

    def decode(self, z: torch.Tensor, encode_list=None) -> torch.Tensor:
        return self.K_S_preimage(z, encode_list)

    def latent_forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.C_forward is None:
            raise RuntimeError("C_forward must be assigned before calling latent_forward")
        return torch.mm(z, self.C_forward)


class ERA5InverseModel(nn.Module):
    """Inverse observation model: 10-step observation history -> single state field.

    Wraps `ERA5_K_O` along with frozen copies of `K_S` / `K_S_preimage` so the
    released `inverse_model.pt` state dict loads without remapping.
    """

    def __init__(self):
        super().__init__()
        self.K_O = ERA5_K_O()
        self.K_S = ERA5_K_S()
        self.K_S_preimage = ERA5_K_S_preimage()
        for p in self.K_S.parameters():
            p.requires_grad = False
        for p in self.K_S_preimage.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.K_O(obs)

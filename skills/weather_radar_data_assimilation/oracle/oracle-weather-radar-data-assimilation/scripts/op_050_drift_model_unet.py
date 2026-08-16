"""Einops-free UNet drift model + checkpoint loader for flow-based radar DA.

Self-contained: depends only on torch. Run with the run-local python.
Instantiates the architecture matching the pretrained checkpoint and exposes
load_drift_model(checkpoint_path, device).
"""
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *a, **k):
        return self.fn(x, *a, **k) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half = dim // 2
        self.weights = nn.Parameter(torch.randn(half), requires_grad=not is_random)

    def forward(self, x):
        x = x.view(-1, 1)
        freqs = x * self.weights.view(1, -1) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, classes_emb_dim=None, groups=8):
        super().__init__()
        it = int(time_emb_dim)
        ic = int(classes_emb_dim) if classes_emb_dim is not None else 0
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(it + ic, dim_out * 2))
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            ce = tuple(filter(exists, (time_emb, class_emb)))
            ce = torch.cat(ce, dim=-1)
            ce = self.mlp(ce)
            ce = ce.view(ce.shape[0], ce.shape[1], 1, 1)
            scale_shift = ce.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        dh = out.shape[-1]
        out = out.reshape(b, self.heads, h, w, dh).permute(0, 1, 4, 2, 3).reshape(b, self.heads * dh, h, w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, dim=128,
                 dim_mults=(1, 2, 2, 2), resnet_block_groups=8,
                 learned_sinusoidal_cond=True, random_fourier_features=False,
                 learned_sinusoidal_dim=32, attn_dim_head=64, attn_heads=4,
                 use_classes=True):
        super().__init__()
        self.use_classes = use_classes
        self.in_channels = in_channels
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        bk = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            spe = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fdim = learned_sinusoidal_dim + 1
        else:
            spe = SinusoidalPosEmb(dim)
            fdim = dim
        self.time_mlp = nn.Sequential(spe, nn.Linear(fdim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        if self.use_classes:
            self.classes_emb = nn.Embedding(num_classes, dim)
            cd = dim * 4
            self.classes_mlp = nn.Sequential(nn.Linear(dim, cd), nn.GELU(), nn.Linear(cd, cd))
        else:
            cd = None
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (di, do) in enumerate(in_out):
            last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                bk(di, di, time_emb_dim=time_dim, classes_emb_dim=cd),
                bk(di, di, time_emb_dim=time_dim, classes_emb_dim=cd),
                Residual(PreNorm(di, LinearAttention(di))),
                Downsample(di, do) if not last else nn.Conv2d(di, do, 3, padding=1)]))
        md = dims[-1]
        self.mid_block1 = bk(md, md, time_emb_dim=time_dim, classes_emb_dim=cd)
        self.mid_attn = Residual(PreNorm(md, Attention(md, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = bk(md, md, time_emb_dim=time_dim, classes_emb_dim=cd)
        for ind, (di, do) in enumerate(reversed(in_out)):
            last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                bk(do + di, do, time_emb_dim=time_dim, classes_emb_dim=cd),
                bk(do + di, do, time_emb_dim=time_dim, classes_emb_dim=cd),
                Residual(PreNorm(do, LinearAttention(do))),
                Upsample(do, di) if not last else nn.Conv2d(do, di, 3, padding=1)]))
        self.out_channels = out_channels
        self.final_res_block = bk(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=cd)
        self.final_conv = nn.Conv2d(dim, self.out_channels, 1)

    def forward(self, x, time, classes=None):
        c = self.classes_mlp(self.classes_emb(classes)) if classes is not None else None
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for b1, b2, attn, ds in self.downs:
            x = b1(x, t, c); h.append(x); x = b2(x, t, c); x = attn(x); h.append(x); x = ds(x)
        x = self.mid_block1(x, t, c); x = self.mid_attn(x); x = self.mid_block2(x, t, c)
        for b1, b2, attn, us in self.ups:
            x = torch.cat((x, h.pop()), dim=1); x = b1(x, t, c)
            x = torch.cat((x, h.pop()), dim=1); x = b2(x, t, c); x = attn(x); x = us(x)
        x = torch.cat((x, r), dim=1); x = self.final_res_block(x, t, c)
        return self.final_conv(x)


class DriftModel(nn.Module):
    def __init__(self, in_channels=7, out_channels=1, unet_channels=128,
                 dim_mults=(1, 2, 2, 2), use_classes=False, num_classes=1):
        super().__init__()
        self.use_classes = use_classes
        self._arch = Unet(num_classes=num_classes, in_channels=in_channels,
                          out_channels=out_channels, dim=unet_channels, dim_mults=dim_mults,
                          resnet_block_groups=8, learned_sinusoidal_cond=True,
                          random_fourier_features=False, learned_sinusoidal_dim=32,
                          attn_dim_head=64, attn_heads=4, use_classes=use_classes)

    def forward(self, zt, t, y, cond=None):
        if not self.use_classes:
            y = None
        if cond is not None:
            zt = torch.cat([zt, cond], dim=1)
        return self._arch(zt, t, y)


def load_drift_model(checkpoint_path, device="cpu"):
    model = DriftModel(in_channels=7, out_channels=1, unet_channels=128,
                       dim_mults=(1, 2, 2, 2), use_classes=False, num_classes=1)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "public/visible_data/pretrained_models/latest.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_drift_model(ckpt, dev)
    zt = torch.randn(1, 1, 128, 128, device=dev)
    cond = torch.randn(1, 6, 128, 128, device=dev)
    t = torch.ones(1, device=dev)
    with torch.no_grad():
        out = m(zt, t, None, cond=cond)
    print("forward ok", tuple(out.shape))

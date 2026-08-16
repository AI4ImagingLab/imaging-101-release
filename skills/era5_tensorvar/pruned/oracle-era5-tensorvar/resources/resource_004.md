# Inverse-observation module construction

Important constructor constants:

```text
in_channel = 50
out_channel = 5
LayerNorm_type = "WithBias"
ffn_expansion_factor = 2.66
bias = False
num_blocks = (2, 2, 2, 2)
dim_list = [100, 200, 100, 5]
num_heads = [5, 10, 5, 1]
```

Forward sequence:

```text
x = OverlapPatchEmbed(50, 100)(obs)
x = block1(x)       # 2 transformer blocks, 100 channels, 5 heads
x = Conv3x3(100,200)(x)
x = block2(x)       # 2 transformer blocks, 200 channels, 10 heads
x = Conv3x3(200,100)(x)
x = block3(x)       # 2 transformer blocks, 100 channels, 5 heads
x = Conv3x3(100,5)(x)
x = block4(x)       # 2 transformer blocks, 5 channels, 1 head
return x            # (batch, 5, 64, 32)
```

Layer norm details:
- Bias-free variant: `x / sqrt(var(x, unbiased=False) + 1e-5) * weight`.
- With-bias variant: `(x - mean(x)) / sqrt(var(x, unbiased=False) + 1e-5) * weight + bias`.
- Convert `(B,C,H,W)` to `(B,H*W,C)` for normalization and back afterward.

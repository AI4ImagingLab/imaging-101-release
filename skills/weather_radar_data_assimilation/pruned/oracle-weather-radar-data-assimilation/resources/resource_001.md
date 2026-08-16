# Drift model I/O contract

- forward(zt, t, label, cond): zt=(B,1,128,128), cond=(B,6,128,128), t=(B,) float in [0,1], label=None.
- Concatenation order is [zt, cond] on channel dim -> 7 channels; this order matters (state first, then the six past frames).
- Output = predicted drift/velocity b(zt,t,cond), shape (B,1,128,128).
- Checkpoint dict keys: 'model_state_dict', 'optimizer_state_dict', 'step'. Load only 'model_state_dict'.
- State-dict param names are prefixed '_arch.' (the UNet lives under attribute _arch of the drift wrapper).
- The mid-block full-attention output reshape must interleave heads x head_dim back to channels; the bundled script does this with reshape/permute instead of einops.

# Real-NVP architecture reference

ActNorm (per-tensor scalar loc + log_scale_inv, data-dependent init):
```
forward:  out = (1/exp(log_scale_inv)) * (input + loc); logdet = in_dim * sum(-log_scale_inv)
reverse:  out = input*exp(log_scale_inv) - loc;         logdet = -in_dim * sum(-log_scale_inv)
init (first batch): loc = -mean(input); log_scale_inv = log(std(input) + 1e-6)
```

ZeroFC: Linear(in,out) with weight=bias=0, learnable scale=0; out = fc(input) * exp(scale*3).

AffineCoupling (affine=True, batch_norm=True):
```
hidden = max(2, int(ndim / (2*seqfrac)))     # seqfrac = 0.25
net = Sequential(Linear(in_half,hidden), LeakyReLU(0.01),
                 BatchNorm1d(hidden, eps=1e-2, affine=True),
                 Linear(hidden,hidden), LeakyReLU(0.01),
                 BatchNorm1d(hidden, eps=1e-2, affine=True),
                 ZeroFC(hidden, 2*(ndim//2)))
first two Linear layers: weight ~ N(0,0.05), bias = 0
forward: a,b = chunk(input,2); log_s0,t = net(a).chunk(2)
         log_s = tanh(log_s0); s = exp(log_s); out_b = (b+t)*s; logdet = sum(log_s)
```

Flow block: actnorm -> coupling -> reverse-index -> actnorm2 -> coupling2 -> reverse-index.
(reverse-index = out[:, arange(ndim-1,-1,-1)])

RealNVP: n_flow=16 blocks; per-block permutation order = RandomState(seed=i).permutation(ndim).
Sampling uses flow.reverse(z), z ~ N(0,I); then params = sigmoid(params_samp).

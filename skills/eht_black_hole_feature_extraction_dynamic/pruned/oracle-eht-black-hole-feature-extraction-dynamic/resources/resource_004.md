# Alpha-DPI training loop reference (alpha_divergence = 1 default)

Weights: camp_weight=1.0; cphase_weight=n_cphase/n_camp; scale_factor=1/n_camp.

```
optimizer = Adam(flow.parameters(), lr=1e-4)
loss_best = 1e5; best_state = None; n_smooth = 10
for k in range(n_epoch):            # n_epoch = 5000
    data_weight = min(10 ** (-start_order + k / decay_rate), 1.0)   # start_order=4, decay_rate=1000
    z = randn(batch_size, nparams)                                   # nparams=4
    params_samp, logdet = flow.reverse(z)
    params = sigmoid(params_samp)
    img = geo_model(params)
    vis, visamp, cphase, logcamp = forward_model(img)
    det_sigmoid = sum(-params_samp - 2*softplus(-params_samp), dim=-1)
    logdet = logdet + det_sigmoid
    loss_cp = Loss_cphase(cphase_true, cphase)
    loss_ca = Loss_logca(logcamp_true, logcamp)
    loss_data = 0.5 * (camp_weight*loss_ca + cphase_weight*loss_cp) / scale_factor
    logprob = -logdet - 0.5*sum(z**2, dim=1)
    loss = data_weight*loss_data + logprob
    loss_train = mean(scale_factor * loss)          # alpha == 1
    optimizer.zero_grad(); loss_train.backward()
    clip_grad_norm_(flow.parameters(), grad_clip)   # grad_clip = 1e-4
    optimizer.step()
    elbo = mean(scale_factor*(loss_data + logprob))
    # track elbo; after k>n_smooth+1, if mean(last n_smooth elbos) <= loss_best: save state
restore best_state
```

alpha != 1, beta == 0 variant:
```
rej = softmax(-(1-alpha)*loss).detach()
loss_train = sum(rej * scale_factor * loss)
```

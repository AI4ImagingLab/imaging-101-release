# Importance sampling reference

```
n_samples = 10000; batch = min(n_samples, 2048)
cphase_weight = n_cphase / n_camp; camp_weight = 1.0
all_p = []; all_neg_logw = []
with no_grad():
  while remaining > 0:
    z = randn(bs, 4)
    params_samp, logdet = flow.reverse(z)
    params = sigmoid(params_samp)
    img = geo_model(params)
    vis, visamp, cphase, logcamp = forward_model(img)
    det_sigmoid = sum(-params_samp - 2*softplus(-params_samp), dim=-1)
    logdet = logdet + det_sigmoid
    loss_cp = Loss_cphase(cphase_true, cphase)
    loss_ca = Loss_logca(logcamp_true, logcamp)
    loss_data = camp_weight*loss_ca + cphase_weight*loss_cp
    logprob = -logdet - 0.5*sum(z**2, dim=1)
    loss_orig = 0.5*loss_data + logprob
    all_p.append(params.cpu().numpy()); all_neg_logw.append(loss_orig.cpu().numpy())
params_unit = concat(all_p)
importance_weights = scipy.special.softmax(-concat(all_neg_logw))
```

Physical conversion (columns [diameter, width, asymmetry, pa_deg]):
```
diameter = 2*(r_range[0] + p[:,0]*(r_range[1]-r_range[0]))
width    = width_range[0] + p[:,1]*(width_range[1]-width_range[0])
asymmetry = p[:,2]
pa_deg   = 181/180 * 180 * (2*p[:,3] - 1)
```

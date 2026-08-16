# Guided EM sampler reference (single frame)

Inputs: model, interpolant, base=(B,1,128,128) [= last cond frame], cond=(B,6,128,128),
        observation=(B,1,128,128) [model-space], operator (masking), noiser, n_steps=500,
        mc_times=25, guidance_scale=0.1.

Pseudocode:
  ts = linspace(0.0, 0.999, n_steps); dt = ts[1]-ts[0]; ones = ones(B)
  xt = base.requires_grad_(True)
  for i, ts_i in enumerate(ts):
      t = (ts_i * ones).to(device)
      bF = model(xt, t, None, cond=cond).requires_grad_(True)
      sigma = interpolant.sigma(t)
      es_x1 = taylor_est(model, xt, interpolant._wide(t), bF, cond, mc_times)  # list of 25
      norm_grad, norm = grad_and_value(xt, es_x1, observation, operator)
      mu = xt + bF * dt
      if norm_grad is None: norm_grad = 0
      xt = mu + sigma * randn_like(mu) * dt.sqrt() - guidance_scale * norm_grad
  return mu   # last-step drift mean (denoised)

Notes:
- The final returned value is mu (mean), not the noisy xt.
- guidance_scale multiplies and SUBTRACTS the gradient (moves toward lower residual).
- sqrt(dt) is the Brownian increment scale; sigma is the diffusion coefficient sigma(t)=1-t.

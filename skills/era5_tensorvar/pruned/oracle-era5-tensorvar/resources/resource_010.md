# Assimilation execution pseudocode

```python
np.random.seed(0)
torch.manual_seed(0)
assimilation_window = 5
total_steps = 5
sample_index = 0

raw = {k: v[0] for k, v in loaded_raw.items()}
obs_history = torch.tensor(raw['obs_history'], dtype=torch.float32, device=device)
B = np.eye(512) * 0.1
R = np.eye(512)
Q = np.eye(512) * 0.1
F = forward_model.C_forward.detach().cpu().numpy()
z_b_np = z_b.detach().cpu().numpy()

with torch.no_grad():
    obs_window = obs_history[0:5]
    inv_seq = inverse_model(obs_window)
    seq_z, encode_list = forward_model.encode(inv_seq, return_encode_list=True)
    qp_result = solve_latent_qp(z_b_np, seq_z.detach().cpu().numpy(), F, B, R, Q)
    z_star = torch.tensor(qp_result, dtype=torch.float32, device=device)
    decoded = forward_model.decode(z_star, encode_list)
    trajectory = decoded.detach().cpu().numpy()
```

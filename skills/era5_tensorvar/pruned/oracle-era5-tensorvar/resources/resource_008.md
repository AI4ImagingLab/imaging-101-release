# Feature-sequence construction

Pseudo-code:

```python
obs_history_t = torch.tensor(selected['obs_history'], dtype=torch.float32, device=device)
with torch.no_grad():
    inv_seq = inverse_model(obs_history_t)                         # (5,5,64,32)
    seq_z, encode_list = forward_model.encode(inv_seq, return_encode_list=True)
    seq_z_np = seq_z.detach().cpu().numpy()                        # (5,512)
```

Do not loop over time for the neural network unless the batch/time dimension is preserved exactly. The skip list should correspond to the same `inv_seq` batch that produced `seq_z`.

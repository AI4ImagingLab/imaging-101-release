# Checkpoint inventory and loading roles

Expected files under `public/visible_data/cases/case_000/input_data/checkpoints/`:

- `forward_model.pt`: state dict for the state encoder/decoder wrapper.
- `C_forward.pt`: latent dynamics matrix with shape `(512, 512)`, assigned as a tensor attribute of the forward wrapper.
- `inverse_model.pt`: state dict for the inverse-observation wrapper.
- `z_b.pt`: latent background mean vector with shape `(512,)`.

Model state dict names are intentionally matched by the architecture. Do not rename checkpoint keys. The inverse-observation wrapper contains frozen copies of the state encoder and decoder solely so the checkpoint loads with its expected keys.

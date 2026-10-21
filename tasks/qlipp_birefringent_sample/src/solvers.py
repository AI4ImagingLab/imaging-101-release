"""Calibrated four-state inversion with independent measured background."""
import torch
from .preprocessing import subtract_dark
from .physics_model import apply_matrix,inverse_mueller,estimate_parameters

def reconstruct(data,correct_background=True,black_level=100.58):
    """Return four float32 maps (1,position,z,y,x) on native detector grid.

    No smoothing, masking, clipping, empirical normalization, rotation or flip.
    Background light level defines relative transmission, not absolute power.
    """
    raw,bg=subtract_dark(data,black_level)
    analysis=torch.linalg.pinv(torch.as_tensor(data['instrument_matrix'][0],dtype=torch.float32))
    stokes=apply_matrix(analysis,torch.from_numpy(raw))
    if correct_background:
        background_stokes=apply_matrix(analysis,torch.from_numpy(bg))
        stokes=apply_matrix(inverse_mueller(background_stokes)[:,:,None,None],stokes)
    return {k:v.numpy()[None] for k,v in zip(['reconstruction','orientation','transmittance','depolarization'],estimate_parameters(stokes))}

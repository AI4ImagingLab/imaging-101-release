"""
End-to-End Variational Network (VarNet) Solver
================================================

Implements inference using a pretrained End-to-End Variational Network
for accelerated multi-coil MRI reconstruction.

VarNet is an unrolled optimization network with 12 cascades. Each cascade:
1. Estimates sensitivity maps from the input k-space (SensitivityModel)
2. Applies a U-Net regularizer in image domain
3. Enforces data consistency in k-space

The model jointly estimates coil sensitivities and reconstructs the image
end-to-end, trained with SSIM loss on the fastMRI dataset.

Reference
---------
Sriram et al., "End-to-End Variational Networks for Accelerated MRI
Reconstruction," MICCAI 2020.
"""

import os
import numpy as np
import torch
from fastmri.models.varnet import VarNet


def load_varnet(weights_path: str, device: str = "cpu") -> VarNet:
    """
    Load a pretrained VarNet model.

    Parameters
    ----------
    weights_path : str
        Path to state dict .pt file.
    device : str
        Device to load model on.

    Returns
    -------
    model : VarNet
        Pretrained model in eval mode.
    """
    model = VarNet(
        num_cascades=12, pools=4, chans=18,
        sens_pools=4, sens_chans=8,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def varnet_reconstruct(
    model: VarNet,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run VarNet inference on a single slice.

    Parameters
    ----------
    model : VarNet
        Pretrained model.
    masked_kspace : Tensor, (Nc, H, W, 2) float32
        Undersampled multi-coil k-space.
    mask : Tensor, boolean
        Undersampling mask.
    device : str

    Returns
    -------
    recon : ndarray, (H, W) float32
        Reconstructed magnitude image.
    """
    masked_kspace = masked_kspace.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(masked_kspace, mask, num_low_frequencies=None)

    return output.squeeze().cpu().numpy()


def varnet_reconstruct_batch(
    model: VarNet,
    kspace_slices: np.ndarray,
    acceleration: int = 4,
    center_fraction: float = 0.08,
    target_h: int = 320,
    target_w: int = 320,
    device: str = "cpu",
) -> tuple:
    """
    Run VarNet on multiple slices with undersampling and center cropping.

    Parameters
    ----------
    model : VarNet
    kspace_slices : (N, Nc, H, W) complex64
    acceleration : int
    center_fraction : float
    target_h, target_w : int
        Output image size (center crop).
    device : str

    Returns
    -------
    recons : (N, target_h, target_w) float32
    zerofills : (N, target_h, target_w) float32
    """
    from src.preprocessing import apply_mask
    from src.physics_model import zero_filled_recon, center_crop

    recons = []
    zerofills = []

    for i in range(kspace_slices.shape[0]):
        masked_ks, mask = apply_mask(
            kspace_slices[i], acceleration, center_fraction, seed=42,
        )

        recon = varnet_reconstruct(model, masked_ks, mask, device)
        recon = center_crop(recon, target_h, target_w)
        recons.append(recon)

        zf = zero_filled_recon(masked_ks)
        zf = center_crop(zf, target_h, target_w)
        zerofills.append(zf)

    return np.array(recons), np.array(zerofills)

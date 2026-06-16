"""
Data Preprocessing for FPM-INR
===============================

Handles loading raw FPM measurement data, computing optical parameters,
and preparing inputs for the physics model and solver.

Pipeline: raw_data.npz + ground_truth.npz + meta_data.json -> calibrated measurements + optical params
"""

import os
import json
import numpy as np
import torch


def load_raw_data(data_dir: str = "data") -> dict:
    """
    Load raw FPM measurement data from raw_data.npz.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing raw_data.npz.

    Returns
    -------
    dict with keys:
        'I_low'     : (M, N, n_leds) float32 - raw low-res measurements (batch dim squeezed)
        'na_calib'  : (n_leds, 2) float32 - calibrated NA per LED (batch dim squeezed)
        'mag'       : float32 - magnification
        'dpix_c'    : float32 - camera pixel pitch (um)
        'na_cal'    : float32 - objective NA
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))

    return {
        "I_low": raw["I_low"][0].astype("float32"),        # (1,M,N,L) -> (M,N,L)
        "na_calib": raw["na_calib"][0].astype("float32"),   # (1,L,2) -> (L,2)
        "mag": raw["mag"].astype("float32"),
        "dpix_c": raw["dpix_c"].astype("float32"),
        "na_cal": raw["na_cal"].astype("float32"),
    }


def load_ground_truth(data_dir: str = "data") -> dict:
    """
    Load ground truth z-stack from ground_truth.npz.

    Returns
    -------
    dict with keys:
        'I_stack' : (H, W, n_z) float32 - ground truth amplitude z-stack (batch dim squeezed)
        'zvec'    : (n_z,) float32 - z-positions in micrometers (batch dim squeezed)
    """
    gt = np.load(os.path.join(data_dir, "ground_truth.npz"))
    return {
        "I_stack": gt["I_stack"][0].astype("float32"),  # (1,H,W,Z) -> (H,W,Z)
        "zvec": gt["zvec"][0].astype("float32"),        # (1,Z) -> (Z,)
    }


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load meta_data.json file.

    Returns
    -------
    dict with all optical and training parameters.
    """
    meta_path = os.path.join(data_dir, "meta_data.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def compute_optical_params(raw_data: dict, metadata: dict) -> dict:
    """
    Compute derived optical parameters from raw data and metadata.

    Parameters
    ----------
    raw_data : dict from load_raw_data()
    metadata : dict from load_metadata()

    Returns
    -------
    dict with keys:
        'Fxx1'         : (NN,) float - spatial frequency x-coordinates
        'Fyy1'         : (MM,) float - spatial frequency y-coordinates
        'ledpos_true'  : (n_leds, 2) int - LED positions in frequency space
        'Isum'         : (M, N, n_leds) float32 - normalized, reordered measurements
        'order'        : (n_leds,) int - LED ordering by NA
        'u', 'v'       : (n_leds,) float - sorted illumination NA components
        'M', 'N'       : int - raw measurement dimensions
        'MM', 'NN'     : int - upsampled dimensions
        'k0'           : float - free-space k-vector
        'kmax'         : float - maximum k-value
        'D_pixel'      : float - pixel size at image plane
        'MAGimg'       : int - upsampling ratio
        'ID_len'       : int - number of LEDs
    """
    num_modes = metadata["num_modes"]
    MAGimg = metadata["upsample_ratio"]
    wavelength = metadata["wavelength_um"]

    I = raw_data["I_low"]
    # Select ROI (2*num_modes for 3D mode)
    I = I[0 : int(num_modes * 2), 0 : int(num_modes * 2), :]

    M = I.shape[0]
    N = I.shape[1]
    ID_len = I.shape[2]

    NAs = raw_data["na_calib"]
    NAx = NAs[:, 0]
    NAy = NAs[:, 1]

    k0 = 2 * np.pi / wavelength
    mag = raw_data["mag"]
    pixel_size = raw_data["dpix_c"]
    D_pixel = pixel_size / mag
    NA = raw_data["na_cal"]
    kmax = NA * k0

    MM = int(M * MAGimg)
    NN = int(N * MAGimg)

    # Define spatial frequency coordinates
    Fxx1, Fyy1 = np.meshgrid(
        np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2)
    )
    Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
    Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)

    # Calculate illumination NA and sort by distance
    u = -NAx
    v = -NAy
    NAillu = np.sqrt(u**2 + v**2)
    order = np.argsort(NAillu)
    u = u[order]
    v = v[order]

    # NA shift in pixel from different LED illuminations
    ledpos_true = np.zeros((ID_len, 2), dtype=int)
    for idx in range(ID_len):
        Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
        ledpos_true[idx, 0] = np.argmin(Fx1_temp)
        Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
        ledpos_true[idx, 1] = np.argmin(Fy1_temp)

    # Normalize and reorder measurements
    Isum = I[:, :, order] / np.max(I)

    return {
        "Fxx1": Fxx1,
        "Fyy1": Fyy1,
        "ledpos_true": ledpos_true,
        "Isum": Isum,
        "order": order,
        "u": u,
        "v": v,
        "M": M,
        "N": N,
        "MM": MM,
        "NN": NN,
        "k0": k0,
        "kmax": kmax,
        "D_pixel": D_pixel,
        "MAGimg": MAGimg,
        "ID_len": ID_len,
        "NA": NA,
        "wavelength": wavelength,
    }


def compute_pupil_and_propagation(optical_params: dict) -> dict:
    """
    Compute pupil function and angular spectrum propagation kernel.

    Parameters
    ----------
    optical_params : dict from compute_optical_params()

    Returns
    -------
    dict with keys:
        'Pupil0' : ndarray (M, N) - pupil support mask
        'kzz'    : ndarray (M, N) complex - z-component of k-vector
    """
    M = optical_params["M"]
    N = optical_params["N"]
    Fxx1 = optical_params["Fxx1"]
    k0 = optical_params["k0"]
    kmax = optical_params["kmax"]
    D_pixel = optical_params["D_pixel"]

    # Define angular spectrum
    kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])
    kxx, kyy = kxx - np.mean(kxx), kyy - np.mean(kyy)
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    kzz_ampli = mask_k * np.abs(
        np.sqrt((k0**2 - krr.astype("complex64") ** 2))
    )
    kzz_phase = np.angle(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)

    # Define pupil support
    Fx1, Fy1 = np.meshgrid(
        np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2)
    )
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1

    return {
        "Pupil0": Pupil0,
        "kzz": kzz,
    }


def compute_z_params(metadata: dict, optical_params: dict) -> dict:
    """
    Compute z-plane sampling parameters for 3D reconstruction.

    Returns
    -------
    dict with keys:
        'DOF'      : float - depth of field
        'delta_z'  : float - z-slice separation
        'num_z'    : int - number of z-slices for training
        'z_min'    : float
        'z_max'    : float
    """
    NA = optical_params["NA"]
    z_min = metadata["z_min_um"]
    z_max = metadata["z_max_um"]

    DOF = 0.5 / NA**2
    delta_z = 0.8 * DOF
    num_z = int(np.ceil((z_max - z_min) / delta_z))

    return {
        "DOF": float(DOF),
        "delta_z": float(delta_z),
        "num_z": num_z,
        "z_min": z_min,
        "z_max": z_max,
    }


def prepare_data(data_dir: str = "data", device: str = "cuda:0") -> dict:
    """
    Full data preparation pipeline.

    Parameters
    ----------
    data_dir : str
        Path to data directory.
    device : str
        Torch device.

    Returns
    -------
    dict with keys:
        'metadata'       : dict - raw metadata from JSON
        'optical_params' : dict - computed optical parameters
        'pupil_data'     : dict - Pupil0 and kzz as torch tensors on device
        'z_params'       : dict - z-sampling parameters
        'Isum'           : torch.Tensor on device - normalized measurements
    """
    metadata = load_metadata(data_dir)
    raw_data = load_raw_data(data_dir)
    optical_params = compute_optical_params(raw_data, metadata)
    pupil_data_np = compute_pupil_and_propagation(optical_params)
    z_params = compute_z_params(metadata, optical_params)

    # Convert to torch tensors on device
    Pupil0 = (
        torch.from_numpy(pupil_data_np["Pupil0"])
        .view(1, 1, optical_params["M"], optical_params["N"])
        .to(device)
    )
    kzz = torch.from_numpy(pupil_data_np["kzz"]).to(device).unsqueeze(0)
    Isum = torch.from_numpy(optical_params["Isum"]).to(device)

    pupil_data = {
        "Pupil0": Pupil0,
        "kzz": kzz,
    }

    return {
        "metadata": metadata,
        "optical_params": optical_params,
        "pupil_data": pupil_data,
        "z_params": z_params,
        "Isum": Isum,
    }

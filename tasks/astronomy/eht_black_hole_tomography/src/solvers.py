"""
BH-NeRF solver: NeRF model, training loop, and loss functions.

Ported from:
- bhnerf/network.py (MLP, posenc, NeRF_Predictor, loss_fn_image, loss_fn_eht)
- bhnerf/optimization.py (Optimizer, TrainStep)

Reference: Levis et al. "Gravitationally Lensed Black Hole Emission Tomography" (CVPR 2022)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.physics_model import (
    velocity_warp_coords, fill_unsupervised, trilinear_interpolate,
    volume_render, rotation_matrix_torch
)


# ---------------------------------------------------------------------------
# Positional encoding
# Adapted from bhnerf/network.py posenc()
# ---------------------------------------------------------------------------

def positional_encoding(x, deg=3):
    """
    Concatenate x with a positional encoding of x with degree deg.

    Uses cos(x) = sin(x + pi/2) for a single vectorized call.

    Parameters
    ----------
    x : torch.Tensor, shape (*, D)
        Input coordinates.
    deg : int
        Number of frequency bands (L in the paper).

    Returns
    -------
    encoded : torch.Tensor, shape (*, D + 2*D*deg)
    """
    if deg == 0:
        return x
    scales = 2.0 ** torch.arange(deg, dtype=x.dtype, device=x.device)
    # x: (*, D), scales: (deg,) -> xb: (*, D*deg)
    xb = (x[..., None, :] * scales[:, None]).reshape(
        list(x.shape[:-1]) + [-1]
    )
    four_feat = torch.sin(
        torch.cat([xb, xb + 0.5 * np.pi], dim=-1)
    )
    return torch.cat([x, four_feat], dim=-1)


# ---------------------------------------------------------------------------
# MLP
# Adapted from bhnerf/network.py MLP class
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Multi-layer perceptron with optional skip connections.

    Parameters
    ----------
    in_features : int
        Input dimension.
    net_depth : int
        Number of hidden layers.
    net_width : int
        Hidden layer width.
    out_channel : int
        Output dimension.
    do_skip : bool
        If True, add skip connection at layer net_depth // 2.
    """

    def __init__(self, in_features, net_depth=4, net_width=128,
                 out_channel=1, do_skip=True):
        super().__init__()
        self.net_depth = net_depth
        self.do_skip = do_skip

        if do_skip:
            self.skip_layer = net_depth // 2

        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = in_features
            elif do_skip and i == self.skip_layer:
                dim_in = net_width + in_features
            else:
                dim_in = net_width
            layer = nn.Linear(dim_in, net_width)
            # He uniform init
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        final_in = net_width
        self.output_layer = nn.Linear(final_in, out_channel)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (*, in_features)

        Returns
        -------
        out : torch.Tensor, shape (*, out_channel)
        """
        inputs = x
        for i, layer in enumerate(self.layers):
            if self.do_skip and i == self.skip_layer and i > 0:
                x = torch.cat([x, inputs], dim=-1)
            x = F.relu(layer(x))
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# BH-NeRF Model
# Adapted from bhnerf/network.py NeRF_Predictor
# ---------------------------------------------------------------------------

class BHNeRFModel(nn.Module):
    """
    BH-NeRF: Neural radiance field for black hole emission with Keplerian dynamics.

    Given ray coordinates and a time frame:
    1. Apply velocity warp (Keplerian shearing)
    2. Scale coordinates and apply positional encoding
    3. Pass through MLP to get emission
    4. Apply sigmoid with offset (sigmoid(x - 10))
    5. Zero out unsupervised region

    Parameters
    ----------
    scale : float
        Domain scale for normalizing NN inputs.
    rmin : float
        Minimum radius for emission recovery.
    rmax : float
        Maximum radius for emission recovery.
    z_width : float
        Maximum |z| for emission.
    posenc_deg : int
        Positional encoding degree.
    net_depth : int
        MLP depth.
    net_width : int
        MLP width.
    """

    def __init__(self, scale, rmin=0.0, rmax=float('inf'), z_width=float('inf'),
                 posenc_deg=3, net_depth=4, net_width=128):
        super().__init__()
        self.scale = scale
        self.rmin = rmin
        self.rmax = rmax
        self.z_width = z_width
        self.posenc_deg = posenc_deg

        in_features = 3 + 2 * 3 * posenc_deg  # 3D coords + positional encoding
        self.mlp = MLP(in_features, net_depth, net_width, out_channel=1)

    def forward(self, t_frame, coords, Omega, t_start_obs, t_geo,
                t_injection, rot_axis):
        """
        Predict emission on ray coordinates for one time frame.

        Parameters
        ----------
        t_frame : float
            Observation time.
        coords : torch.Tensor, shape (3, *spatial)
            Ray coordinates [x, y, z].
        Omega : torch.Tensor, shape (*spatial)
            Angular velocity at each point.
        t_start_obs : float
            Start time of observations.
        t_geo : torch.Tensor, shape (*spatial)
            Coordinate time along rays.
        t_injection : float
            Hotspot injection time.
        rot_axis : torch.Tensor, shape (3,)
            Rotation axis.

        Returns
        -------
        emission : torch.Tensor, shape (*spatial)
        """
        # Velocity warp
        warped_coords = velocity_warp_coords(
            coords, Omega, t_frame, t_start_obs, t_geo, t_injection,
            rot_axis=rot_axis
        )

        # Mask pre-injection coordinates
        valid_mask = torch.isfinite(warped_coords).all(dim=-1)
        net_input = torch.where(
            valid_mask.unsqueeze(-1), warped_coords,
            torch.zeros_like(warped_coords)
        )

        # Scale and encode
        net_input = net_input / self.scale
        net_input = positional_encoding(net_input, self.posenc_deg)

        # MLP prediction
        net_output = self.mlp(net_input)
        emission = torch.sigmoid(net_output[..., 0] - 10.0)

        # Zero outside valid region
        emission = fill_unsupervised(emission, coords, self.rmin, self.rmax,
                                     self.z_width)

        # Zero pre-injection
        emission = emission * valid_mask.float()

        return emission


# ---------------------------------------------------------------------------
# Loss functions
# Adapted from bhnerf/network.py loss_fn_image(), loss_fn_eht()
# ---------------------------------------------------------------------------

def loss_fn_image(pred_images, target_images, sigma=1.0):
    """
    L2 loss on image plane.

    L = sum(|pred - target|^2 / sigma^2)

    Parameters
    ----------
    pred_images : torch.Tensor, shape (*, H, W)
    target_images : torch.Tensor, shape (*, H, W)
    sigma : float

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    return ((pred_images - target_images) / sigma).pow(2).sum()


def loss_fn_lightcurve(pred_images, target_lightcurve, sigma=1.0):
    """
    L2 loss on lightcurves (total flux).

    Parameters
    ----------
    pred_images : torch.Tensor, shape (n_frames, H, W)
    target_lightcurve : torch.Tensor, shape (n_frames,)
    sigma : float

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    pred_lc = pred_images.sum(dim=(-1, -2))
    return ((pred_lc - target_lightcurve) / sigma).pow(2).sum()


def loss_fn_visibility(pred_vis, target_vis, sigma):
    """
    Chi-squared loss on complex visibilities.

    Parameters
    ----------
    pred_vis : torch.Tensor, complex, shape (n_frames, M)
    target_vis : torch.Tensor, complex, shape (n_frames, M)
    sigma : torch.Tensor, shape (n_frames, M)

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    return (torch.abs(pred_vis - target_vis) / sigma).pow(2).sum()


# ---------------------------------------------------------------------------
# BH-NeRF Solver
# Adapted from bhnerf/optimization.py Optimizer
# ---------------------------------------------------------------------------

class BHNeRFSolver:
    """
    Full solver for BH-NeRF training and inference.

    Parameters
    ----------
    metadata : dict
        Training hyperparameters from meta_data.
    device : str or torch.device
        Compute device.
    """

    def __init__(self, metadata, device=None):
        self.metadata = metadata
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available()
                                       else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.rot_axis = None
        self._is_trained = False

    def reconstruct(self, obs_data, seed=42):
        """
        Train BH-NeRF model on observation data.

        Parameters
        ----------
        obs_data : dict
            From load_observation().
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        result : dict with keys:
            'model_state_dict' : trained model weights
            'loss_history' : list of per-iteration losses
            'rot_axis' : np.ndarray, shape (3,) — recovered rotation axis
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        meta = self.metadata
        fov_M = obs_data['fov_M']
        t_frames = obs_data['t_frames']
        t_start_obs = obs_data['t_start_obs']
        t_injection = obs_data['t_injection']
        n_iters = meta['n_iters']
        batch_size = meta['batch_size']

        # Build model
        scale = fov_M / 2.0
        model = BHNeRFModel(
            scale=scale,
            rmin=meta['rmin_M'],
            rmax=meta['rmax_M'],
            z_width=meta['z_width_M'],
            posenc_deg=meta['posenc_deg'],
            net_depth=meta['net_depth'],
            net_width=meta['net_width'],
        ).to(self.device)

        # Fixed rotation axis (z-axis), following the original bhnerf convention.
        # The original code does NOT optimize the rotation axis.
        rot_axis_fixed = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32,
                                      device=self.device)
        max_grad_norm = 1.0

        # Move data to device
        ray_coords = obs_data['ray_coords'].to(self.device)
        Omega = obs_data['Omega'].to(self.device)
        g_doppler = obs_data['g_doppler'].to(self.device)
        dtau = obs_data['dtau'].to(self.device)
        Sigma = obs_data['Sigma'].to(self.device)
        t_geo = obs_data['t_geo'].to(self.device)
        target_images = obs_data['images'].to(self.device)

        # Optimizer with polynomial LR schedule
        lr_init = meta['lr_init']
        lr_final = meta['lr_final']
        all_params = list(model.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr_init)

        def lr_lambda(step):
            t = min(step / max(n_iters, 1), 1.0)
            return (1 - t) * 1.0 + t * (lr_final / lr_init)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        n_frames = len(t_frames)
        loss_history = []

        # Training loop
        for step in tqdm(range(n_iters), desc="Training BH-NeRF"):
            optimizer.zero_grad()

            # Sample batch of time frames
            batch_idx = np.random.choice(n_frames, min(batch_size, n_frames),
                                         replace=False)

            total_loss = torch.tensor(0.0, device=self.device)

            for idx in batch_idx:
                t = float(t_frames[idx])

                # Forward: predict emission on rays
                emission = model(t, ray_coords, Omega, t_start_obs, t_geo,
                                 t_injection, rot_axis_fixed)

                # Volume render
                pred_image = volume_render(emission, g_doppler, dtau, Sigma)

                # Loss
                loss = loss_fn_image(pred_image, target_images[idx])
                total_loss = total_loss + loss

            total_loss = total_loss / len(batch_idx)
            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            optimizer.step()
            scheduler.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)

        self.model = model
        self.rot_axis = np.array([0.0, 0.0, 1.0])
        self._obs_data = obs_data
        self._is_trained = True

        return {
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'rot_axis': self.rot_axis.copy(),
        }

    @torch.no_grad()
    def predict_emission_3d(self, fov_M=None, resolution=64):
        """
        Sample trained NeRF on a regular 3D grid.

        Parameters
        ----------
        fov_M : float
            Field of view. Uses metadata value if None.
        resolution : int
            Grid resolution.

        Returns
        -------
        emission : np.ndarray, shape (resolution, resolution, resolution)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call reconstruct() first.")

        if fov_M is None:
            fov_M = self.metadata['fov_M']

        self.model.eval()
        grid_1d = np.linspace(-fov_M / 2, fov_M / 2, resolution)
        xx, yy, zz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        coords = torch.tensor(
            np.stack([xx, yy, zz]),
            dtype=torch.float32, device=self.device
        )

        Omega_grid = torch.zeros(resolution, resolution, resolution,
                                 device=self.device)
        t_geo_grid = torch.zeros_like(Omega_grid)
        rot_axis = torch.tensor(self.rot_axis, dtype=torch.float32,
                                device=self.device)

        emission = self.model(
            0.0, coords, Omega_grid, 0.0, t_geo_grid, 0.0, rot_axis
        )
        return emission.cpu().numpy()

    @torch.no_grad()
    def predict_movie(self, obs_data=None):
        """
        Generate image-plane movie from trained model.

        Parameters
        ----------
        obs_data : dict, optional
            Observation data. Uses stored data if None.

        Returns
        -------
        images : np.ndarray, shape (n_frames, num_alpha, num_beta)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call reconstruct() first.")

        if obs_data is None:
            obs_data = self._obs_data

        self.model.eval()
        ray_coords = obs_data['ray_coords'].to(self.device)
        Omega = obs_data['Omega'].to(self.device)
        g_doppler = obs_data['g_doppler'].to(self.device)
        dtau = obs_data['dtau'].to(self.device)
        Sigma_val = obs_data['Sigma'].to(self.device)
        t_geo = obs_data['t_geo'].to(self.device)
        t_frames = obs_data['t_frames']
        t_start_obs = obs_data['t_start_obs']
        t_injection = obs_data['t_injection']
        rot_axis = torch.tensor(self.rot_axis, dtype=torch.float32,
                                device=self.device)

        images = []
        for t in t_frames:
            emission = self.model(float(t), ray_coords, Omega, t_start_obs,
                                  t_geo, t_injection, rot_axis)
            img = volume_render(emission, g_doppler, dtau, Sigma_val)
            images.append(img.cpu().numpy())

        return np.stack(images)

"""Linear quasi-static volume-conductor calibration in SI units."""
import numpy as np
def projection(channels=64):return np.eye(channels)-np.ones((channels,channels))/channels
def forward(currents,lead_field):
    # Current dipole moment (node,xyz,time), A m; lead field V/(A m).
    return projection(lead_field.shape[0])@lead_field@currents.reshape(-1,currents.shape[-1])
def adjoint(potentials,lead_field):
    values=lead_field.T@projection(lead_field.shape[0])@potentials
    return values.reshape(-1,3,potentials.shape[-1])

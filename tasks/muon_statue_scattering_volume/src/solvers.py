"""Qualified published POCA geometric reconstruction, explicit local stages."""
import numpy as np
from .physics_model import fit_tracks,scattering_angles,nonparallel_mask,closest_approach,in_volume,assign_voxels,aggregate_rms

def reconstruct(hits, return_stages=False, correct_direction=True, correct_angle=True,
                correct_mask=True, angle_quantile=.999, prediction_quantile=.999):
    """Reconstruct (N,6,3) hits mm into float32(36,80,40) RMS radians.

    Defaults implement the qualified geometric method. Correction switches exist
    only to compute the original source and three controlled notebook ablations.
    No truth, checkpoint, source imports or unknown-dependent calibration is used.
    """
    if hits.ndim != 3 or hits.shape[1:] != (6,3) or hits.dtype != np.float32:
        raise ValueError('Expected float32(N,6,3) detector events')
    u,p = fit_tracks(hits[:,:3].transpose(2,1,0), corrected=correct_direction)
    v,q = fit_tracks(hits[:,3:].transpose(2,1,0), corrected=correct_direction)
    angles = scattering_angles(u,v,corrected=correct_angle)
    mask = nonparallel_mask(u,v,corrected=correct_mask)
    poca = closest_approach(p[mask],q[mask],u[mask],v[mask])
    volume_mask = in_volume(poca)
    poca = poca[volume_mask]
    retained = angles[mask][volume_mask]
    indices = assign_voxels(poca)
    image,counts = aggregate_rms(indices,retained,angle_quantile,prediction_quantile)
    if return_stages:
        return image,dict(tracks_in=u,tracks_out=v,points_in=p,points_out=q,
                          angles=angles,parallel_mask=mask,volume_mask=volume_mask,
                          pocas=poca,retained_angles=retained,indices=indices,counts=counts)
    return image

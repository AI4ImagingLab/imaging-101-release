"""Brooks-Humphreys DAMAS projected Gauss-Seidel, Acoular BSD-3-Clause."""
import numpy as np
from .preprocessing import calibrate,cross_spectral_matrix
from .physics_model import grid_positions,dirty_map,point_spread,SAMPLE_FREQUENCY,SOUND_SPEED
def damas(psf,dirty,iterations=100):
    """Initialize x=dirty; update in grid order, relaxing with factor one."""
    x=dirty.copy()
    for _ in range(iterations):
        for i in range(len(x)):
            other=psf[i]@x-psf[i,i]*x[i];x[i]=max(0.,dirty[i]-other)
    return x
def solve(data,iterations=100,sound_speed=SOUND_SPEED,remove_diagonal=True):
    y,valid=calibrate(data['microphone_signal'],data['calibration_factor']);mics=data['microphone_position'][valid];csm=cross_spectral_matrix(y);freq=np.fft.rfftfreq(128,1/SAMPLE_FREQUENCY);grid=grid_positions()
    ids=np.flatnonzero((freq>=4000/2**(1/6))&(freq<4000*2**(1/6)));clean=np.zeros(len(grid));dirty=np.zeros(len(grid))
    for j in ids:
        b=dirty_map(csm[j],freq[j],mics,grid,sound_speed,remove_diagonal);a=point_spread(freq[j],mics,grid,sound_speed);clean+=damas(a,b,iterations);dirty+=b
    return clean.reshape(13,13),dirty.reshape(13,13)
def reconstruct(data):
    clean,_=solve(data);return {'reconstruction':clean[None,:]}

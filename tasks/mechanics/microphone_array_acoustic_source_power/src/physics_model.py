"""Free-field monopole propagation and true-level PSF, Acoular BSD-3-Clause."""
import numpy as np
SOUND_SPEED=346.04
SAMPLE_FREQUENCY=51200.
def grid_positions(increment=.05):
    x=np.linspace(-.6,0,round(.6/increment)+1);y=np.linspace(-.3,.3,round(.6/increment)+1)
    xx,yy=np.meshgrid(x,y,indexing='ij');return np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,-.68)])
def geometry(microphones,grid):
    return np.linalg.norm(grid,axis=1),np.linalg.norm(grid[:,None,:]-microphones[None,:,:],axis=2)
def transfer(frequency,microphones,grid,sound_speed=SOUND_SPEED):
    r0,rm=geometry(microphones,grid);phase=(2*np.pi*frequency/sound_speed*(rm-r0[:,None])).astype(np.float32)
    return (np.cos(phase.astype(np.float64)).astype(np.float32).astype(np.float64)-1j*np.sin(phase.astype(np.float64)).astype(np.float32).astype(np.float64))*r0[:,None]/rm
def steering(frequency,microphones,grid,sound_speed=SOUND_SPEED):
    h=transfer(frequency,microphones,grid,sound_speed);return h/np.sum(h*h.conj(),axis=1)[:,None]
def dirty_map(csm,frequency,microphones,grid,sound_speed=SOUND_SPEED,remove_diagonal=True):
    r0,rm=geometry(microphones,grid);phase=(2*np.pi*frequency/sound_speed*rm).astype(np.float32);v=(np.cos(phase.astype(np.float64)).astype(np.float32).astype(np.float64)-1j*np.sin(phase.astype(np.float64)).astype(np.float32).astype(np.float64))/rm
    c=csm.copy();m=len(microphones)
    if remove_diagonal:np.fill_diagonal(c,0)
    out=np.einsum('gm,mn,gn->g',v.conj(),c,v).real/(r0*np.sum(1/rm**2,axis=1))**2
    if remove_diagonal:out*=m/(m-1)
    return np.maximum(out,0)
def point_spread(frequency,microphones,grid,sound_speed=SOUND_SPEED):
    r0,rm=geometry(microphones,grid);k=2*np.pi*frequency/sound_speed;out=np.empty((len(grid),len(grid)))
    for i in range(len(grid)):
        phase=(k*(rm[i,None,:]-rm)).astype(np.float32)
        s=np.sum((np.cos(phase.astype(np.float64)).astype(np.float32).astype(np.float64)-1j*np.sin(phase.astype(np.float64)).astype(np.float32).astype(np.float64))/rm/rm[i,None,:],axis=1)
        norm=r0/r0[i]/np.sum(1/rm[i]**2);out[i]=abs(s)**2*norm**2
    return out

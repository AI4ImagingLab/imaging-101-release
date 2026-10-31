"""Independently derived complex Fourier-state Bloch propagation, milliseconds."""
import numpy as np

def rf_matrix(angle,phase):
    """Right-handed physical Rodrigues rotation in (F+, F-, Z) coordinates."""
    axis=np.array([np.cos(phase),np.sin(phase),0.])
    cross=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    R=np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross
    C=np.array([[1,1j,0],[1,-1j,0],[0,0,1]],complex)
    return C@R@np.linalg.inv(C)

def epg_signal(t1_ms,t2_ms,flip_scale,magnitude=True):
    """(B,32) signal for unit density; exact 257-state source discretization."""
    t2=np.atleast_1d(t2_ms).astype(float);state=np.zeros((len(t2),3,257),complex);zero=128
    state[:,2,zero]=1-np.exp(-1000./t1_ms)
    state=np.einsum('ij,bjk->bik',rf_matrix(flip_scale*np.pi/2,0.),state)
    ref=rf_matrix(flip_scale*np.pi,np.pi/2)
    e1=np.exp(-4.6/t1_ms);e2=np.exp(-4.6/t2)
    out=np.empty((len(t2),32),complex)
    for echo in range(32):
        state[:,:2]*=e2[:,None,None];state[:,2]*=e1;state[:,2,zero]+=1-e1
        state[:,0]=np.roll(state[:,0],2,axis=-1);state[:,1]=np.roll(state[:,1],-2,axis=-1)
        state=np.einsum('ij,bjk->bik',ref,state)
        state[:,0]=np.roll(state[:,0],2,axis=-1);state[:,1]=np.roll(state[:,1],-2,axis=-1)
        state[:,:2]*=e2[:,None,None];state[:,2]*=e1;state[:,2,zero]+=1-e1
        out[:,echo]=state[:,0,zero]
    return abs(out) if magnitude else out

def build_dictionary(centroids,t2_ms):
    """Return (cluster,echo,bin), with unit-density column amplitudes."""
    return np.stack([epg_signal(t1,t2_ms,k).T for t1,k in centroids])

def predict(dictionary,coefficients,labels):
    """(voxel,bin) masses to complete (voxel,echo) observation curves."""
    out=np.empty((len(coefficients),32))
    for c in range(len(dictionary)):
        m=labels==c;out[m]=coefficients[m]@dictionary[c].T
    return out

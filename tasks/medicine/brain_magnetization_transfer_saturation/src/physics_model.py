"""Small-angle steady-state FLASH model; angles radians, times seconds."""
import numpy as np
ANGLES=np.array([6.,20.,6.])*(np.pi/180)
TR=np.array([.028,.018,.028])
def flash_signal(amplitude,r1,delta,angle,tr):
    """Approximate signal A*a*TR*R1/(TR*R1+a²/2+delta); delta is a fraction."""
    return amplitude*angle*tr*r1/(tr*r1+angle**2/2+delta)
def exact_flash_signal(amplitude,r1,angle,tr):
    """Independent no-MT Bloch steady state, used only to assess approximation."""
    e=np.exp(-tr*r1)
    return amplitude*np.sin(angle)*(1-e)/(1-np.cos(angle)*e)
def predict_from_parameters(data,mtsat,r1,mask):
    """Infer nuisance amplitude from PDw, predict three measurements on mask."""
    r=r1[mask];a=ANGLES[0];tr=TR[0]
    amplitude=data['PDw'][mask]*(tr*r+a*a/2)/(a*tr*r)
    predicted=np.stack([flash_signal(amplitude,r,delta,ang,rep) for delta,ang,rep in zip((0.,0.,mtsat[mask]/100),ANGLES,TR)])
    return amplitude,predicted

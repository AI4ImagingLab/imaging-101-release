"""Explicit travel-time, clock and displacement equations and derivatives.

Published mathematics: Zhang et al., arXiv2403.05791v1, Equations5/6/9/13.
Author code Hybrid-TDOA-Calib@150df084, MIT attribution in NOTICE.
No upstream or acoustic-domain library is imported.
"""
import numpy as np
SOUND_SPEED=340.
MICROPHONES=10
EVENTS=14

def free_indices(microphones=MICROPHONES,events=EVENTS):
    """C-order state coordinates; first offset and six spatial gauges fixed."""
    ids=[0,1,2,4]
    ids.extend(range(5,5*microphones))
    ids.extend([5*(microphones+1),5*(microphones+2),5*(microphones+2)+1])
    for j in range(3,events):ids.extend(range(5*(microphones+j),5*(microphones+j)+3))
    return np.asarray(ids,dtype=np.int64)

def pack(state):
    return np.asarray(state).ravel()[free_indices()]

def unpack(parameters):
    state=np.zeros((MICROPHONES+EVENTS,5),dtype=np.float64)
    state.ravel()[free_indices()]=parameters
    return state

def predict_measurements(state,intervals,speed=SOUND_SPEED):
    """Return successive(K-1,M), inter-sensor(M-1,K), odometry(K-1,3).

    State rows are sensors then source events, columns xyz(m),offset(s),drift.
    Offsets/drifts of source rows are unused computational padding.
    """
    k=len(intervals);m=len(state)-k;s=state[m:,:3];sensors=state[:m,:3]
    distance=np.linalg.norm(sensors[:,None,:]-s[None,:,:],axis=2)
    successive=((distance[:,1:]-distance[:,:-1])/speed+(1+state[:m,4,None])*intervals[None,1:]).T
    time=np.cumsum(intervals)
    between=(distance[1:]-distance[0])/speed+state[1:m,3,None]+(state[1:m,4,None]-state[0,4])*time[None,:]
    return successive,between,np.diff(s,axis=0)

def residual_jacobian(state,data):
    """All295 residuals and295x85 analytic Jacobian in the native ordering."""
    m,k=MICROPHONES,EVENTS;intervals=data['emission_intervals'];time=np.cumsum(intervals)
    delta=state[:m,None,:3]-state[None,m:,:3];distance=np.linalg.norm(delta,axis=2)
    if np.any(distance<=0):raise ValueError('Coincident source/sensor makes distance derivative undefined')
    unit=delta/distance[:,:,None]
    successive,between,displacement=predict_measurements(state,intervals)
    residual=np.r_[(successive-data['tdoa_successive']).T.ravel(),(between-data['tdoa_between']).T.ravel(),(displacement-data['displacement']).ravel()]
    full=np.zeros((len(residual),5*(m+k)),dtype=np.float64)
    for i in range(m):
        for j in range(k-1):
            row=i*(k-1)+j
            full[row,5*i:5*i+3]=(unit[i,j+1]-unit[i,j])/SOUND_SPEED
            full[row,5*i+4]=intervals[j+1]
            full[row,5*(m+j):5*(m+j)+3]=unit[i,j]/SOUND_SPEED
            full[row,5*(m+j+1):5*(m+j+1)+3]=-unit[i,j+1]/SOUND_SPEED
    start=m*(k-1)
    for j in range(k):
        for i in range(1,m):
            row=start+j*(m-1)+i-1
            full[row,:3]=-unit[0,j]/SOUND_SPEED
            full[row,4]=-time[j]
            full[row,5*i:5*i+3]=unit[i,j]/SOUND_SPEED
            full[row,5*i+3]=1;full[row,5*i+4]=time[j]
            full[row,5*(m+j):5*(m+j)+3]=(unit[0,j]-unit[i,j])/SOUND_SPEED
    start+=k*(m-1)
    for j in range(k-1):
        full[start+3*j:start+3*j+3,5*(m+j):5*(m+j)+3]=-np.eye(3)
        full[start+3*j:start+3*j+3,5*(m+j+1):5*(m+j+1)+3]=np.eye(3)
    return residual,full[:,free_indices()]

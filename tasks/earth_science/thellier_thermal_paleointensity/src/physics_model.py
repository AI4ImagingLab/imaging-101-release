"""Explicit Thellier thermoremanent reciprocity and directional diagnostics."""
import numpy as np

def forward_arai(acquired_trm,field,lab_field,intercept):
    """Residual NRM = intercept - (ancient field / lab field) * gained TRM."""
    return intercept-(field/lab_field)*np.asarray(acquired_trm)

def principal_direction(vectors,anchored=True):
    """Explicit second-moment eigenproblem; sign follows initial magnetization."""
    vectors=np.asarray(vectors,dtype=np.float64)
    centered=vectors if anchored else vectors-np.mean(vectors,axis=0)
    values,basis=np.linalg.eigh(centered.T@centered)
    direction=basis[:,-1]
    if np.dot(direction,vectors[0])<0:direction=-direction
    # The Gram matrix is positive semidefinite; remove only negative roundoff
    # in an exactly rank-one diagnostic case (no effect on the bound data).
    mad=np.degrees(np.arctan(np.sqrt(max(0.,float(np.sum(values[:2])))/values[-1])))
    return direction,float(mad),values[::-1]

def vector_fraction(vectors,start,end):
    """FRAC = interval vector-difference sum / full vector-difference sum."""
    delta=np.linalg.norm(np.diff(vectors,axis=0),axis=1)
    total=np.sum(delta)+np.linalg.norm(vectors[-1])
    return float(np.sum(delta[start:end])/total)

def scatter_box(x,y,slope,checks,lo,hi,beta_threshold=.1):
    """Published SCAT box on Arai, repeat pTRM and tail measurements."""
    mx=np.mean(x);my=np.mean(y);err=abs(slope)*beta_threshold
    s1=slope+2*err;s2=slope-2*err;c1=my-s1*mx;c2=my-s2*mx
    x1=-c1/s1;x2=-c2/s2
    low=np.array([c1,-c1/x2]);high=np.array([c2,-c2/x1])
    groups=[np.column_stack([x,y])]
    for check in checks.values():
        selected=(check['temperature']>=lo)&(check['temperature']<=hi)&(check['starting']>=lo)&(check['starting']<=hi)
        groups.append(check['points'][selected])
    points=np.concatenate(groups)
    px=points[:,0];py=points[:,1]
    passed=bool(np.all((px>=0)&(py>=0)&(px<=x1)&(py<=c2)&(py>=low[0]+low[1]*px)&(py<=high[0]+high[1]*px)))
    return passed,low,high,points

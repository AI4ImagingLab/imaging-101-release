"""Test stochastic initializations by support and distribution, not a seed."""
import numpy as np
from src.solvers import initial_parameters

def test_initializer_support_and_uniform_moments():
    rng=np.random.default_rng()
    count=4096
    q=np.array([initial_parameters(rng) for _ in range(count)])
    assert q.shape==(count,12) and np.isfinite(q).all() and (q>0).all()
    assert ((q[:,0]>=15)&(q[:,0]<=250)).all()
    assert ((q[:,1]>=100)&(q[:,1]<=10000)).all()
    assert ((q[:,2]>=.00666666)&(q[:,2]<=.033333)).all()
    assert (np.diff(q[:,3:6],axis=1)<=0).all()
    assert ((q[:,3:7]>=10)&(q[:,3:7]<=50000)).all()
    assert (q[:,7:11]<=200).all()
    assert ((q[:,11]>=0)&(q[:,11]<=500)).all()
    totals=q[:,7:11].sum(axis=1)
    assert ((totals>=100-1e-10)&(totals<=400+1e-10)).all()
    # Source draws these scalar marginals uniformly. Eight standard errors
    # provide a predeclared high-confidence statistical control, not exact RNG
    # equality. Rejection of individual a>200 does not redraw the total.
    for values,lo,hi in [(q[:,0],15,250),(q[:,1],100,10000),
                         (1/q[:,2],1/.033333,1/.00666666),
                         (q[:,11],0,500),(totals,100,400)]:
        expected=(lo+hi)/2
        standard_error=(hi-lo)/np.sqrt(12*count)
        assert abs(values.mean()-expected)<=8*standard_error

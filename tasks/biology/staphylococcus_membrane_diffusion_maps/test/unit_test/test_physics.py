import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from src.physics_model import correlation_model
from src.preprocessing import bin_labels,correlate_trace

def test_forward_formula_against_independent_overlap_integral():
    a=.32;sigma=.19;f=2.1;D=.8;N=35.;offset=-.00003
    for tau in (.0001,.01,.5,4.):
        v=sigma*sigma+f*D*tau
        overlap=quad(lambda s:(a-abs(s))*np.exp(-s*s/(4*v))/np.sqrt(4*np.pi*v),-a,a,epsabs=1e-13)[0]/a
        assert_allclose(correlation_model(np.array([tau]),N,D,offset)[0],overlap**2/N+offset,rtol=1e-9,atol=1e-13)

def test_scaling_offset_and_identifiability():
    t=np.geomspace(.001,10,20)
    base=correlation_model(t,30.,.7,0.)
    assert_allclose(correlation_model(t,60.,.7,.01),base/2+.01,rtol=1e-13)
    assert_allclose(correlation_model(t,30.,.7*2.1,0.,curvature=1.),base,rtol=1e-13)
    assert_allclose(correlation_model(t,30.,0.,0.),np.full(t.size,correlation_model(t[:1],30.,0.,0.)[0]))

def test_correlation_first_register_independent_direct_sum():
    trace=np.arange(1001,dtype=float)%17+100.;curve=correlate_trace(trace,dt=.003)
    centered=trace-trace.mean()
    expected=[np.sum(centered[:-k]*centered[k:])/(trace.mean()**2*(len(trace)-k)) for k in range(1,17)]
    assert_allclose(curve[:16,1],expected,rtol=1e-14,atol=1e-14)
    assert_allclose(curve[:16,0],np.arange(1,17)*.003)

def test_coarsened_register_uses_pair_means_and_original_mean():
    trace=np.arange(1001,dtype=float)%17+100.;centered=trace-trace.mean();paired=(centered[:1000:2]+centered[1:1000:2])/2
    curve=correlate_trace(trace,dt=.003)
    assert_allclose(curve[16,1],np.sum(paired[:-9]*paired[9:])/(trace.mean()**2*(500-9)),rtol=1e-14)
    assert curve[16,0]==.003*18

def test_label_binning_no_mixed_cells_and_partial_background():
    labels=np.array([[0,2,1,3],[2,2,1,1],[0,0,4,4],[0,0,4,0]],dtype=np.uint8)
    assert np.array_equal(bin_labels(labels,2),[[2,0],[0,4]])

def test_constant_and_invalid_correlation_traces():
    assert np.all(correlate_trace(np.ones(1001))[:,1]==0)
    with pytest.raises(ValueError):correlate_trace(np.zeros(1001))
    with pytest.raises(ValueError):correlate_trace(np.ones(16))

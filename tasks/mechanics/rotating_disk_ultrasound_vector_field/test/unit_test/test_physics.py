import numpy as np
import pytest
from src.physics_model import beamformer,doppler_velocity,invert_receive_projections,predicted_radial,automatic_fnumber
from src.solvers import dct_filter,robust_smooth

def test_bistatic_two_component_inverse_and_physical_sign():
    rng=np.random.default_rng(8);v=rng.uniform(-.25,.25,(2,7,8))
    plus,minus=predicted_radial(v)
    np.testing.assert_allclose(invert_receive_projections(plus,minus),v,atol=3e-16)
    p=predicted_radial(np.array([[[0.]],[[.2]]]))
    assert np.all(p<0) # Positive depth motion is away from the transducer.

def test_phase_alias_period_and_finite_packet_variance():
    for phase in [.4,1.4*np.pi]:
        iq=np.broadcast_to(np.exp(1j*phase*np.arange(8)),(5,6,8)).copy()
        velocity,variance,ac=doppler_velocity(iq,window=1)
        wrapped=np.angle(np.exp(1j*phase))
        np.testing.assert_allclose(velocity,.74*wrapped/np.pi,atol=2e-15)
        np.testing.assert_allclose(variance,2*(.74/np.pi)**2/8,atol=2e-16)
    # This is the source finite-packet estimator, not a claim of zero-noise variance.

def test_sparse_receive_operator_complex_adjoint():
    x=np.array([-.003,.001,.007]);z=np.array([.018,.023,.028])
    A=beamformer(x,z,np.pi/12);rng=np.random.default_rng(19)
    source=rng.normal(size=A.shape[1])+1j*rng.normal(size=A.shape[1]);test=rng.normal(size=3)+1j*rng.normal(size=3)
    np.testing.assert_allclose(np.vdot(A@source,test),np.vdot(source,A.conj().T@test),rtol=2e-14,atol=2e-14)
    assert A.shape==(3,334*128) and np.isfinite(A.data).all()
    assert automatic_fnumber(np.pi/12)==automatic_fnumber(-np.pi/12)

def test_dct_penalty_matches_independent_repeating_boundary_matrix():
    def one(n):
        L=np.diag(np.r_[1,np.full(n-2,2),1]).astype(float)
        L+=np.diag(np.full(n-1,-1),1)+np.diag(np.full(n-1,-1),-1)
        return L
    shape=(4,5);L=np.kron(one(4),np.eye(5))+np.kron(np.eye(4),one(5))
    y=np.random.default_rng(9).normal(size=shape);s=2.7
    direct=np.linalg.solve(np.eye(20)+s*L.T@L,y.ravel()).reshape(shape)
    np.testing.assert_allclose(dct_filter(y,s),direct,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(dct_filter(np.full(shape,3.),s),3.,atol=2e-14)

def test_invalid_packets_and_masks_rejected():
    with pytest.raises(ValueError):doppler_velocity(np.zeros((3,3,2)),lag=2)
    with pytest.raises(ValueError):robust_smooth(np.zeros((2,3,4)),np.zeros((3,4),bool))
    with pytest.raises(ValueError):robust_smooth(np.zeros((2,3,4)),np.ones((3,4),bool),smoothing=-1)

def test_carrier_alias_bands_are_disjoint_at_saved_rate():
    fc=5e6;width=.15*fc;fs=4*fc/3
    # Map physical positive/negative narrow passbands to the base Nyquist interval.
    bands=[]
    for sign in [-1,1]:
        center=(sign*fc+fs/2)%fs-fs/2
        bands.append((center-width/2,center+width/2))
    assert bands[0][1]<bands[1][0] or bands[1][1]<bands[0][0]
    assert all(-fs/2<lo<hi<fs/2 for lo,hi in bands)

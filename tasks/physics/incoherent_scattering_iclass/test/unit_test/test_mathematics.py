"""Independent small finite cases and frozen upstream numerical contracts."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from src.physics_model import (circular_convolution,circular_correlation,
    phase_direction,spatial_frames,estimate_mtf,rms_object,incoherent_forward,
    covariance_coherence)
from src.preprocessing import load_observations,prepare_matrix
from src.solvers import solve
from src.visualization import align_circular,comparison_metrics,review_figure,transfer_figure

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_upstream_complex_operators():
    with np.load(ROOT/'test/fixtures/operators.npz',allow_pickle=False) as f:
        a=torch.from_numpy(f['input_matrix'])
        np.testing.assert_allclose(circular_convolution(a.conj().flipud(),a.fliplr()),f['output_convolution'],rtol=1e-12,atol=1e-13)
        conv=circular_convolution(a.conj().flipud(),a.fliplr())[:,4]
        np.testing.assert_allclose(circular_correlation(a.flipud(),conv[:,None].conj()),f['output_correlation'],rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(phase_direction(a),f['output_direction'],rtol=1e-12,atol=1e-13)


def test_independent_dense_virtual_reflection_phase():
    rng=np.random.default_rng(4)
    a=rng.normal(size=(12,5))+1j*rng.normal(size=(12,5))
    n,m=a.shape
    covariance=a@a.conj().T
    diagonals=np.array([sum(covariance[(j+d)%n,j] for j in range(n)) for d in range(n)])
    dense=np.array([sum(covariance[i,j].conj()*diagonals[(i-j)%n] for j in range(n)) for i in range(n)])
    fast=phase_direction(torch.from_numpy(a)).numpy()
    np.testing.assert_allclose(fast,dense/(n*n*m*m*m),rtol=2e-13,atol=1e-14)
    expected=np.sum(np.abs(diagonals)**2)/np.trace(covariance).real**2
    assert covariance_coherence(torch.from_numpy(a))==pytest.approx(expected,rel=1e-12)


def test_spatial_inverse_does_not_mix_frame_axis():
    rng=np.random.default_rng(8)
    frames=rng.normal(size=(7,8,10))
    spectrum=np.fft.fftshift(np.fft.fft2(frames),axes=(-2,-1))
    matrix=spectrum.transpose(2,1,0).reshape(80,7).copy()
    actual=spatial_frames(torch.from_numpy(matrix),(8,10)).numpy()
    np.testing.assert_allclose(actual,frames,atol=2e-14)


def test_paper_s18_per_frame_filter_before_rms():
    rng=np.random.default_rng(9)
    frames=rng.normal(size=(13,8,10))
    spectrum=np.fft.fftshift(np.fft.fft2(frames),axes=(-2,-1))
    matrix=spectrum.transpose(2,1,0).reshape(80,13).copy()
    mtf=np.linspace(.05,1,80)
    filtered=spectrum/(mtf.reshape(10,8).T[None]+.01)
    expected=np.sqrt(np.mean(np.abs(np.fft.ifft2(np.fft.ifftshift(filtered,axes=(-2,-1))))**2,axis=0))
    expected/=expected.max()
    actual=rms_object(torch.from_numpy(matrix),(8,10),torch.from_numpy(mtf)).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-14)


def test_intensity_forward_conservation_and_known_point():
    obj=np.zeros((8,10));obj[0,0]=3
    psf=np.zeros_like(obj);psf[0,0]=.7;psf[1,2]=.3
    illumination=np.ones((4,8,10))
    measured=incoherent_forward(obj,illumination,psf)
    np.testing.assert_allclose(measured,3*np.broadcast_to(psf,measured.shape),atol=1e-15)
    np.testing.assert_allclose(measured.sum(axis=(-2,-1)),3,atol=1e-14)
    assert measured.min()>-1e-14


def test_covariance_phase_piston_and_gain_invariance():
    rng=np.random.default_rng(10)
    a=torch.from_numpy(rng.normal(size=(12,5))+1j*rng.normal(size=(12,5)))
    b=a*np.exp(1j*.77)*3.9
    assert covariance_coherence(a)==pytest.approx(covariance_coherence(b),rel=2e-13)


def test_loader_rejects_bad_shape_and_dtype(tmp_path):
    path=tmp_path/'data.npz'
    np.savez(path,camera_frames=np.ones((1,2,3,4),np.float32))
    with pytest.raises(ValueError):load_observations(path)
    np.savez(path,other=np.ones(2))
    with pytest.raises(ValueError):load_observations(path)


def test_preprocessing_and_solver_do_not_mutate_inputs():
    rng=np.random.default_rng(11);a=rng.random((5,8,10)).astype(np.float32);saved=a.copy()
    matrix,rms=prepare_matrix(a)
    assert matrix.shape==(80,5) and matrix.dtype==torch.complex64
    result,details=solve(a,iterations=2,device='cpu',record_steps=(1,))
    assert result['reconstruction'].shape==(1,8,10)
    assert 1 in details['snapshots']
    np.testing.assert_array_equal(a,saved)


@pytest.mark.parametrize('function,args',[
    (estimate_mtf,(torch.zeros((4,3),dtype=torch.complex64),)),
    (spatial_frames,(torch.ones((4,3),dtype=torch.complex64),(2,3))),
    (rms_object,(torch.zeros((4,3),dtype=torch.complex64),(2,2))),
    (solve,(np.ones((3,4,4),np.float32),0)),
])
def test_core_degenerate_inputs(function,args):
    with pytest.raises(ValueError):function(*args)


def test_metrics_known_shift_and_figures():
    import matplotlib.pyplot as plt
    a=np.arange(80,dtype=float).reshape(8,10)+1;b=np.roll(a,(2,-3),(0,1))
    aligned,_=align_circular(b,a);np.testing.assert_array_equal(aligned,a)
    metric=comparison_metrics(b,a)
    assert metric['ncc']==pytest.approx(1)
    assert metric['nrmse']==pytest.approx(0)
    fig=review_figure([a]*6,['test']*6);fig.canvas.draw();plt.close(fig)
    fig=transfer_figure(np.ones_like(a,dtype=complex),a/a.max());fig.canvas.draw();plt.close(fig)

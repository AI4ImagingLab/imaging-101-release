from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from scipy.signal import correlate2d
from src.physics_model import blur_image,structural_similarity,loss_for_step
from src.solvers import KernelPrior,learning_rates,reconstruct
from src.preprocessing import load_observation,IMAGE_SHAPE,KERNEL_SIZE
from src.visualization import image_metrics,align_translation,forward_nrmse

ROOT=Path(__file__).resolve().parents[2]


def test_public_data_and_metadata_constants():
    y=load_observation(ROOT/'data/raw_data.npz')
    assert y.shape==IMAGE_SHAPE and y.dtype==np.float32
    assert KERNEL_SIZE==51
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for literal in ('753','668','51','255','25','50'):assert literal in text


@pytest.mark.parametrize('bad', ['shape','dtype','key','nan','range'])
def test_loader_rejects_bad_archive(tmp_path,bad):
    a=np.zeros((1,753,668),dtype=np.float32);key='observation'
    if bad=='shape':a=a[:,:-1]
    if bad=='dtype':a=a.astype(np.float64)
    if bad=='key':key='incorrect'
    if bad=='nan':a[0,0,0]=np.nan
    if bad=='range':a[0,0,0]=1.1
    p=tmp_path/'input.npz';np.savez(p,**{key:a})
    with pytest.raises(ValueError):load_observation(p)


def test_valid_correlation_analytic_and_adjoint():
    rng=np.random.default_rng(5);x=rng.normal(size=(11,13));k=rng.normal(size=(3,5));r=rng.normal(size=(9,9))
    xt=torch.tensor(x[None,None],requires_grad=True);kt=torch.tensor(k[None,None],requires_grad=True)
    y=blur_image(xt,kt)
    np.testing.assert_allclose(y.detach().numpy()[0,0],correlate2d(x,k,mode='valid'),rtol=1e-13,atol=1e-13)
    (y*torch.tensor(r[None,None])).sum().backward()
    rhs=torch.nn.functional.conv_transpose2d(torch.tensor(r[None,None]),kt.detach())
    np.testing.assert_allclose(xt.grad.numpy(),rhs.numpy(),rtol=1e-13,atol=1e-13)
    eps=1e-6;xp=x.copy();xm=x.copy();xp[5,6]+=eps;xm[5,6]-=eps
    fd=np.sum((correlate2d(xp,k,'valid')-correlate2d(xm,k,'valid'))*r)/(2*eps)
    assert float(xt.grad[0,0,5,6])==pytest.approx(fd,rel=1e-8,abs=1e-8)


def test_blind_translation_gauge_independent_discrete_model():
    # In physical convolution image/kernel shifts are opposite. In this task's
    # cross-correlation array convention the two arrays shift in the same direction.
    rng=np.random.default_rng(9);x=rng.normal(size=(120,140));k=np.zeros((51,51));k[21:26,22:27]=rng.uniform(size=(5,5));k/=k.sum()
    shifted_x=np.roll(x,(3,-4),axis=(0,1));shifted_k=np.roll(k,(3,-4),axis=(0,1))
    original=correlate2d(x,k,'valid');shifted=correlate2d(shifted_x,shifted_k,'valid')
    np.testing.assert_allclose(original,shifted,rtol=1e-13,atol=1e-13)


def test_kernel_simplex_and_differentiable():
    torch.manual_seed(1);net=KernelPrior();out=net(torch.rand(200))
    assert out.shape==(1,1,51,51) and out.min()>0
    assert out.sum().item()==pytest.approx(1,abs=2e-7)
    out[0,0,10,20].backward();assert torch.isfinite(net.hidden.weight.grad).all()


def test_ssim_identity_and_independent_finite_difference():
    rng=np.random.default_rng(7);a=torch.tensor(rng.uniform(.1,.9,size=(1,1,15,17)),requires_grad=True);b=torch.tensor(rng.uniform(.1,.9,size=(1,1,15,17)))
    assert structural_similarity(a,a).item()==pytest.approx(1,abs=1e-13)
    value=structural_similarity(a,b);value.backward()
    ap=a.detach().clone();am=ap.clone();ap[0,0,7,8]+=1e-6;am[0,0,7,8]-=1e-6
    fd=(structural_similarity(ap,b)-structural_similarity(am,b)).item()/2e-6
    assert a.grad[0,0,7,8].item()==pytest.approx(fd,rel=1e-7,abs=1e-8)
    assert loss_for_step(a.detach(),b,499)==torch.nn.functional.mse_loss(a.detach(),b)
    assert loss_for_step(a.detach(),b,500)==1-structural_similarity(a.detach(),b)


@pytest.mark.parametrize('step,rates',[(0,(.01,.0001)),(1599,(.01,.0001)),(1600,(.005,.00005)),(1900,(.0025,.000025)),(2200,(.00125,.0000125))])
def test_learning_schedule(step,rates):assert learning_rates(step)==rates


def test_known_metrics_and_alignment():
    yy,xx=np.mgrid[:130,:140];truth=np.exp(-((yy-65)**2+(xx-70)**2)/200);image=np.roll(truth,(3,-4),(0,1))
    m=image_metrics(image,truth)
    assert m['shift_row']==-3 and m['shift_column']==4
    assert m['ncc']==pytest.approx(1,abs=1e-13) and m['nrmse']==pytest.approx(0,abs=1e-13)
    k=np.zeros((51,51));k[25,25]=1
    assert forward_nrmse(truth,k,truth)==pytest.approx(0,abs=1e-14)
    with pytest.raises(ValueError):image_metrics(np.zeros((130,140)),np.ones((130,140)))


def test_reconstruction_preconditions():
    with pytest.raises(ValueError):reconstruct(np.zeros((10,10),dtype=np.float64))
    with pytest.raises(ValueError):reconstruct(np.zeros((10,10),dtype=np.float32),iterations=0)

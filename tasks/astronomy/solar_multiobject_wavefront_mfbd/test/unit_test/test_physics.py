import numpy as np
import pytest
import torch
from src.physics_model import canonical_modes, pupil_to_psf, reduced_loss, noll_indices
from src.preprocessing import mosaic, cosine_window, normalize_frames
from src.solvers import annealing_schedule


@pytest.mark.parametrize('index,expected', [(1,(0,0)),(2,(1,1)),(3,(1,-1)),(4,(2,0)),(5,(2,-2)),(6,(2,2))])
def test_noll_known_orders(index, expected):
    assert noll_indices(index) == expected


def test_flux_hermitian_and_null_mode(optics):
    modes = torch.linspace(-0.2,0.3,12*54).reshape(1,12,54)
    psf, otf = pupil_to_psf(modes, optics['pupil'], optics['basis'])
    torch.testing.assert_close(psf.sum((-1,-2)), torch.ones((1,12)), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(otf[...,0,0].real, torch.full((1,12),1/88), rtol=2e-6, atol=1e-8)
    assert torch.all(psf >= 0)
    recovered = torch.fft.ifft2(otf,norm='ortho')
    assert torch.max(torch.abs(recovered.imag)) < 1e-8
    assert torch.count_nonzero(optics['basis'][-1]) == 0
    changed = modes.clone()
    changed[...,-1] += 100
    torch.testing.assert_close(pupil_to_psf(changed,optics['pupil'],optics['basis'])[0],psf)


def test_circular_forward_adjoint(optics):
    generator=torch.Generator().manual_seed(3)
    x=torch.randn((1,12,88,88),generator=generator,dtype=torch.float64)
    y=torch.randn((1,12,88,88),generator=generator,dtype=torch.float64)
    _,otf=pupil_to_psf(torch.zeros((1,12,54),dtype=torch.float64),
                      optics['pupil'].double(),optics['basis'].double())
    transfer=88*otf
    forward=torch.fft.ifft2(torch.fft.fft2(x)*transfer).real
    adjoint=torch.fft.ifft2(torch.fft.fft2(y)*torch.conj(transfer)).real
    torch.testing.assert_close((forward*y).sum(),(x*adjoint).sum(),rtol=1e-11,atol=1e-11)


def test_reduced_loss_directional_derivative(oracle,optics):
    modes=torch.tensor(oracle['input_test_modes'],dtype=torch.float64,requires_grad=True)
    frames=[torch.tensor(oracle[f'output_frame_ft{i}'],dtype=torch.complex128) for i in range(2)]
    sigma=[torch.tensor(oracle[f'output_sigma{i}'],dtype=torch.float64) for i in range(2)]
    pupil,basis,mask=(optics[key].double() for key in ['pupil','basis','diffraction_mask'])
    def objective(value):
        otfs=[pupil_to_psf(canonical_modes(value),pupil,basis)[1] for _ in range(2)]
        return reduced_loss(frames,otfs,sigma,mask)
    value=objective(modes)
    gradient=torch.autograd.grad(value,modes)[0]
    direction=torch.cos(torch.arange(modes.numel(),dtype=torch.float64)).reshape(modes.shape)
    direction/=torch.linalg.vector_norm(direction)
    epsilon=1e-5
    finite=(objective(modes.detach()+epsilon*direction)-objective(modes.detach()-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close(finite,(gradient*direction).sum(),rtol=2e-7,atol=2e-7)


def test_translation_gauge_and_annealing():
    modes=torch.arange(2*12*54).reshape(2,12,54).float()
    centered=canonical_modes(modes)
    assert torch.count_nonzero(centered[:,0,:2]) == 0
    torch.testing.assert_close(centered[...,2:],modes[...,2:])
    schedule=annealing_schedule()
    assert schedule.shape == (250,)
    assert schedule[0] == 2 and schedule[-1] == 54
    assert set(schedule) == {2,5,9,14,20,27,35,44,54}
    assert np.all(np.diff(schedule)>=0)


def test_mosaic_coordinate_contract():
    rows=torch.arange(512).reshape(1,512,1).expand(2,512,512).float()
    patches=torch.stack([rows[:,r:r+88,c:c+88] for r in range(0,425,40)
                        for c in range(0,425,40)],dim=1)
    result=mosaic(patches)
    assert result.shape == (2,476,476)
    torch.testing.assert_close(result[:,10:-10,10:-10],rows[:,16:472,16:472],rtol=2e-6,atol=1e-4)
    assert torch.count_nonzero(result[:,0]) == 0
    assert torch.count_nonzero(result[:,-1]) == 0


def test_normalization_rejects_nonpositive_means():
    with pytest.raises(ValueError):
        normalize_frames(np.zeros((2,12,512,512),dtype=np.float32))

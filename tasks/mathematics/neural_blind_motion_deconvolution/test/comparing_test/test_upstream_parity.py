from pathlib import Path
import gc
import os
import numpy as np
import pytest
import torch
import subprocess
import sys
from src.solvers import initialize_priors
from src.physics_model import blur_image,structural_similarity,loss_for_step
from src.visualization import image_metrics
ROOT=Path(__file__).resolve().parents[2]
UTF8_ENV=dict(os.environ,PYTHONIOENCODING='utf-8')


@pytest.fixture(autouse=True)
def release_cuda_allocations():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_independent_upstream_ssim_and_gradient():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as d:
        a=torch.tensor(d['input_ssim_a'],requires_grad=True);b=torch.tensor(d['input_ssim_b'])
        value=structural_similarity(a,b);value.backward()
        assert value.item()==pytest.approx(float(d['output_ssim']),abs=2e-7)
        np.testing.assert_allclose(a.grad.numpy(),d['output_ssim_gradient'],rtol=2e-5,atol=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='full-resolution upstream stage requires validated CUDA runtime')
def test_full_resolution_network_loss_gradients_and_adam():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as d:
        image,kn,z,zk=initialize_priors((753,668))
        optimizer=torch.optim.Adam([{'params':image.optimization_parameters()},{'params':kn.parameters(),'lr':.0001}],lr=.01)
        latent=image(z+.001*torch.from_numpy(d['input_noise']).cuda());kernel=kn(zk);prediction=blur_image(latent,kernel)
        with np.load(ROOT/'data/raw_data.npz') as raw:target=torch.from_numpy(raw['observation'][:,None]).cuda()
        objective=loss_for_step(prediction,target,0);objective.backward()
        for result,key in [(latent,'output_latent'),(kernel,'output_kernel'),(prediction,'output_prediction'),(image.scales[0].side[0][1].weight.grad,'output_gradient_image'),(kn.hidden.weight.grad,'output_gradient_kernel')]:
            np.testing.assert_allclose(result.detach().cpu().numpy(),d[key],rtol=2e-6,atol=2e-7)
        assert objective.item()==pytest.approx(float(d['output_loss']),abs=2e-7)
        parameters=list(image.optimization_parameters())
        assert len(parameters)==int(d['config_parameter_count'])
        for index,parameter in enumerate(parameters):
            np.testing.assert_allclose(parameter.grad.detach().cpu().numpy(),d[f'output_parameter_gradient_{index:03d}'],rtol=2e-6,atol=2e-7)
        optimizer.step()
        np.testing.assert_allclose(image.scales[0].side[0][1].weight.detach().cpu().numpy(),d['output_update_image'],rtol=2e-6,atol=2e-7)
        np.testing.assert_allclose(kn.hidden.weight.detach().cpu().numpy(),d['output_update_kernel'],rtol=2e-6,atol=2e-7)


@pytest.fixture(scope='module')
def fresh_pipeline(tmp_path_factory):
    if not torch.cuda.is_available():pytest.skip('complete integration test requires validated CUDA runtime')
    folder=tmp_path_factory.mktemp('fresh-blind-pipeline');path=folder/'reconstruction.npz'
    completed=subprocess.run([sys.executable,'-m','src.main','--output',str(path)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=UTF8_ENV,timeout=5400)
    (folder/'pipeline.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf8')
    assert completed.returncode==0,completed.stderr[-2000:]
    checked=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(folder/'judge.json')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=UTF8_ENV,timeout=30)
    assert checked.returncode==0,checked.stderr
    return path


def test_fresh_full_pipeline_against_frozen_upstream(fresh_pipeline):
    with np.load(fresh_pipeline) as fresh,np.load(ROOT/'notebooks/reference_outputs/upstream.npz') as upstream:
        # Fixed before the cleaned full run; no target-dependent rescaling/cropping.
        np.testing.assert_allclose(fresh['reconstruction'],upstream['reconstruction'],rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(fresh['kernel'],upstream['kernel'],rtol=2e-5,atol=2e-6)
        assert fresh['reconstruction'].dtype==np.float32 and fresh['kernel'].shape==(1,51,51)

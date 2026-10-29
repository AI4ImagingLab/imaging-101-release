from pathlib import Path
import numpy as np
from src.physics_model import createDerivatives0, vec3, psfshift, proj2simplex

FIXTURES = Path(__file__).resolve().parents[1]/'fixtures'


def test_frozen_upstream_operators():
    with np.load(FIXTURES/'upstream_operators.npz', allow_pickle=False) as z:
        dx, dy = createDerivatives0(z['x'].shape)
        np.testing.assert_allclose(dx@vec3(z['x']), z['dx'], atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(dy@vec3(z['x']), z['dy'], atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(psfshift(z['x'], (11,13)), z['padded'], atol=1e-12)
        np.testing.assert_allclose(proj2simplex(z['probe']), z['simplex'], atol=1e-12)


def test_first_alternating_update_against_original_source():
    from src.preprocessing import load_data
    from src.solvers import Params, StateH, StateFM, estimateH, estimateFM
    i,b=load_data()
    params=Params();params.maxiter=1;params.visualize=False;params.verbose=False
    params.projection_mode='upstream_cyclic'
    m0=np.ones((36,36));f0=np.ones((36,36,3))
    h,state_h=estimateH(i,b,m0,f0,state=StateH(),params=params)
    f,m,state_fm=estimateFM(i,b,h,m0,f0,state=StateFM(),params=params)
    with np.load(FIXTURES/'upstream_first_update.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(h,z['kernel'],atol=1e-10,rtol=1e-9)
        # Finite CG updates cross numerical backends, unlike exact difference
        # operators. Use the predeclared output NRMSE criterion and verify the
        # actual linear normal operators independently in test_normal_operators.
        for actual,key in ((f,'appearance'),(m,'mask')):
            nrmse=np.sqrt(np.mean((actual-z[key])**2))/np.ptp(z[key])
            assert nrmse<=0.0001,(key,nrmse)


def test_full_pipeline_from_standard_entry_point():
    import subprocess, sys
    task=FIXTURES.parents[1]
    run=subprocess.run([sys.executable, '-m', 'src.main', '--projection', 'upstream_cyclic', '--output', 'output/upstream_parity.npz'], cwd=task,
                       capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr
    with np.load(task/'output/upstream_parity.npz',allow_pickle=False) as fresh, np.load(FIXTURES/'upstream_pipeline.npz',allow_pickle=False) as reference:
        for key in ('reconstruction','mask','kernel'):
            x,y=fresh[key],reference[key]
            ncc=np.vdot(x.ravel(),y.ravel())/(np.linalg.norm(x)*np.linalg.norm(y))
            nrmse=np.sqrt(np.mean((x-y)**2))/np.ptp(y)
            assert ncc>=0.99999,(key,ncc)
            assert nrmse<=0.0001,(key,nrmse)


def test_corrected_standard_pipeline_against_independent_dykstra_oracle():
    import subprocess, sys
    task=FIXTURES.parents[1]
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=task,
                       capture_output=True,text=True,timeout=900)
    assert run.returncode==0,run.stderr
    with np.load(task/'output/reconstruction.npz',allow_pickle=False) as fresh, np.load(task/'evaluation/baseline_reference.npz',allow_pickle=False) as reference:
        for key in ('reconstruction','mask','kernel'):
            x,y=fresh[key],reference[key]
            ncc=np.vdot(x.ravel(),y.ravel())/(np.linalg.norm(x)*np.linalg.norm(y))
            nrmse=np.sqrt(np.mean((x-y)**2))/np.ptp(y)
            assert ncc>=0.99999,(key,ncc)
            assert nrmse<=0.0001,(key,nrmse)
        # The linear-solve primal is not the constrained splitting auxiliary.
        # Quantify finite-ADMM feasibility without altering returned arrays.
        assert fresh['reconstruction'].min()>-.02
        assert fresh['mask'].min()>=0 and fresh['mask'].max()<=1
        assert np.max(fresh['reconstruction']-fresh['mask'][...,None])<.003
        assert fresh['kernel'].min()>=0
        assert abs(fresh['kernel'].sum()-1)<1e-12

from pathlib import Path
import numpy as np
import pytest
from src import preprocessing as pre,physics_model as physics,solvers
ROOT=Path(__file__).resolve().parents[2]

def fixture(name):
    with np.load(ROOT/'test/fixtures'/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def test_original_metric_dspacing_grid():
    f=fixture('source_geometry.npz');raw=pre.load_observations(ROOT/'data/raw_data.npz')
    A=pre.reciprocal_metric_coefficients()
    np.testing.assert_allclose(A,f['A'],rtol=2e-14,atol=1e-17)
    np.testing.assert_allclose(pre.dspacings(raw['hkl']),f['dspacings'],rtol=2e-14)
    np.testing.assert_array_equal(pre.half_grid(),f['hmax'])
    np.testing.assert_array_equal(pre.half_grid(3.5,offset_search=True),f['offset_hmax'])

def test_original_compiled_symmetry_cases():
    f=fixture('source_geometry.npz')
    for i,h in enumerate(f['test_hkl']):
        absent,multiplicity,u,p=pre.equivalent_reflections(h)
        assert absent==f['absences'][i] and multiplicity==f['multiplicities'][i]
        count=f['counts'][i]
        np.testing.assert_array_equal(u,f['uniqs'][i,:count])
        np.testing.assert_allclose(p,f['phases'][i,:count],atol=1e-15)

def test_original_full_initialization():
    f=fixture('upstream_stages.npz');raw=pre.load_observations(ROOT/'data/raw_data.npz')
    initial=pre.initialize_structure_factors(raw,seed=20261022)
    np.testing.assert_allclose(initial,f['Ehkl_initial'],rtol=5e-7,atol=1e-9)

@pytest.mark.parametrize('steps',[1,2,7,25,100])
def test_original_each_iterative_stage(steps):
    # An independent single update at every old-backend source state checks
    # equations without pretending chaotic NumPy1.26/2.4 trajectories coincide.
    f=fixture('upstream_stages.npz');initial=f['Ehkl_initial']
    before=np.fft.ifftshift(np.fft.ifftn(f[f'rho_before_update_{steps}']))
    state,diag=physics.charge_flip_update(before,abs(initial))
    np.testing.assert_allclose(state,f[f'Ehkl_{steps}'],rtol=5e-7,atol=1e-9)
    np.testing.assert_allclose(diag['density'],f[f'rho_before_update_{steps}'],rtol=5e-7,atol=1e-9)
    np.testing.assert_allclose(diag['flipped_density'],f[f'flipped_rho_{steps}'],rtol=5e-7,atol=1e-9)
    np.testing.assert_allclose(diag['temporary_factors'],f[f'CFhkl_{steps}'],rtol=5e-7,atol=1e-9)
    np.testing.assert_allclose(diag['sigma'],f[f'CEsigma_{steps}'],rtol=5e-7)

def test_complete_original_pipeline():
    raw=pre.load_observations(ROOT/'data/raw_data.npz');result=solvers.reconstruct(raw)
    f=fixture('upstream_modern_stages.npz');reference=f['reconstruction']
    np.testing.assert_allclose(result['reconstruction'],reference,rtol=5e-7,atol=1e-9)
    np.testing.assert_allclose(result['history'],f['history'],rtol=5e-7,atol=1e-9)
    assert result['iterations']==200 and result['reconstruction'].shape==(72,24,96)

@pytest.mark.parametrize('steps',[1,2,7,25,100])
def test_modern_original_full_free_running_trajectory(steps):
    f=fixture('upstream_modern_stages.npz');initial=f['Ehkl_initial'];state=initial.copy()
    for _ in range(steps):state,diag=physics.charge_flip_update(state,abs(initial))
    np.testing.assert_allclose(state,f[f'Ehkl_{steps}'],rtol=5e-7,atol=1e-9)
    np.testing.assert_allclose(diag['density'],f[f'rho_before_update_{steps}'],rtol=5e-7,atol=1e-9)

def test_old_and_modern_whole_results_meet_unchanged_prelocal_quality_limits():
    from evaluation.judge import align_periodic,physical_metrics
    old=fixture('upstream_stages.npz');new=fixture('upstream_modern_stages.npz')
    # Retain and quantify the entire original history difference explicitly.
    assert old['history'].shape==new['history'].shape==(200,2)
    assert np.max(abs(old['history'][:,1]-new['history'][:,1]))>0
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:r=z['reconstruction'][0]
    aligned,_,_=align_periodic(new['reconstruction'],r)
    import json
    limits=json.loads((ROOT/'evaluation/metrics.json').read_text())
    assert np.sum(aligned*r)/(np.linalg.norm(aligned)*np.linalg.norm(r))>=limits['ncc_boundary']
    assert np.sqrt(np.mean((aligned-r)**2))/np.ptp(r)<=limits['nrmse_boundary']
    assert physical_metrics(new['reconstruction'],ROOT/'data/raw_data.npz')['intensity_relative_l2']<=limits['intensity_relative_l2_boundary']

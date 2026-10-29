from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,fill_gaps
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_clock_edge_biharmonic_svd_and_fourier_stages():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:reference={k:f[k] for k in f.files}
    d=load_observations(ROOT/'data/raw_data.npz')[0]
    np.testing.assert_array_equal(d.dyn,reference['native'])
    d.trim_edges();d.dyn=fill_gaps(d.dyn)
    np.testing.assert_allclose(d.dyn,reference['refilled'],rtol=1e-8,atol=1e-9)
    d.correct_dyn();np.testing.assert_allclose(d.dyn,reference['corrected'],rtol=1e-8,atol=1e-9)
    d.calc_sspec()
    # Mean subtraction makes DC mathematically zero. Its log is undefined;
    # compare every bin in physical linear power at the unchanged tolerance.
    np.testing.assert_allclose(10**(d.sspec/10),10**(reference['secondary_frequency']/10),rtol=1e-8,atol=1e-9)
    assert 10**(d.sspec[0,d.sspec.shape[1]//2]/10)<1e-20
    d.scale_dyn();np.testing.assert_allclose(d.lamdyn,reference['wavelength_dynamic'],rtol=1e-8,atol=1e-9)
    d.fit_arc(etamin=2,etamax=400,delmax=1,numsteps=1000,nsmooth=7,startbin=2)
    np.testing.assert_allclose(10**(d.lamsspec/10),10**(reference['final_lamsspec']/10),rtol=1e-8,atol=1e-9)
    assert 10**(d.lamsspec[0,d.lamsspec.shape[1]//2]/10)<1e-20
    for actual,key in [(d.fdop,'final_fdop'),(d.tdel,'final_tdel'),(d.beta,'final_beta'),(d.eta_array,'final_eta_array'),(d.norm_sspec_avg,'final_norm_sspec_avg')]:
        np.testing.assert_allclose(actual,reference[key],rtol=1e-8,atol=1e-9)

def test_full_all_eight_plus_stitched_pipeline():
    result,diagnostics,record=reconstruct(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:reference=f['reconstruction']
    np.testing.assert_allclose(result,reference,rtol=1e-7,atol=0)
    assert len(diagnostics)==9 and record['native_segment_count']==8

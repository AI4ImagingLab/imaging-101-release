from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,prepare_system,prepare_measurement
from src.solvers import kaczmarz,row_norms
ROOT=Path(__file__).resolve().parents[2]

def test_first_updates_and_row_norms_frozen_original():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        rows=z['system_rows'];ids=z['row_indices'];lam=float(z['lambda_value'])
        np.testing.assert_array_equal(row_norms(rows),z['row_norm'][ids])
        np.testing.assert_allclose(kaczmarz(rows[:10],z['measurement'][:10],1,lam),z['first10_update'],rtol=2e-7,atol=1e-6)

def test_complete_native_calibration_measurement_and_all_sweeps():
    with load_observations(ROOT/'data/raw_data.npz') as z:
        S=prepare_system(z['calibration'],z['calibration_background'],copy=False)
        fg=z['foreground'];fgmean=fg[0].mean(axis=0);del fg
        bg=z['background'];bgmean=bg[0].mean(axis=0);del bg
        u=fgmean-bgmean
        positions=z['voxel_positions'][0]
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(S[f['row_indices']],f['system_rows']);np.testing.assert_array_equal(u,f['measurement'])
        np.testing.assert_array_equal(fgmean,f['foreground_mean']);np.testing.assert_array_equal(bgmean,f['background_mean']);np.testing.assert_array_equal(row_norms(S),f['row_norm'])
        x,history,lam=kaczmarz(S,u,3,return_history=True);assert lam==f['lambda_value']
        for a,key in zip(history,['one_sweep','two_sweeps','three_sweeps']):np.testing.assert_allclose(a,f[key],rtol=2e-7,atol=1e-6)
    assert np.all(np.diff(positions[:19,0])==.002) or np.allclose(np.diff(positions[:19,0]),.002)
    assert np.all(positions[:19,1:]==positions[0,1:]);assert np.allclose(positions[[0,-1]],[[ -.018,-.018,-.009],[.018,.018,.009]])

def test_standard_entrypoint_fresh_judge(tmp_path):
    from src.main import run
    from evaluation.judge import evaluate
    p=tmp_path/'reconstruction.npz';x,elapsed=run(ROOT/'data/raw_data.npz',p)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:np.testing.assert_allclose(x,z['reconstruction'],rtol=2e-7,atol=1e-6)
    assert evaluate(p,tmp_path/'report.json')==0


def test_native_fourier_sign_amplitude_and_all_receiver_mapping():
    with np.load(ROOT/'test/fixtures/native_fourier_cases.npz',allow_pickle=False) as f:
        n=np.arange(53856)
        for wave,k,c in zip(f['waveform'],f['frequency_index'],f['coefficient']):
            expected=np.dot(wave.astype(float),np.exp(-2j*np.pi*k*n/53856))
            np.testing.assert_allclose(c,expected,rtol=1e-12,atol=1e-7)
        with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
            for key,offset in [('foreground',0),('background',1000)]:
                spectra=z[key][0]
                for row,frame,coefficient in zip(f['row'],f['frame'],f['coefficient']):
                    if offset<=frame<offset+1000:np.testing.assert_allclose(spectra[frame-offset,row],coefficient,rtol=1e-9,atol=1e-4)
                del spectra

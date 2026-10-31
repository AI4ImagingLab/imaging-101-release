from pathlib import Path
import numpy as np
from src.preprocessing import load_data,analytic_filter,DT,SOUND_SPEED
from src.physics_model import reconstruction_grid,travel_times,reciprocity_weights
from src.solvers import reconstruct,delay_and_sum
ROOT=Path(__file__).resolve().parents[2]

def test_upstream_stages_and_pipeline():
    f=np.load(ROOT/'test/fixtures/upstream.npz');d=load_data(ROOT/'data/raw_data.npz')
    for key,name in [('traces','input_traces'),('tx','input_tx'),('rx','input_rx'),('time','input_time'),('positions','input_positions')]:np.testing.assert_array_equal(d[key],f[name])
    np.testing.assert_allclose(analytic_filter(d['traces']),f['output_filtered'],rtol=2e-12,atol=1e-14)
    x,z=reconstruction_grid();np.testing.assert_array_equal(x,f['input_x']);np.testing.assert_array_equal(z,f['input_z'])
    np.testing.assert_allclose(reconstruct(d),f['output_reconstruction'],rtol=1e-10,atol=2e-14)

def test_travel_times_and_reciprocity():
    p=np.array([[0.,0.,0.],[.003,0.,0.]])
    t=travel_times(p,np.array([0.]),np.array([.004]));np.testing.assert_allclose(t,[[.004/6300,.005/6300]],rtol=1e-15)
    np.testing.assert_array_equal(reciprocity_weights(np.array([0,0,1]),np.array([0,1,1])),[1,2,1])
    traces=np.ones((3,20),complex);assert np.allclose(delay_and_sum(traces,np.array([0,0,1]),np.array([0,1,1]),np.zeros((1,2))),4/3)
    assert np.allclose(delay_and_sum(traces,np.array([0,0,1]),np.array([0,1,1]),np.ones((1,2))),0)

def test_frequency_filter_and_constants():
    n=750;t=np.arange(n)*DT; a=analytic_filter(np.cos(2*np.pi*2e6*t)[None])[0]
    assert abs(np.abs(a[100:-100]).mean()-1)<.005
    text=(ROOT/'README.md').read_text(encoding='utf8');assert '6300' in text and '4e-8' in text and SOUND_SPEED==6300 and DT==4e-8


def test_public_array_table_matches_native_dtypes():
    d=load_data(ROOT/'data/raw_data.npz');text=(ROOT/'README.md').read_text(encoding='utf8')
    for key,array in d.items():
        line=next(line for line in text.splitlines() if line.startswith('| '+key+' |'))
        assert '| '+str(array.dtype)+' |' in line

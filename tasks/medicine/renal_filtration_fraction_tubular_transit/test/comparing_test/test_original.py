from pathlib import Path
import numpy as np
from src.preprocessing import load_data,case_data
from src.physics_model import arterial_concentration,compartment_concentration,predict,exponential_convolution,steady_signal
from src.solvers import reconstruct,INITIAL
ROOT=Path(__file__).resolve().parents[2]

def test_initial_and_independent_stages():
    np.testing.assert_array_equal(INITIAL,[.02,.15,.1,120.])
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f,np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as ref:
        np.testing.assert_allclose(exponential_convolution(f['input_flux'],3.,f['input_time']),f['output_expconv'],atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(compartment_concentration(f['input_flux'],f['input_time'],[.02,.15,.2,120.]),f['output_concentration'],atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(steady_signal(100.,np.array([.5,1,2,4]),.005,17),f['output_ss_signal'],atol=1e-12,rtol=1e-12)
        for i in range(228):
            t,y,a,cal=case_data(data,i);ca=arterial_concentration(a,cal)
            np.testing.assert_allclose(ca,f['output_ca'][0,i,:len(t)],atol=1e-10,rtol=1e-10)
            s,C,R1=predict(ref['reconstruction'][0,i],t,ca,cal,f['output_S0'][0,i])
            np.testing.assert_allclose(C,f['output_compartments'][0,i,:,:len(t)],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(R1,f['output_R1'][0,i,:len(t)],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(s,ref['signal'][0,i,:len(t)],atol=1e-10,rtol=1e-10)

def test_full_cohort_and_model_comparisons():
    data=load_data(ROOT/'data/raw_data.npz')
    result,_=reconstruct(data)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as ref:
        for k in result:np.testing.assert_allclose(result[k],ref[k],atol=1e-10,rtol=1e-10)
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as ref:
        for variant in ['linear','linear_delay','ss_literature']:
            result,_=reconstruct(data,variant)
            np.testing.assert_allclose(result['reconstruction'],ref[variant+'_parameters'],atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(result['signal'],ref[variant+'_signal'],atol=1e-9,rtol=1e-9)

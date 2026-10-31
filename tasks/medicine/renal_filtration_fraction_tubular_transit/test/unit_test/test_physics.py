from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from src.preprocessing import HCT,load_data,case_data
from src.physics_model import exponential_convolution,compartment_concentration,steady_signal,arterial_concentration
from src.visualization import clinical_statistics
ROOT=Path(__file__).resolve().parents[2]

def test_exponential_convolution_analytic_constant():
    t=np.array([0.,.1,1.,2.7,8.]);f=np.ones(len(t))*2
    np.testing.assert_allclose(exponential_convolution(f,3,t),2*(1-np.exp(-t/3)),atol=1e-14)
    np.testing.assert_array_equal(exponential_convolution(f,0,t),f)

def test_transport_independent_ode_and_mass_balance():
    t=np.linspace(0,30,2001);ca=np.ones(len(t))*.002
    Fp,vp,FF,Tt=.02,.15,.2,120.;Ft=Fp*FF;Tp=vp/(Fp+Ft)
    C=compartment_concentration(ca,t,[Fp,vp,FF,Tt])
    solution=solve_ivp(lambda time,c:[Fp*.002-c[0]/Tp,Ft*c[0]/vp-c[1]/Tt],[0,30],[0,0],t_eval=t,rtol=1e-11,atol=1e-14).y
    np.testing.assert_allclose(C,solution,atol=2e-10,rtol=2e-6)
    assert np.all(C>=0)
    # Total input equals storage plus integrated plasma/tubular outflow.
    out=Fp*C[0]/vp+C[1]/Tt
    assert abs(C[:,-1].sum()+np.trapezoid(out,t)-Fp*.002*30)<1e-9

def test_signal_calibration_roundtrip_and_units():
    cal={'blood_t1':1.6,'n0':3,'relaxivity':3500.,'tr':.005,'flip_angle':17.}
    concentration=np.r_[np.zeros(3),np.linspace(0,.001,20)]
    signal=steady_signal(500,1/1.6+3500*concentration,.005,17.)
    np.testing.assert_allclose(arterial_concentration(signal,cal),concentration,atol=5e-17)
    assert HCT==.45

def test_public_data_unchanged_and_padding():
    data=load_data(ROOT/'data/raw_data.npz')
    assert data['lengths'].min()==44 and data['lengths'].max()==125
    for i,L in enumerate(data['lengths'][0]):
        for k in ['measurements','arterial_signal','time']:assert np.all(data[k][0,i,L:]==0)
    assert set(data['field_strength'].ravel())=={1.,3.}
    assert 'isotope' not in ' '.join(data)

def test_clinical_metrics_known_case():
    s=clinical_statistics(np.array([12,18]),np.array([10,20]))
    assert s=={'bias':0.,'sd':2.,'rmse':2.,'p30':1.}

def test_readme_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for literal in ['0.45','3500','4350','1.378','1.6286644951140066','1.017','1.542','228','125']:assert literal in text

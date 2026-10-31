from pathlib import Path
import numpy as np
from src.generate_data import simulate
from src.preprocessing import load_observations
ROOT=Path(__file__).resolve().parents[2]

def test_original_generation_subcase_before_extraction():
    data=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_generation.npz',allow_pickle=False) as z:
        fields=simulate(data['mass'],data['stiffness'],data['damping'],z['input_force_matrix'],z['input_forces'])
        for a,name in zip(fields,['displacement','velocity','acceleration']):
            np.testing.assert_allclose(a[::10],z['output_'+name],atol=1e-8,rtol=1e-7)

def test_constant_force_free_mass_independent_solution():
    dt=.01;time=np.arange(101)*dt
    u,v,a=simulate(np.array([[2.]]),np.zeros((1,1)),np.zeros((1,1)),np.ones((1,1)),np.ones((101,1))*6,dt)
    np.testing.assert_allclose(a[:,0],3,atol=1e-10)
    np.testing.assert_allclose(v[:,0],3*time,atol=1e-10)
    np.testing.assert_allclose(u[:,0],1.5*time**2,atol=1e-10)

def test_undamped_unforced_energy_conservation():
    M=np.eye(2);K=np.diag([1.,4.]);u,v,a=simulate(M,K,np.zeros((2,2)),np.eye(2),np.zeros((1001,2)),.01,initial_displacement=[1.,.5])
    energy=.5*np.einsum('ti,ij,tj->t',u,K,u)+.5*np.sum(v*v,axis=1)
    np.testing.assert_allclose(energy,energy[0],rtol=1e-11,atol=1e-12)

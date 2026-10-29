from pathlib import Path
import numpy as np
from src.physics_model import thermal_properties,time_derivative,laplacian,energy_terms,predicted_storage
from src.solvers import reconstruct
from src.preprocessing import load_data

ROOT=Path(__file__).resolve().parents[2]

def test_second_order_temporal_derivative():
    time=np.arange(5.)[:,None,None];temperature=np.broadcast_to(300+time**2,(5,6,7))
    np.testing.assert_allclose(time_derivative(temperature),np.broadcast_to(2*time,temperature.shape),atol=1e-12)

def test_square_grid_quadratic_laplacian():
    x=np.arange(7)*.0127;y=np.arange(6)*.0127
    t=np.broadcast_to(300+3*y[None,:,None]**2+2*x[None,None,:]**2,(5,6,7))
    np.testing.assert_allclose(laplacian(t),10,atol=3e-9)

def test_equilibrium_incoming_ambient_radiation():
    data={'temperature':np.full((5,6,7),293.15)}
    q,terms=reconstruct(data)
    np.testing.assert_allclose(q,5.67e-8*293.15**4/1000,rtol=1e-14)
    np.testing.assert_array_equal(terms['storage'],0.)
    np.testing.assert_array_equal(terms['convection'],0.)
    np.testing.assert_array_equal(terms['conduction'],0.)

def test_calibrated_material_formula():
    t=np.array([300.,500.]);k,cp,rho=thermal_properties(t)
    np.testing.assert_allclose(k,8.1119*np.log(t)-30.377,atol=0)
    np.testing.assert_allclose(cp,1000*(.1414*np.log(t)-.3534),atol=0)
    assert rho==8050

def test_energy_balance_independent_of_reference():
    data=load_data(ROOT/'data/raw_data.npz');q,terms=reconstruct(data)
    np.testing.assert_allclose(predicted_storage(data['temperature'],q),terms['storage'],rtol=1e-11,atol=1e-10)

def test_measurement_collar_locality():
    t=np.full((5,6,7),300.);before=laplacian(t)
    t[:,0,2]+=1
    difference=laplacian(t)-before
    assert np.count_nonzero(difference)==5
    np.testing.assert_allclose(difference[:,0,1],1/.0127**2)

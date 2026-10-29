from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from src.solvers import from_internal
from src.physics_model import transmission
from src.preprocessing import load_data
from evaluation.judge import evaluate
ROOT=Path(__file__).resolve().parents[2]
def test_original_de_initial_population_and_sine_bounds():
    rows=[]
    def objective(internal):
        physical=from_internal(internal);rows.append(physical)
        if len(rows)>21:raise StopIteration
        r=np.r_[physical/np.array([1.,100.,4200.]),1.]
        return (r*r).sum()
    try:differential_evolution(objective,[(-np.pi/2,np.pi/2)]*3,seed=20261029,popsize=7,tol=1e-7,atol=0.,polish=False,maxiter=1500)
    except StopIteration:pass
    with np.load(ROOT/'test/fixtures/upstream_de_population.npz',allow_pickle=False) as z:np.testing.assert_array_equal(rows[:21],z['physical_parameters'])
def test_full_native_original_parameter_response_identity():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        for seed in [20261029,20261030,20261031]:
            result={};parameters=[]
            for angle,(x,y) in data.items():
                b,t,theta=z[f'parameters_{angle}_{seed}'][0]
                fresh=transmission(x,t,b,np.cos(theta),np.sin(theta))
                np.testing.assert_allclose(fresh,z[f'response_{angle}_{seed}'][0],rtol=1e-8,atol=1e-12)
                parameters.append([b*1e-4,t+273.15,theta]);result[f'reconstruction_{angle}']=fresh[None]
            parameters=np.array(parameters);result.update(magnetic_field=parameters[None,:,0],temperature=parameters[None,:,1],polarization=parameters[None,:,2])
            assert evaluate(result)['passed'],evaluate(result)

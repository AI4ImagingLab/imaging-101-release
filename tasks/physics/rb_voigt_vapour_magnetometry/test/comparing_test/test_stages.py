from pathlib import Path
import numpy as np
from src.physics_model import Hamiltonian,susceptibility,transmission

def test_original_stages():
    with np.load(Path(__file__).parents[1]/'fixtures/upstream_stages.npz',allow_pickle=False) as a:
        for isotope in ['Rb85','Rb87']:
            model=Hamiltonian(isotope,'D2',1.,4200.)
            np.testing.assert_allclose(model.groundEnergies,a[isotope+'_ground_energies'],rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(model.excitedEnergies,a[isotope+'_excited_energies'],rtol=1e-10,atol=1e-7)
        for angle in [0,45,90]:
            x=a[f'input_detuning_{angle}']
            np.testing.assert_allclose(susceptibility(x,100.,4200.),a[f'output_chi_{angle}'],rtol=1e-8,atol=1e-12)
            theta=np.deg2rad(np.clip(angle,2,88))
            np.testing.assert_allclose(transmission(x,100.,4200.,np.cos(theta),np.sin(theta)),a[f'output_transmission_{angle}'],rtol=1e-8,atol=1e-12)

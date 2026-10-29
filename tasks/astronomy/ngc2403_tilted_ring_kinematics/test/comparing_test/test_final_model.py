from pathlib import Path
import numpy as np
from src import physics_model as model
F=Path(__file__).resolve().parents[1]/'fixtures'

def test_original_density_profile():
    function=getattr(model,'azimuthal_profile',None)
    assert callable(function),'input-only annular flux profile is not implemented'
    with np.load(F/'density_profile.npz',allow_pickle=False) as z:
        actual=function(z['input_moment0'],z['input_parameters'])
        np.testing.assert_allclose(actual,z['output_profile'],rtol=0,atol=1e-6)

def test_original_full_disk_discretization():
    with np.load(F/'final_cloud_model.npz',allow_pickle=False) as z:
        radius=np.arange(15,1216,30,dtype=np.float32);radius[0]=0;radius[-1]+=15
        rings,metadata,_=model.internal_rings(radius,z['input_parameters'],z['input_density'])
        np.testing.assert_allclose(rings,z['output_internal_rings'],rtol=0,atol=1e-5)
        np.testing.assert_allclose(metadata,z['output_metadata'],rtol=0,atol=1e-8)

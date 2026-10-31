import numpy as np
from src.solvers import model_at
from src.preprocessing import load,regions
def test_index_matched_sphere_is_invisible():
 c=np.array([[10.,30.,80.],[5.,40.,50.],[0.,0.,0.]])
 assert np.allclose(model_at([20.,20.,200.,1.,1.34],c),1.,atol=2e-13)
def test_observation_precision_and_normalization():
 image,bg,z=load();assert image.dtype==np.uint8 and bg.dtype==np.uint8
 assert np.array_equal(z,(image.astype(float)-13)/(bg.astype(float)-13))
 for crop,coords,valid in regions(z):assert coords.shape==(3,crop.size) and valid.sum()==crop.size-1

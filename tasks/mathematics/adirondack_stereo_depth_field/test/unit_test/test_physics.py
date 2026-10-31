import numpy as np
from src.physics_model import census_bytes,cost_volume,depth_from_disparity,project_points
from src.solvers import minplus_message
from pathlib import Path

def test_independent_camera_projection_scale_and_sign():
    points=np.array([[100.,-30.,2000.],[-50.,90.,1600.],[0.,0.,3000.]])
    left,right=project_points(points)
    expected_left=points[:,:2]/points[:,2,None]*1038.018+[322.037,243.393]
    expected_right=(points[:,:2]-[176.252,0])/points[:,2,None]*1038.018+[375.308,243.393]
    np.testing.assert_allclose(left,expected_left,rtol=0,atol=1e-12)
    np.testing.assert_allclose(right,expected_right,rtol=0,atol=1e-12)
    np.testing.assert_allclose(depth_from_disparity(left[:,0]-right[:,0]),points[:,2],rtol=1e-13)

def test_exhaustive_pairwise_minplus_potential():
    values=np.array([[0,2,4,1,3],[3,1,0,2,4]],dtype=np.float32)
    labels=np.arange(5);delta=abs(labels[:,None]-labels[None])
    potential=np.where(delta==0,0,np.where(delta==1,2,5)).astype(np.float32)
    expected=np.min(values[:,None,:]+potential[None],axis=-1)-values.min(axis=-1,keepdims=True)
    np.testing.assert_array_equal(minplus_message(values,2.,5.),expected)

def test_monotone_radiometry_and_boundary_bits():
    image=np.random.RandomState(7).randint(20,80,(13,17,3)).astype(np.float32)
    np.testing.assert_array_equal(census_bytes(image),census_bytes(2*image+13))
    constant=census_bytes(np.full((9,11,3),23,np.float32))
    assert not constant.any()

def test_known_integer_stereo_shift():
    right=np.random.RandomState(31).randint(0,256,(17,31,3),dtype=np.uint8)
    left=np.zeros_like(right);left[:,5:]=right[:,:-5]
    volume=cost_volume(left,right,dmin=-8,dmax=0)
    estimate=np.argmin(volume,axis=-1)-8
    np.testing.assert_array_equal(estimate[2:-2,7:-2],-5*np.ones((13,22),int))

def test_readme_acquisition_matches_actual_constants():
    from src import physics_model as model
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    for label,value in [('Focal length f',model.FOCAL_LENGTH),('Baseline B',model.BASELINE_MM),('Rectified origin offset a',model.DISPARITY_OFFSET)]:
        assert f'| {label} | {value} |' in text
    for label,values in [('Left principal point cL,cy',model.LEFT_CENTRE),('Right principal point cR,cy',model.RIGHT_CENTRE)]:
        assert f'| {label} | {values[0]},{values[1]} |' in text

def test_presentation_metrics_independent_known_case():
    from src.visualization import agreement
    estimate=np.array([1.,2.,4.]);reference=np.array([2.,3.,4.]);mask=np.ones(3,bool)
    actual=agreement(estimate,reference,mask)
    assert abs(actual['ncc']-24/np.sqrt(21*29))<1e-14
    assert abs(actual['nrmse']-np.sqrt(2/3)/2)<1e-14
    assert actual['bad1']==0

def test_source_boundary_guard_and_first_label_tie():
    from src.solvers import aggregate
    cost=np.zeros((5,7,4),np.float32)
    indices,diagnostics=aggregate(cost,capture_points=np.array([[0,0],[4,6],[2,3]]))
    np.testing.assert_array_equal(indices,np.zeros((5,7),int))
    for sample in diagnostics['pass_samples']:assert not sample.any()

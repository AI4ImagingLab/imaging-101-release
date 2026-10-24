from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import filter_adf,find_peaks
from src.physics_model import gaussian
from src.solvers import fit_crop
F=Path(__file__).parents[1]/'fixtures'
def test_full_filter_and_initial_peaks():
    z=np.load(F/'upstream_stages.npz',allow_pickle=False)
    raw=np.load(Path(__file__).parents[2]/'data/raw_data.npz',allow_pickle=False)
    a=filter_adf(raw['adf'][0])
    assert_allclose(a,z['run_image_filtering_0'],atol=2e-13,rtol=2e-13)
    assert np.array_equal(find_peaks(a),z['get_atom_positions_0'])
def test_first_three_independent_source_fits():
    z=np.load(F/'upstream_stages.npz',allow_pickle=False)
    for i,(x,y) in enumerate([(370,449),(796,434),(731,448)]):
        a=fit_crop(z[f'fit{i}_image'],z[f'fit{i}_initial'][0],x,y)
        b=z[f'fit{i}_final'][0]
        # Six-parameter Gaussian fits have equivalent width/rotation gauges.
        yy,xx=np.indices(z[f'fit{i}_image'].shape)
        assert_allclose(gaussian(a,xx+x,yy+y),gaussian(b,xx+x,yy+y),atol=1e-2,rtol=1e-5)
def test_full_a_sublattice_refinement():
    from src.solvers import refine
    from src.physics_model import physical_columns
    z=np.load(F/'upstream_stages.npz',allow_pickle=False)
    a=refine(z['refine_sublattice_0_input_image0'],z['refine_sublattice_0_before'],.35,gaussian_repetitions=2)
    aa=physical_columns(a,1);bb=physical_columns(z['refine_sublattice_0_after'],1)
    assert_allclose(aa[:,:2],bb[:,:2],atol=.002,rtol=0)
    assert np.linalg.norm(aa[:,2:5]-bb[:,2:5])/np.linalg.norm(bb[:,2:5])<.0001
def test_original_lattice_midpoints():
    from src.solvers import midpoint_lattice
    from scipy.spatial import cKDTree
    z=np.load(F/'upstream_stages.npz',allow_pickle=False)
    for source,index,target in [('s0_parameters',1,'refine_sublattice_1_before'),('s1_parameters',0,'refine_sublattice_2_before')]:
        a=midpoint_lattice(z[source][:,:2],index)
        b=z[target][:,:2]
        assert a.shape==b.shape
        assert cKDTree(a).query(b)[0].max()<1e-9
def test_original_subtractions_and_secondary_fits():
    from src.solvers import subtract_columns,refine,nearest
    from src.physics_model import physical_columns
    z=np.load(F/'upstream_stages.npz',allow_pickle=False)
    raw=np.load(Path(__file__).parents[2]/'data/raw_data.npz',allow_pickle=False)
    for j,(image,p,nk) in enumerate([(raw['adf'][0],z['s0_parameters'],15),(1/raw['abf'][0],z['s0_parameters'],15),(z['remove_atoms_from_image_using_2d_gaussian_1'],z['s1_parameters'],9)]):
        neighbors=nearest(z['refine_sublattice_1_before'][:,:2],nk) if j==2 else nearest(p[:,:2],nk)
        a=subtract_columns(image,p,neighbors,fraction=.30 if j==2 else .35)
        b=z[f'remove_atoms_from_image_using_2d_gaussian_{j}']
        assert np.linalg.norm(a-b)/np.linalg.norm(b)<1e-5
    for i in [1,2]:
        a=refine(z[f'refine_sublattice_{i}_input_image0'],z[f'refine_sublattice_{i}_before'],.25,com=True)
        aa=physical_columns(a,1);bb=physical_columns(z[f'refine_sublattice_{i}_after'],1)
        assert_allclose(aa[:,:2],bb[:,:2],atol=.001,rtol=0)
        assert np.linalg.norm(aa[:,2:5]-bb[:,2:5])/np.linalg.norm(bb[:,2:5])<.0001

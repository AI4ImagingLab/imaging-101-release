from pathlib import Path
import sys
import numpy as np,torch,pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,prepare_patterns,prepare_master
from src.physics_model import detector_rays,interpolate_master,rosca_lambert_side_by_side,bruker_geometry_to_SE3
from src.orientations import cu2ho,ho2ax,cu2qu,qu_apply,qu_prod,sample_cubic_grid,cubic_reduce,cubic_elements,get_local_orientation_grid
from src.solvers import update_covariance,reconstruct,index_direct,refine
torch.set_num_threads(1)
@pytest.fixture(scope='module')
def stages():return dict(np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False))
def test_original_preprocessing_all_bound_patterns(stages):
    raw,m=load_data(ROOT/'data/raw_data.npz');p,s=prepare_patterns(raw,return_stages=True)
    np.testing.assert_array_equal(m,stages['master_raw'])
    for k,v in s.items():np.testing.assert_array_equal(v,stages[k])
    np.testing.assert_array_equal(p,stages['cleaned_patterns']);mp,initial=prepare_master(m,True);np.testing.assert_array_equal(initial,stages['master_minmax']);np.testing.assert_array_equal(mp,stages['master_processed'])
def test_original_corrected_geometry(stages):np.testing.assert_array_equal(detector_rays(),stages['detector_rays'])
def test_original_rotation_conversion_stages(stages):
    cu=torch.from_numpy(stages['cubochoric_cases']);np.testing.assert_allclose(cu2ho(cu),stages['homochoric'],atol=1e-7);np.testing.assert_allclose(ho2ax(cu2ho(cu)),stages['axis_angle'],atol=1e-7);np.testing.assert_allclose(cu2qu(cu),stages['quaternions'],atol=1e-7)
    q=cu2qu(cu);v=torch.from_numpy(stages['vectors']);np.testing.assert_allclose(qu_apply(q,v),stages['quaternion_rotated'],atol=1e-7);np.testing.assert_allclose(qu_prod(q,torch.from_numpy(stages['quaternion_cases'][0])),stages['quaternion_product'],atol=1e-7)
def test_original_lambert_forward_stages(stages):
    np.testing.assert_array_equal(rosca_lambert_side_by_side(torch.from_numpy(stages['vectors'])),stages['sphere_projection']);master=torch.from_numpy(stages['master_processed']);q=torch.from_numpy(stages['quaternion_cases']);r=detector_rays();sim=interpolate_master(master,qu_apply(q[:,None],r[None]));np.testing.assert_allclose(sim,stages['simulated_dictionary_patterns'],atol=1e-7)
    r=r/r.norm(dim=-1,keepdim=True);sim=interpolate_master(master,qu_apply(q[:,None],r[None]),align_corners=True,normalize_coords=False);np.testing.assert_allclose(sim,stages['simulated_refinement_patterns'],atol=1e-7)
def test_original_dictionary_sampling_and_local_proposals(stages):
    torch.manual_seed(20261021);q=sample_cubic_grid();np.testing.assert_array_equal(q,stages['dictionary_orientation_grid']);assert len(q)==102448
    np.testing.assert_array_equal(get_local_orientation_grid(2.,1,3),stages['local_grid']);np.testing.assert_array_equal(cubic_reduce(torch.from_numpy(stages['quaternion_product'])),stages['fundamental_zone'])
    # Source TorchScript promotes sqrt(1/2) from rounded FP32 even in FP64
    # group tensors. Runtime orientations are FP32, where these are bitwise equal.
    np.testing.assert_allclose(cubic_elements(),stages['cubic_symmetry'],rtol=0,atol=2e-8)
    np.testing.assert_array_equal(cubic_elements().float(),stages['cubic_symmetry'].astype(np.float32))
def test_original_covariance_steps(stages):
    a=torch.zeros(13,13);n=torch.tensor([0]);mean=torch.zeros(1,13)
    for i,x in enumerate(torch.split(torch.from_numpy(stages['covariance_data']),32)):
        update_covariance(a,n,mean,x);np.testing.assert_array_equal(a,stages['cov'+str(i)]);np.testing.assert_array_equal(mean,stages['mean'+str(i)])
def test_independent_covariance_matches_batch():
    torch.manual_seed(41);x=torch.randn(91,7,dtype=torch.float64);a=torch.zeros(7,7,dtype=torch.float64);n=torch.tensor([0]);mean=torch.zeros(1,7,dtype=torch.float64)
    # Original explicit update intentionally accumulates FP32 deltas; tolerance reflects this.
    for chunk in torch.split(x,13):update_covariance(a,n,mean,chunk)
    exact=np.cov(x.numpy(),rowvar=False);np.testing.assert_allclose(a.numpy()/(len(x)-1),exact,rtol=3e-6,atol=3e-7)
def test_independent_bruker_pixel_center_and_rigid_geometry():
    R,t=bruker_geometry_to_SE3(torch.tensor([[.4221,.2179,.4954]]),torch.tensor(-70.),torch.tensor(0.),(30,30));np.testing.assert_allclose((R@R.transpose(-1,-2))[0],np.eye(3),atol=1e-7);assert torch.det(R).item()==pytest.approx(1,abs=1e-6)
    center=torch.tensor([.2179*30-.5,.4221*30-.5,0.]);normal=R[0]@center+t[0];expected=.4954*30*np.array([np.cos(np.deg2rad(-70)),0,-np.sin(np.deg2rad(-70))]);np.testing.assert_allclose(normal,expected,atol=2e-6)
    r=detector_rays().reshape(30,30,3);np.testing.assert_allclose((r[1,0]-r[0,0]).norm(),1,atol=1e-6);np.testing.assert_allclose((r[0,1]-r[0,0]).norm(),1,atol=1e-6)
def test_independent_quaternion_rotation_matrix():
    theta=np.pi/3;q=torch.tensor([np.cos(theta/2),0,0,np.sin(theta/2)]);v=torch.tensor([1.,0,0],dtype=torch.float64);np.testing.assert_allclose(qu_apply(q,v),[np.cos(theta),np.sin(theta),0],atol=1e-15)
    a=torch.randn(11,4,dtype=torch.float64);a/=a.norm(dim=-1,keepdim=True);v=torch.randn(11,3,dtype=torch.float64);np.testing.assert_allclose(qu_apply(a,v).norm(dim=-1),v.norm(dim=-1),atol=1e-14)
def test_independent_cubic_group_proper_rotations():
    g=cubic_elements();assert g.shape==(24,4);np.testing.assert_allclose(g.norm(dim=-1),1,atol=1e-15);eye=torch.eye(3,dtype=torch.float64);mat=qu_apply(g[:,None],eye[None]).transpose(-1,-2);np.testing.assert_allclose(mat@mat.transpose(-1,-2),np.broadcast_to(np.eye(3),(24,3,3)),atol=1e-14);np.testing.assert_allclose(torch.linalg.det(mat),1,atol=1e-14)
def test_lambert_poles_and_hemispheres():
    rays=torch.tensor([[0.,0,1],[0,0,-1],[1,0,0],[0,1,0]]);uv=rosca_lambert_side_by_side(rays);np.testing.assert_allclose(uv[:2],[[-.5,0],[.5,0]],atol=1e-7);assert torch.isfinite(uv).all() and uv.abs().max()<=1
def test_complete_corrected_original_pipeline():
    raw,master=load_data(ROOT/'data/raw_data.npz');q,diag=reconstruct(raw,master,n_components=100,return_diagnostics=True)
    with np.load(ROOT/'test/fixtures/upstream_pipeline.npz') as z:
        a=q.reshape(-1,4).astype(float);b=z['refined_orientation'].astype(float);a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True);angles=2*np.rad2deg(np.arccos(np.clip(abs((a*b).sum(-1)),0,1)));assert angles.max()<.001 and angles.mean()<.0001
        np.testing.assert_allclose(abs(diag['components'].T@z['pca']),np.eye(100),atol=2e-5)
        np.testing.assert_allclose(diag['coarse'],z['coarse_orientation'],rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(diag['rays'],z['detector_coords'])
    with np.load(ROOT/'test/fixtures/upstream_refinement_check.npz') as z:
        assert z['center_component_changes'].shape==(8000,) and not z['center_component_changes'].any()
        np.testing.assert_allclose(diag['refinement_dots'],z['dot_trajectory'][:,1:],rtol=1e-5,atol=1e-5)
        np.testing.assert_array_equal(z['final_orientation'],np.load(ROOT/'test/fixtures/upstream_pipeline.npz')['refined_orientation'])

def test_complete_original_uncompressed_direct_dictionary_indexing():
    raw,master=load_data(ROOT/'data/raw_data.npz');p=prepare_patterns(raw);mp=prepare_master(master);rays=detector_rays()
    from src.physics_model import get_radial_mask
    torch.manual_seed(20261021);coarse,_=index_direct(mp,p,rays,get_radial_mask((30,30)).flatten());q,_=refine(mp,p,rays,coarse)
    with np.load(ROOT/'test/fixtures/upstream_direct.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(coarse,z['coarse']);np.testing.assert_array_equal(q,z['refined'])

def test_complete_selected512_original_pipeline():
    raw,master=load_data(ROOT/'data/raw_data.npz');q,diag=reconstruct(raw,master,return_diagnostics=True)
    with np.load(ROOT/'test/fixtures/upstream_pipeline512.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(q.reshape(-1,4),z['refined_orientation']);np.testing.assert_array_equal(diag['coarse'],z['coarse_orientation']);np.testing.assert_allclose(abs(diag['components'].T@z['pca']),np.eye(512),atol=2e-5)
        np.testing.assert_array_equal(diag['rays'],z['detector_coords'])
    with np.load(ROOT/'test/fixtures/upstream_refinement512_check.npz',allow_pickle=False) as z:
        assert z['center_component_changes'].shape==(8000,) and not z['center_component_changes'].any();np.testing.assert_allclose(diag['refinement_dots'],z['dot_trajectory'][:,1:],rtol=1e-5,atol=1e-5)

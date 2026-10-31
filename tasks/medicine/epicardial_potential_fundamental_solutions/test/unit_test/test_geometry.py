from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import virtual_sources
from src.visualization import rms
def winding(points,vertices,faces):
    tri=vertices[faces];out=[]
    for point in points:
        a,b,c=(tri[:,i]-point for i in range(3));la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
        num=np.einsum('ij,ij->i',a,np.cross(b,c))
        den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        out.append(np.sum(2*np.arctan2(num,den))/(4*np.pi))
    return np.asarray(out)
def test_closed_oriented_meshes_and_sources_outside_harmonic_domain():
    d=load_data(Path(__file__).resolve().parents[2]/'data/raw_data.npz');sources,_=virtual_sources(d)
    for name in ['heart','torso']:
        vertices=d[name+'_vertices'];faces=d[name+'_faces']
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        assert np.all(np.unique(edges,axis=0,return_counts=True)[1]==2)
        assert np.einsum('ij,ij->i',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()>0
    np.testing.assert_allclose(winding(sources[284:],d['heart_vertices'],d['heart_faces']),1,atol=1e-12)
    np.testing.assert_allclose(winding(sources[:284],d['torso_vertices'],d['torso_faces']),0,atol=1e-12)
def test_visual_metric_known_case():assert rms([3.,-3.])==3.

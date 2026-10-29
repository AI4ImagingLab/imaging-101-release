from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.physics_model import assemble_fem, MUAX,MUAM,KAPPAX,KAPPAM,refractive_robin
T=Path(__file__).resolve().parents[2]
def test_native_fem_operators():
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as raw,np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for name,mua,kappa in [('excitation_matrix',MUAX,KAPPAX),('emission_matrix',MUAM,KAPPAM)]:
            actual,mass,vol=assemble_fem(raw['nodes'][0],raw['tetrahedra'][0],mua,kappa)
            ref=csr_matrix((f[name+'_data'],f[name+'_indices'],f[name+'_indptr']),shape=actual.shape)
            assert np.linalg.norm((actual-ref).data)/np.linalg.norm(ref.data)<1e-12
        assert abs(refractive_robin()-.21292410845998552)<1e-15

def test_known_single_tetra_mass_and_robin_conservation():
    nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    elements=np.array([[0,1,2,3]])
    a,m,v=assemble_fem(nodes,elements,0.,0.,ksi=1.)
    assert np.allclose(m.toarray(),(np.ones((4,4))+np.eye(4))/120,rtol=0,atol=1e-15)
    assert abs(np.ones(4)@a@np.ones(4)-(1.5+np.sqrt(3)/2))<1e-14


def test_native_geometric_interpolation_and_forward_stages():
    from src.preprocessing import load_raw,geometry_operators
    from src.physics_model import continuum_prejacobian,forward_fluorescence
    from scipy.sparse.linalg import splu
    raw=load_raw(T/'data/raw_data.npz');source,detector,adj,C,B=geometry_operators(raw)
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for actual,key,elements,nodes in [(C,'fine_to_coarse',raw['tetrahedra'],raw['nodes']),(B,'coarse_to_fine',raw['coarse_tetrahedra'],raw['coarse_nodes'])]:
            mapping=f[key];inds=mapping[:,0].astype(int)-1;weights=mapping[:,1:]
            values=np.sin(nodes[:,0]*.31+nodes[:,1]*.17+nodes[:,2]*.09)
            reference=(values[elements[inds]]*weights).sum(axis=1)
            assert np.max(abs(actual@values-reference))<1e-9
        Ax,mass,vol=assemble_fem(raw['nodes'],raw['tetrahedra'],MUAX,KAPPAX)
        Am,_,_=assemble_fem(raw['nodes'],raw['tetrahedra'],MUAM,KAPPAM)
        phix=splu(Ax).solve(source.toarray()*np.cos(.15))
        assert np.linalg.norm(phix-f['phix'])/np.linalg.norm(f['phix'])<1e-10
        aphim=splu(Am.astype(complex)).solve(adj.toarray()*np.exp(-1j*.15))
        assert np.linalg.norm(aphim-f['adjoint_phim'])/np.linalg.norm(f['adjoint_phim'])<1e-10
        pre=continuum_prejacobian(raw['tetrahedra'],vol,phix,aphim,raw['links'])
        assert np.linalg.norm(pre-f['prejacobian'])/np.linalg.norm(f['prejacobian'])<1e-8
        calibrated=raw['fluorescence']*(detector@phix)[raw['links'][:,1],raw['links'][:,0]]/raw['excitation']
        assert np.linalg.norm(calibrated-f['calibrated_fluorescence'])/np.linalg.norm(f['calibrated_fluorescence'])<1e-10


def test_exact_discrete_chain_native_fd_and_inexact_source_distinction():
    from src.preprocessing import load_raw,geometry_operators
    from scipy.sparse.linalg import splu
    raw=load_raw(T/'data/raw_data.npz');source,detector,adj,C,B=geometry_operators(raw)
    Am,mass,vol=assemble_fem(raw['nodes'],raw['tetrahedra'],MUAM,KAPPAM)
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f,np.load(T/'test/fixtures/native_directional_physics.npz',allow_pickle=False) as q:
        a=splu(Am).solve(adj.toarray());chain=np.asarray([f['phix'][:,s]*(mass@a[:,d]) for s,d in raw['links']]);actual=chain@q['direction']
        assert np.linalg.norm(actual-q['native_central_FD'])/np.linalg.norm(q['native_central_FD'])<1e-8
        continuum=f['prejacobian'].real@q['direction'];assert np.allclose(continuum,q['native_continuum_direction'],rtol=1e-10,atol=1e-16)
        # The published/source inexact method is deliberately not relabeled as
        # the exact derivative. Test a measurable discrepancy independently.
        assert np.linalg.norm(continuum-actual)/np.linalg.norm(actual)>.005

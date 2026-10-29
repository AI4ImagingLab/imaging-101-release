import numpy as np,itertools
from src.preprocessing import parameters
from src.physics_model import model_matrix,pseudoinverse_and_projection_matrices,modulation_vectors
from src.solvers import QPBO,get_updates,downsample_residual,upsample_B0
def sample():return parameters({'echoes':np.ones((1,3,101,101,4),dtype=np.complex64)})
def test_complex_signal_model_and_units():
    d,a,m=sample();A=model_matrix(d,m,40.)
    t=np.array([.00287,.00607,.00927]);cs=m.CS.astype(np.float64)
    direct=np.zeros((3,2),dtype=complex)
    for echo in range(3):
        for component in range(2):
            for peak in range(7):direct[echo,component]+=float(m.alpha[component,peak])*np.exp(1j*2*np.pi*42.576*1.494*(cs[peak]-cs[0])*t[echo]-40*(t[echo]-t[0]))
    np.testing.assert_allclose(A,direct,rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(A[:,0],np.exp(-40*(t-t[0])),rtol=1e-7)
def test_projection_hermitian_idempotent_and_complex_amplitude_recovery():
    d,a,m=sample();pinv,null=pseudoinverse_and_projection_matrices(d,a,m)
    for r,b in [(0,0),(40,11),(144,99)]:
        P=null[r,b];np.testing.assert_allclose(P,P.conj().T,atol=2e-7)
        np.testing.assert_allclose(P@P,P,atol=2e-7)
        matrix=modulation_vectors(100,3)[b]@model_matrix(d,m,r)
        components=np.array([1.2+.2j,.3-.1j]);signal=matrix@components
        np.testing.assert_allclose(pinv[r,b]@signal,components,rtol=1e-6,atol=1e-7)
        assert np.linalg.norm(P@signal)<1e-6
def test_qpbo_generic_energy_matches_exhaustive_minimum():
    D=np.array([[0.,.2,.7],[.9,.4,0.]])
    ngb=np.array([[1,2,2]]);V=np.zeros((4,1,3));V[:,0,0]=[0,.5,.3,0];V[:,0,1]=[0,.2,.2,0]
    label=QPBO(D,V,ngb).astype(int);assert np.all(label>=0)
    def cost(x):return sum(D[x[k],k] for k in range(3))+sum(V[2*x[k]+x[ngb[0,k]],0,k] for k in range(3))
    assert cost(label)==min(cost(x) for x in itertools.product([0,1],repeat=3))

def test_spatial_candidate_endpoints_all_four_states(monkeypatch):
    from types import SimpleNamespace
    from src import solvers
    A=np.array([0,1]);B=np.array([2,4]);ngb=np.array([[1,1]]);D=np.array([[0.,2.],[100.,0.]])
    V=np.zeros((4,1,2));V[:,0,0]=[1,16,1,4]
    edges=[]
    class Recorder:
        def add_node(self,count):pass
        def add_unary_term(self,*args):pass
        def add_pairwise_term(self,i,j,*table):edges.append((i,j,np.array(table)))
        def solve(self):pass
        def get_label(self,q):return 0
    monkeypatch.setattr(solvers,'tq',SimpleNamespace(QPBOFloat=Recorder))
    solvers.QPBO(D,V,ngb)
    assert len(edges)==1
    i,j,table=edges[0]
    for x in itertools.product([0,1],repeat=2):
        physical=(np.array([A[0],B[0]])[x[0]]-np.array([A[1],B[1]])[x[1]])**2
        assert table[2*x[i]+x[j]]==physical

def test_asymmetric_candidates_obtain_physical_global_minimum():
    D=np.array([[0.,2.],[100.,0.]]);V=np.zeros((4,1,2));V[:,0,0]=[1,16,1,4]
    labels=QPBO(D,V,np.array([[1,1]])).astype(int)
    np.testing.assert_array_equal(labels,[0,0])
def test_update_order_and_multiscale_block_conservation():
    assert get_updates(2)==[0,-1,1,-2,2]
    J=np.arange(2*4*4,dtype=float).reshape(2,-1);coarse=downsample_residual(J,(2,2,1),(4,4,1))
    np.testing.assert_allclose(coarse.sum(axis=1)*4,J.sum(axis=1))
    np.testing.assert_array_equal(upsample_B0(np.array([0,1,2,3]),(2,2,1),(4,4,1)),np.repeat(np.repeat(np.array([[0,1],[2,3]]),2,0),2,1).ravel())

def test_predict_echoes_positive_precession_and_first_echo_amplitude():
    from src.physics_model import predict_echoes
    d,a,m=sample();coefficients=np.array([1.2+.3j,0],dtype=np.complex64).reshape(2,1,1,1)
    value=predict_echoes(coefficients,np.ones((1,1,1)),np.full((1,1,1),10.),d,m).ravel()
    relative=np.array([0.,.0032,.0064]);expected=(1.2+.3j)*np.exp(-10*relative+1j*2*np.pi*42.576*1.494*relative)
    np.testing.assert_allclose(value,expected,rtol=1e-6,atol=1e-7)

def test_visualization_metric_known_case_and_readme_acquisition():
    from src.visualization import metrics_for_display
    from pathlib import Path
    reference={key:np.array([1.,2.,3.]) for key in ['water','fat','fat_fraction','fieldmap','r2star']}
    estimate={key:value*.5 for key,value in reference.items()}
    score=metrics_for_display(estimate,reference)
    np.testing.assert_allclose(score['water_nrmse'],np.sqrt((.25+1+2.25)/3)/2)
    assert score['fat_dominance_disagreement_percent']==0
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    for literal in ['0.00287,0.00607,0.00927','1.494','42.576','1.5,1.5,5.0','4.64655','0.048,0.039,0.004,0.128,0.693,0.087']:assert literal in text

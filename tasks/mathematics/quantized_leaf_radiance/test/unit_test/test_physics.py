import numpy as np
from src.physics_model import gradient,divergence,symmetric_gradient,symmetric_divergence,quantization_intervals
from src.solvers import midpoint_prox,project_vector,project_tensor,objective,reconstruct

def test_gradient_adjoint():
    rng=np.random.default_rng(56);u=rng.normal(size=(7,9,3));p=rng.normal(size=(7,9,3,2))
    np.testing.assert_allclose(np.sum(gradient(u)*p),-np.sum(u*divergence(p)),rtol=3e-15,atol=3e-14)

def test_weighted_symmetric_adjoint():
    rng=np.random.default_rng(92);w=rng.normal(size=(7,9,3,2));q=rng.normal(size=(7,9,3,3));weights=np.array([1,1,2])
    np.testing.assert_allclose(np.sum(symmetric_gradient(w)*q*weights),-np.sum(w*symmetric_divergence(q)),rtol=2e-14,atol=2e-14)

def test_constant_affine_and_boundary_conventions():
    assert np.count_nonzero(gradient(np.ones((5,7,3))))==0
    assert np.count_nonzero(symmetric_gradient(np.ones((5,7,3,2))))==0
    rr,cc=np.indices((5,7));u=(2*rr+3*cc)[...,None].astype(float)
    np.testing.assert_array_equal(gradient(u)[:-1,:-1,0],np.broadcast_to([2.,3.],(4,6,2)))
    assert np.count_nonzero(gradient(u)[-1,...,0])==0 and np.count_nonzero(gradient(u)[:,-1,...,1])==0

def test_exact_quantizer_and_generator(root):
    from src.generate_data import generate_observations
    with np.load(root/'evaluation/ground_truth.npz',allow_pickle=False) as f:truth=f['reconstruction'][0]
    generated=generate_observations(truth)
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as f:
        for k,v in generated.items():np.testing.assert_array_equal(v,f[k])
    lo,hi=quantization_intervals(np.array([0,.5,1.],dtype=np.float32))
    np.testing.assert_array_equal(lo,hi)

def test_proximal_scalar_solution():
    lo=np.array([.2,.4,.7],dtype=np.float32);hi=np.array([.3,.6,.9],dtype=np.float32);value=np.array([0.,.55,1.2],dtype=np.float32);tau=np.float32(.3)
    actual=midpoint_prox(value,lo,hi,tau)
    independent=np.maximum(lo,np.minimum(hi,(value+tau*(lo+hi)/2)/(1+tau)))
    np.testing.assert_allclose(actual,independent,rtol=1e-7,atol=1e-7)

def test_dual_projection_norms():
    rng=np.random.default_rng(8);p=rng.normal(size=(3,4,3,2)).astype(np.float32);q=rng.normal(size=(3,4,3,3)).astype(np.float32)
    p=project_vector(p,.45);q=project_tensor(q,.9)
    assert np.sqrt(np.sum(p*p,axis=(-2,-1))).max()<.450001
    assert np.sqrt(np.sum(q[...,0]**2+q[...,1]**2+2*q[...,2]**2,axis=-1)).max()<.900001

def test_midpoint_fidelity_and_interval_indicator():
    lo=np.zeros((3,4,3),np.float32);hi=np.ones_like(lo);u=np.full_like(lo,.5)
    assert objective(u,lo,hi)==0
    assert np.isinf(objective(u+1,lo,hi))
    out,_=reconstruct(lo,hi,iterations=30);np.testing.assert_array_equal(out,u)

def test_known_metrics(root):
    import importlib.util
    spec=importlib.util.spec_from_file_location('metric_judge',root/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    values=m.compare(np.array([1.,1.,1.]),np.array([0.,1.,2.]))
    assert abs(values['ncc']-3/np.sqrt(15))<1e-15
    assert abs(values['nrmse']-np.sqrt(2/3)/2)<1e-15

def test_visualization_metrics_same_known_case():
    from src.visualization import metrics
    a=np.array([1.,1.,1.]);b=np.array([0.,1.,2.])
    result=metrics(a,b,np.zeros(3),np.ones(3)*2)
    assert abs(result['ncc']-3/np.sqrt(15))<1e-15
    assert abs(result['nrmse']-np.sqrt(2/3)/2)<1e-15
    assert abs(result['psnr_db']+20*np.log10(np.sqrt(2/3)))<1e-14
    assert result['interval_violation']==0
    assert metrics(b,b,np.zeros(3),np.ones(3)*2)['psnr_db'] is None

def test_readme_constants(root):
    s=(root/'README.md').read_text(encoding='utf8')
    assert '(1, 512, 512, 3)' in s and '| Bin count B | 6 |' in s and 'float32' in s

from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from src import physics_model as p
from src.preprocessing import load_data,estimate_resolution
from src.solvers import objective,initial_hypotheses,select_model
T=Path(__file__).resolve().parents[2]

def fixture():return np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False)

def test_original_operators_and_resolution():
    z=fixture();assert_array_equal(p.OPERATORS,z['Er__O']);assert_array_equal(p.JX,z['Jx']);assert_array_equal(p.JY_IMAG,z['Jy_imag']);assert_array_equal(p.JZ,z['Jz'])
    width,d=estimate_resolution(load_data());assert_array_equal(width,z['width_popt']);assert_array_equal(d['voigt_parameters'],z['poptinitial']);assert_array_equal(d['voigt_covariance'],z['pcovinitial'])
    assert_array_equal(d['width_covariance'],z['width_pcov'])

def test_original_all_eigen_transition_spectrum_stages():
    z=fixture();d=load_data()
    for label in ['initial_model1','initial_model2','final_model1','final_model2']:
        q=z[label+'__parameters'];H,e,v=p.diagonalize(q[:6]);assert_array_equal(H,z[label+'__hamiltonian']);assert_array_equal(e,z[label+'__energies']);assert_array_equal(v,z[label+'__eigenvectors'])
        for temperature in [1.8,15,100]:
            populations=np.exp(-e/(.0861733*temperature));populations/=sum(populations)
            assert_array_equal(populations,z[label+f'__population_{temperature}'])
        assert_array_equal(np.array([[p.transition(w.real,x.real) for x in v] for w in v]),z[label+'__transition'])
        assert_array_equal(p.predict(q,d['energy'],d['temperature'],z['width_popt']),z[label+'__spectra'])

def test_original_first_40_optimizer_evaluations():
    z=fixture();d=load_data();initial=initial_hypotheses()
    for k in range(3):
        for x,expected in zip(z[f'optimizer{k}__first_x'],z[f'optimizer{k}__first_loss']):
            q=initial[0].copy() if k==0 else x.copy()
            if k==0:q[:7]=x
            assert objective(q,d,z['width_popt'])==expected

def test_original_magnetization_and_selection():
    z=fixture();d=load_data();params=[]
    for k in [1,2]:
        q=z[f'final_model{k}__parameters'];params.append(q)
        for fi in [0,14,17,50,77]:
            for axis in range(3):
                field=np.eye(3)[axis]*z['fieldstrengths'][fi]
                assert_allclose(p.magnetization(q[:6],.5,field),z[f'magnetization{k}'][fi,axis],atol=2e-13,rtol=2e-13)
    selected,diag=select_model(np.array(params),d);assert selected==0;assert diag['observed_moment_1T'][0]>diag['observed_moment_1T'][1]

def test_angular_momentum_sum_rules_kramers_and_gauge():
    z=fixture();jx,jyi,jz=p.JX,p.JY_IMAG,p.JZ;jy=1j*jyi
    assert_allclose(jx@jy-jy@jx,1j*jz,atol=1e-14)
    assert_allclose(jx@jx+jy@jy+jz@jz,p.J*(p.J+1)*np.eye(16),atol=1e-14)
    U=np.diag(np.exp(-1j*np.pi/3*np.arange(-p.J,p.J+1)))
    for k in [1,2]:
        q=z[f'final_model{k}__parameters'];H,e,v=p.diagonalize(q[:6]);assert_allclose(e[::2],e[1::2],atol=4e-14)
        assert_allclose(np.sum(z[f'final_model{k}__transition'],axis=1),p.J*(p.J+1),atol=1e-12)
        flip=q[:6].copy();flip[[2,4]]*=-1;other,ee,_=p.diagonalize(flip)
        assert_allclose(other,U@H@U.conj().T,atol=1e-14);assert_allclose(ee,e,atol=3e-14)
        assert_allclose(p.magnetization(q[:6],.5,[0,0,0]),0,atol=1e-11)

def test_finite_domain_signed_width_physical_extension():
    z=fixture();d=load_data();a,b=z['width_popt']
    for k in [1,2]:
        q=z[f'final_model{k}__parameters'];_,e,_=p.diagonalize(q[:6])
        assert np.all(a*e[2:10:2]+b>0)
        for ti,temp in enumerate(d['temperature']):
            E=d['energy'][ti]*(1+temp*q[14])-q[10];weights=np.exp(-e/(.0861733*temp));weights/=sum(weights)
            for i in np.flatnonzero(weights>.001):
                for j in range(16):
                    de=e[j]-e[i];alpha=a*de+b
                    if alpha<0:
                        sigma=alpha/(2*np.sqrt(2*np.log(2)));arg=((E-de)+.5j*q[7+ti])/(sigma*np.sqrt(2))
                        assert np.max((-arg*arg).real)<-745
                        assert_array_equal(p.voigt(E,de,alpha,q[7+ti]),p.voigt(E,de,-alpha,q[7+ti]))
        assert np.min(z[f'final_model{k}__spectra'])>=0

def test_full_executed_pipeline_artifact():
    # This checks the recorded full CLI, rather than running another full fit.
    import json
    execution=json.loads((T/'notebooks/reference_outputs/execution.json').read_text(encoding='utf8'))
    assert all(s['success'] and s['nfev']>1000 for s in execution['stages'])
    z=fixture();fresh=np.load(T/'notebooks/reference_outputs/fresh_measurements.npz',allow_pickle=False)
    assert float(fresh['runtime_seconds'])==execution['runtime_seconds']
    assert_array_equal(fresh['initial_parameters'],initial_hypotheses())
    for k in range(3):
        assert_array_equal(fresh[f'trace{k}_x'],z[f'optimizer{k}__first_x'])
        assert_array_equal(fresh[f'trace{k}_loss'],z[f'optimizer{k}__first_loss'])
    assert_array_equal(fresh['warmup_parameters'][:6],z['warmup__coeff'])
    assert_array_equal(fresh['warmup_parameters'][6:7],z['warmup__pref'])
    assert_allclose(fresh['reconstruction'][0],np.stack([z[f'final_model{i}__parameters'] for i in [1,2]]),rtol=5e-12,atol=1e-15)

def test_original_ground_doublet_g_tensors():
    z=fixture();g=np.load(T/'test/fixtures/upstream_gtensors.npz',allow_pickle=False)['gtensors']
    for k in [1,2]:assert_array_equal(p.gtensor(z[f'final_model{k}__parameters'][:6]),g[k-1])

def test_g_tensor_matches_independent_small_field_zeeman_splitting():
    z=fixture();field=1e-5;mu=.057883818012
    for k in [1,2]:
        q=z[f'final_model{k}__parameters'];H,_,_=p.diagonalize(q[:6]);g=p.gtensor(q[:6])
        for a,A in enumerate([p.JX,p.JY_IMAG*1j,p.JZ]):
            energies=np.linalg.eigvalsh(H+1.2*mu*field*A)
            measured=(energies[1]-energies[0])/(mu*field)
            assert_allclose(measured,np.linalg.norm(g[a]),atol=2e-6,rtol=2e-6)

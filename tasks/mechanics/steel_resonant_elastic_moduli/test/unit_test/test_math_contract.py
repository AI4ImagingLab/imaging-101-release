from pathlib import Path
import importlib.util,itertools,json,subprocess,sys
import numpy as np
import pytest
from src.physics_model import moment_table,basis_blocks,operators,resonances,DIMENSIONS_CM,assemble_block,DENSITY_SCALED,MASS_G
from src.solvers import match_modes,objective
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_volume_and_even_moments():
    a,b,c=DIMENSIONS_CM/2;t=moment_table(3)
    assert t[0,0,0]==pytest.approx(2*np.pi*a*b*c,rel=1e-14)
    assert t[2,0,0]/t[0,0,0]==pytest.approx(a*a/4)
    assert t[0,0,2]/t[0,0,0]==pytest.approx(c*c/3)
    assert np.all(t[1::2]==0)
    assert t[0,0,0]*DENSITY_SCALED/1000==pytest.approx(MASS_G)

def test_tensor_symmetry_positive_energy_and_length_scaling():
    from scipy.linalg import eigvalsh
    identity=np.eye(3);lam=2.7-2*.8;mu=.8
    tensor=lam*np.einsum('ij,kl->ijkl',identity,identity)+mu*(np.einsum('ik,jl->ijkl',identity,identity)+np.einsum('il,jk->ijkl',identity,identity))
    np.testing.assert_array_equal(tensor,tensor.transpose(1,0,2,3))
    np.testing.assert_array_equal(tensor,tensor.transpose(2,3,0,1))
    strain=np.array([[.2,.1,-.3],[.1,.5,.2],[-.3,.2,-.1]])
    assert np.einsum('ij,ijkl,kl',strain,tensor,strain)>0
    spectra=[]
    for scale in [1.,2.]:
        moments=moment_table(3,DIMENSIONS_CM*scale);values=[]
        for basis in basis_blocks(3):
            mass,a,b=assemble_block(basis,moments)
            values.extend(eigvalsh(2.7*a+.8*b,mass))
        spectra.append(np.sort(values)[6:])
    np.testing.assert_allclose(spectra[1],spectra[0]/4,rtol=1e-10,atol=1e-11)

def test_complete_basis_rigid_and_stiffness_scaling():
    blocks=basis_blocks(4)
    assert sum(map(len,blocks))==3*35
    values=np.sort(np.concatenate([np.linalg.eigvalsh(2.7*a+.8*b) for a,b in operators(4)]))
    assert np.max(abs(values[:6]))<1e-11
    assert values[6]>0
    np.testing.assert_allclose(resonances([5.4,1.6],4,20),np.sqrt(2)*resonances([2.7,.8],4,20),rtol=1e-11)

def test_matching_independent_exhaustive():
    measured=np.array([1.1,2.9,5.2]);calculated=np.array([1.,2.,3.,4.,5.])
    candidates=list(itertools.combinations(range(5),3))
    expected=min(candidates,key=lambda p:np.sum((measured/calculated[list(p)]-1)**2))
    np.testing.assert_array_equal(match_modes(measured,calculated),expected)
    np.testing.assert_array_equal(judge.pair(measured,calculated),expected)

def test_objective_and_all_metrics_known():
    assert objective(np.array([1.1,2.]),np.array([1.,2.]),np.array([.5,0.]))==pytest.approx(50.)
    ref={'reconstruction':np.array([[200.,80.]]),'resonances':np.arange(1.,77.)[None]}
    pred={k:v.copy() for k,v in ref.items()};pred['reconstruction']+=np.array([[.001,.0005]])
    cfg={'moduli_absolute_boundary_GPa':.002,'frequency_absolute_boundary_MHz':3e-6,'observation_rms_percent_boundary':1.}
    metrics,_=judge.score(pred,ref,cfg,np.array([1.1,3.]),np.array([.5,0.]))
    assert metrics['c11_absolute_error_GPa']==pytest.approx(.001)
    assert metrics['c44_absolute_error_GPa']==pytest.approx(.0005)
    assert metrics['maximum_frequency_error_MHz']==0
    assert metrics['observation_rms_percent']==pytest.approx(100*np.sqrt(.005))

def invoke(pred,report,cwd=None):
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=cwd,capture_output=True,text=True)
    return result,json.loads(report.read_text()) if report.is_file() else None

@pytest.mark.parametrize('bad',['missing','key','shape','dtype','nan','inf','complex','poor'])
def test_judge_rejections(tmp_path,bad):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:data={k:f[k] for k in f.files}
    p=tmp_path/'pred.npz'
    if bad=='key':del data['reconstruction']
    if bad=='shape':data['reconstruction']=data['reconstruction'].ravel()
    if bad=='dtype':data['reconstruction']=data['reconstruction'].astype('float32')
    if bad=='complex':data['reconstruction']=data['reconstruction'].astype('complex128')
    if bad in ['nan','inf']:data['reconstruction'][0,0]=float(bad)
    if bad=='poor':data={k:v*.5 for k,v in data.items()}
    if bad!='missing':np.savez(p,**data)
    result,report=invoke(p,tmp_path/'nested/report.json',tmp_path)
    assert result.returncode==(1 if bad=='poor' else 2)
    assert not report['passed']
    if bad!='poor':assert report['metrics']=={} and report['errors']

def test_judge_reference_precedence_assets_and_write(tmp_path):
    import shutil
    evaluation=tmp_path/'evaluation';shutil.copytree(ROOT/'evaluation',evaluation)
    shutil.copytree(ROOT/'data',tmp_path/'data')
    with np.load(evaluation/'baseline_reference.npz') as f:d={k:f[k] for k in f.files}
    np.savez(tmp_path/'pred.npz',**d)
    np.savez(evaluation/'ground_truth.npz',reconstruction=d['reconstruction']+1,resonances=d['resonances'])
    command=[sys.executable,str(evaluation/'judge.py'),'--prediction',str(tmp_path/'pred.npz'),'--report',str(tmp_path/'r.json')]
    assert subprocess.run(command,capture_output=True).returncode==1
    (evaluation/'ground_truth.npz').unlink();(evaluation/'baseline_reference.npz').unlink()
    assert subprocess.run(command,capture_output=True).returncode==2
    result=subprocess.run(command[:-1]+[str(tmp_path)],capture_output=True)
    assert result.returncode==2 and b'Report-write failure' in result.stderr

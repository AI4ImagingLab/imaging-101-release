import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import nnls,minimize_scalar
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('feasible_judge',T/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_independent_full_nnls_feasible_candidate(tmp_path):
    context=j.observation_context();phix,M,lu,detector,links,observed=context
    # Independent discrete adjoint forms all 56x5707 sensitivities, without solver code.
    adjoint=lu.solve(detector.T,trans='T')
    sensitivity=np.array([phix[:,s]*(M.T@adjoint[:,d]) for s,d in links])
    baseline=np.load(T/'output/reconstruction.npz')['reconstruction'][0]
    assert np.allclose(sensitivity@baseline,j.forward_prediction(baseline,context),rtol=1e-12,atol=1e-20)
    candidate,residual=nnls(sensitivity/np.linalg.norm(sensitivity,axis=1,keepdims=True),observed/np.linalg.norm(sensitivity,axis=1),maxiter=20000)
    h0=sensitivity@baseline;h1=sensitivity@candidate
    def objective(t):return np.mean((np.log((1-t)*h0+t*h1)-np.log(observed))**2)
    result=minimize_scalar(objective,bounds=(0,1),method='bounded')
    assert result.fun<objective(0)
    field=(1-result.x)*baseline+result.x*candidate
    path=tmp_path/'independent.npz';np.savez(path,reconstruction=field[None])
    report=j.evaluate(path);assert report['passed'],report
    import json
    (tmp_path/'independent_controls.json').write_text(json.dumps(dict(baseline_log_mse=objective(0),candidate_log_mse=result.fun,blend=result.x,report=report),indent=2),encoding='utf8')
    np.savez(path,reconstruction=(10*field)[None]);assert j.evaluate(path)['status']=='fail'
    np.savez(path,reconstruction=field[::-1][None]);assert j.evaluate(path)['status']=='fail'

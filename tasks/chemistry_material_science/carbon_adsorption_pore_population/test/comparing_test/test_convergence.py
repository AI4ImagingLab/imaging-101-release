from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np,pytest
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load_observations
from src.solvers import reconstruct,convergence_certificate
from src.physics_model import build_kernel
def test_all_full_public_starts_certified_against_independent_source():
 raw=load_observations(T/'data/raw_data.npz')
 with np.load(T/'test/fixtures/upstream_analytic_converged.npz') as z:
  for start in [np.zeros(77),np.full(77,.005),np.full(77,.01)]:
   fit=reconstruct(raw,initial=start);certificate=fit['convergence_certificate'];assert certificate['projected_kkt']<1e-11;assert abs(certificate['relative_primal_dual_gap'])<1e-6
   assert abs(fit['objective']-20.970090134873)<1e-8
   assert np.sqrt(np.mean((fit['bin_weights']-z['coefficients'])**2))/np.ptp(z['coefficients'])<1e-4
def test_reported_success_without_first_order_convergence_rejected(monkeypatch):
 import src.solvers as solver
 raw=load_observations(T/'data/raw_data.npz')
 monkeypatch.setattr(solver,'minimize',lambda *a,**k:SimpleNamespace(success=True,message='False success regression',x=np.full(77,.01),nit=5,fun=3182.))
 with pytest.raises(RuntimeError,match='without convex convergence'):solver.reconstruct(raw,initial=np.full(77,.01))
def test_certificate_independent_known_convex_case():
 K=np.eye(2);x=np.array([2.,3.]);y=x.copy();c=convergence_certificate(x,K,y);assert c['projected_kkt']==0.;assert abs(c['relative_primal_dual_gap'])<1e-12
 bad=convergence_certificate(np.ones(2),K,y);assert bad['projected_kkt']>=1.

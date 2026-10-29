import numpy as np
from scipy.optimize import minimize_scalar
from test_judge import j,baseline,write

def test_independent_exact_profile_and_improved_candidate(tmp_path):
    fields=baseline();V=fields['mixing_weights'][0];assets=j.profile_assets();U,X,precisions=assets
    # Independent direct residual posterior, without using Judge profile routine.
    def direct(scale):
        W=scale*V;n=len(W);residual=X-U@W.T;variance=np.sum(residual**2)/X.size
        scores=[]
        for L,normalizer in precisions:
            intercept=(L.sum(0)@W)/L.sum();Z=W-intercept
            q=np.einsum('ik,ij,jk->k',Z,L,Z);lam=(q+n)/(n+4)
            scores.append(-X.size/2*np.log(variance)-np.sum(residual**2)/(2*variance)-n/2*np.log(lam).sum()-.5*np.sum(q/lam)-2*np.log(lam).sum()-np.sum(n/2/lam)+normalizer)
        return -max(scores)/X.size
    assert abs(direct(1)+j.profile_score(V,assets))<1e-12
    optimum=minimize_scalar(direct,bounds=(.9,1.1),method='bounded',options={'xatol':1e-12})
    assert optimum.fun<=direct(1)+1e-12
    fields['mixing_weights']*=optimum.x
    fields['cell_type_proportions']=fields['mixing_weights']/fields['mixing_weights'].sum(2,keepdims=True)
    report,code=j.evaluate(write(tmp_path,fields));assert code==0
    import json
    (tmp_path/'independent_controls.json').write_text(json.dumps(dict(baseline_negative_profile_score=direct(1),candidate_negative_profile_score=optimum.fun,scale=optimum.x,report=report),indent=2),encoding='utf8')
    fields['mixing_weights']*=10
    fields['cell_type_proportions']=fields['mixing_weights']/fields['mixing_weights'].sum(2,keepdims=True)
    report,code=j.evaluate(write(tmp_path,fields));assert code==1 and report['metrics']['profile_nll_excess']>1e-6

def test_analytic_nuisance_profile_is_local_maximum():
    rng=np.random.default_rng(42);V=rng.uniform(.2,.6,(7,3));X=rng.uniform(.2,.4,(9,7));U=rng.uniform(.1,.2,(9,3))
    L=np.diag(np.full(7,7.))-np.full((7,7),.5);Z=V-(L.sum(0)@V)/L.sum();q=np.sum(Z*(L@Z),0);lam=(q+7)/11
    b=(L.sum(0)@V)/L.sum();s=np.mean((X-U@V.T)**2)
    def score(b,s,lam):
        Z=V-b;return -X.size/2*np.log(s)-np.sum((X-U@V.T)**2)/(2*s)-(7/2+2)*np.log(lam).sum()-.5*np.sum((np.sum(Z*(L@Z),0)+7)/lam)
    at=score(b,s,lam)
    assert at>score(b+.01,s,lam)
    assert at>score(b,1.01*s,lam)
    assert at>score(b,s,1.01*lam)

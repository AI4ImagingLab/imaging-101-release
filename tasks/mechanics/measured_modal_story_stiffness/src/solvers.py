"""Bounded multi-start nonlinear least squares with explicit analytical residuals."""
import numpy as np
from scipy.optimize import least_squares
from .preprocessing import normalize_modes
from .physics_model import eigenmodes,residual_jacobian
STARTS=np.array([[0.9120685437784988, 0.8956549741186988, -0.8868972645463826, -0.8302560096821567], [0.6709977562588991, 0.4719399781370466, 0.3394608028804418, -0.3837270848217116], [0.21188833135692486, 0.21360346728167579, 0.16240803422400618, -0.6832342594903889], [-0.13866071941746272, -0.21293635958925727, 0.4460241624749317, 0.9896391258994854], [0.8987909461864871, 0.08835409485864165, -0.11029162254829283, -0.4635185167013438], [-0.9281513412142848, -0.945110285818362, -0.07021227580537581, -0.36306974429264516], [-0.23997015619857676, 0.7835789156565749, 0.05150553829205662, 0.12102072205299774], [-0.5277531857698758, -0.9522838417184356, -0.34971414247767996, -0.7266052140270667], [0.020447691674402302, 0.997367136385104, 0.3489593946917402, -0.6363130063537112], [0.787143073165977, 0.5935198428432791, 0.4688033837879555, 0.8131872997951219], [0.525770967666142, 0.5794952749235265, -0.292426044316793, 0.9619531461442532], [0.9238018757964515, -0.6776306933919622, 0.5080081433037442, 0.43030179647490696]])
def initial_state(alpha,measured,form):
    if form!=3:return np.asarray(alpha).copy()
    # Original initialization normalizes predicted modes at their own measured maxima.
    _,v=eigenmodes(alpha,np.array([0,0]));q=np.argmax(abs(v[:3]),axis=0);v/=v[q,np.arange(2)]
    return np.r_[alpha,v[3]]
def solve(frequency,measured,form=3,starts=None):
    """Return best state and all12 diagnostic optimizer runs; units as raw observations."""
    starts=STARTS if starts is None else np.asarray(starts);runs=[]
    bounds=np.r_[np.ones(4),np.ones(2)*3] if form==3 else np.ones(4)
    for alpha in starts:
        x0=initial_state(alpha,measured,form)
        result=least_squares(lambda x:residual_jacobian(x,frequency,measured,form)[0],x0,jac=lambda x:residual_jacobian(x,frequency,measured,form)[1],bounds=(-bounds,bounds),method='trf',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=1000)
        r,J=residual_jacobian(result.x,frequency,measured,form)
        runs.append(dict(x=result.x,objective=float(r@r),gradient=2*J.T@r,status=int(result.status),evaluations=result.nfev,initial=x0))
    best=min(runs,key=lambda d:d['objective'])
    return best['x'].copy(),runs
def reconstruct(frequency,measured):
    x,runs=solve(frequency,measured)
    return dict(reconstruction=(10*(1+x[:4]))[None],unmeasured_modes=x[4:][None,None,:]),runs

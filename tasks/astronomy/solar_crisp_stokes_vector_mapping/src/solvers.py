"""Original bounded, diagonally damped Levenberg–Marquardt and two-cycle flow.
MIT adaptation: J. de la Cruz Rodriguez, ISP-SU2020. No domain inverse calls.
"""
import numpy as np
from scipy.linalg import qr, solve_triangular
from .physics_model import synthesize_response

SCALE=np.array([1000.,1.,1.,1.,.05,10.,.1,1.,1.])
LOW=np.array([0.,0.,0.,-13.,.001,.01,1e-5,0.,-10.])
HIGH=np.array([6000.,np.pi,np.pi,13.,.2,120.,10.,10.,10.])
INITIAL=np.array([1500.,2.2,1.,-.5,.035,50.,.1,.24,.7])

def pivoted_qr_step(a,b):
    """Column-pivoted Householder QR with Eigen's numerical nonzero-pivot rule.

    LAPACK GELSY instead estimates the triangular condition number and returns
    a minimum-norm rank-deficient solution; that is a different inverse update.
    Eigen solve uses nonzeroPivots(), not rank(): its setThreshold(1e-9) does
    not truncate the solve. Dropped pivot variables are zero.
    """
    q,r,p=qr(a,mode='economic',pivoting=True)
    pivots=np.abs(np.diag(r)); limits=np.linalg.norm(a,axis=0).max()*np.finfo(float).eps*np.sqrt(np.arange(len(b),0,-1)/len(b))
    rank=int(np.count_nonzero(pivots>=limits))
    permuted=np.zeros(len(b))
    if rank: permuted[:rank]=solve_triangular(r[:rank,:rank],(q.T@b)[:rank])
    step=np.zeros(len(b)); step[p]=permuted
    return step

def clip_parameters(model,cyclic=True):
    model=model.copy()
    if cyclic:
        model[:,2]=np.where(model[:,2]>np.pi,model[:,2]-float(np.float32(3.1415926)),model[:,2])
        model[:,2]=np.where(model[:,2]<0,model[:,2]+float(np.float32(3.1416026)),model[:,2])
    return np.clip(model,LOW,HIGH)

def lm_block(initial,observed,sigma,regions,psfs,max_iterations=25,mu=.93):
    """Fit an independent pixel block, retaining per-pixel damping/rejection states."""
    n=len(initial); c=clip_parameters(initial,cyclic=False)/SCALE
    normalization=np.sqrt(np.count_nonzero(sigma<1e20))
    weight=1/(sigma*normalization)
    syn,response=synthesize_response(c*SCALE,regions,psfs,mu)
    residual=(observed-syn)*weight
    chi=np.sum(residual**2,axis=(1,2))
    best_syn=syn.copy(); derivatives=response.copy()
    damping=np.full(n,np.sqrt(10.)); rejected=np.zeros(n,int); accepted=np.zeros(n,int)
    small=np.zeros(n,bool); active=np.ones(n,bool); history=[]
    while np.any(active):
        ids=np.flatnonzero(active)
        jac=(derivatives[ids]*SCALE[None,:,None,None]*weight[None,None]).reshape(len(ids),9,-1)
        res=((observed[ids]-best_syn[ids])*weight).reshape(len(ids),-1)
        normal=jac@jac.transpose(0,2,1)
        diagonal=np.arange(9); normal[:,diagonal,diagonal]*=1+damping[ids,None]
        rhs=np.einsum('nij,nj->ni',jac,res)
        delta=np.stack([pivoted_qr_step(a,b) for a,b in zip(normal,rhs)])
        proposed=clip_parameters((c[ids]+delta)*SCALE)/SCALE
        trial,_=synthesize_response(proposed*SCALE,regions,psfs,mu,with_response=False)
        trial_chi=np.sum(((observed[ids]-trial)*weight)**2,axis=(1,2))
        good=trial_chi<chi[ids]; yes=ids[good]; no=ids[~good]
        improvement=(chi[yes]-trial_chi[good])/chi[yes]
        c[yes]=proposed[good]; chi[yes]=trial_chi[good]; best_syn[yes]=trial[good]
        damping[yes]=np.where(damping[yes]>1.0001e-4,np.clip(damping[yes]/np.sqrt(10.),1e-4,100*np.sqrt(10.)),.001)
        rejected[yes]=0
        quit_small=(improvement<.003)&small[yes]
        small[yes]|=improvement<.003
        accepted[yes]+=1
        damping[no]=np.clip(damping[no]*10.,1e-4,100*np.sqrt(10.)); rejected[no]+=1
        active[no[rejected[no]>=6]]=False
        finished=(chi[yes]<1.)|quit_small|(accepted[yes]>=max_iterations)
        active[yes[finished]]=False
        update=ids[good & active[ids]]
        if len(update):
            _,rf=synthesize_response(c[update]*SCALE,regions,psfs,mu)
            derivatives[update]=rf
        history.append(float(np.mean(chi)))
    return c*SCALE,best_syn,chi,{'accepted':accepted,'history':np.asarray(history)}

def fit_image(initial,data,batch_size=32,max_iterations=25):
    shape=initial.shape[:2]; flat=initial.reshape(-1,9); observed=data['observations'].reshape(-1,4,data['observations'].shape[-1])
    out=np.empty_like(flat); spectra=np.empty_like(observed); chi=np.empty(len(flat)); accepted=np.empty(len(flat),int)
    for start in range(0,len(flat),batch_size):
        stop=min(start+batch_size,len(flat))
        c,s,q,d=lm_block(flat[start:stop],observed[start:stop],data['sigma'],data['regions'],data['psfs'],max_iterations)
        out[start:stop]=c; spectra[start:stop]=s; chi[start:stop]=q; accepted[start:stop]=d['accepted']
    return out.reshape(*shape,9),spectra.reshape(*shape,4,-1),chi.reshape(shape),accepted.reshape(shape)

def smooth_model(model,size=45,fwhm=15.):
    """Original reflected-extension FFT correlation between inversion cycles."""
    sigma2=(fwhm/(2*np.sqrt(2*np.log(2))))**2
    x=np.arange(size)-size//2; psf=np.exp(-.5*(x[:,None]**2+x[None,:]**2)/sigma2); psf/=psf.sum()
    ny,nx=model.shape[:2]; pad=size-1; half=pad//2
    padded=np.zeros((ny+pad,nx+pad,9)); padded[:ny,:nx]=model
    padded[ny:ny+half,:nx]=model[ny-1:ny-half-1:-1]
    padded[ny+half:,:nx]=model[half:0:-1]
    padded[:,nx:nx+half]=padded[:,nx-1:nx-half-1:-1]
    padded[:,nx+half:]=padded[:,half:0:-1]
    kernel=np.zeros((ny+pad,nx+pad)); kernel[:size,:size]=psf
    kernel=np.roll(np.roll(kernel,-size//2+1,axis=0),-size//2+1,axis=1)
    spectrum=np.fft.rfft2(padded,axes=(0,1)); transfer=np.conj(np.fft.rfft2(kernel))
    return np.fft.irfft2(spectrum*transfer[...,None],s=(ny+pad,nx+pad),axes=(0,1))[:ny,:nx]

def reconstruct(data,batch_size=32):
    initial=np.broadcast_to(INITIAL,(*data['raw'].shape[:2],9)).copy()
    one,syn1,q1,a1=fit_image(initial,data,batch_size)
    smoothed=smooth_model(one)
    two,syn2,q2,a2=fit_image(smoothed,data,batch_size)
    calibrated=two.copy(); calibrated[...,3]+=data['cavity']+.45
    return calibrated,{'cycle1':one,'smoothed':smoothed,'cycle2':two,'spectra':syn2,'chi2_cycle1':q1,'chi2_cycle2':q2,'accepted_cycle1':a1,'accepted_cycle2':a2}

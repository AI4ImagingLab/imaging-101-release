"""Explicit mechanical/modal and Matern-5/2 latent-force operators.

Mathematics adapted from Joanna Zou GPLFM2023, MIT; see LICENSE/provenance.
No domain package or ready-made Kalman/inverse routine is imported.
"""
import numpy as np
from scipy.linalg import eigh,expm,solve_continuous_lyapunov
LOWER=np.array([.01,.0001]);SPAN=np.array([99.99,9.9999])

def modal_model(mass,stiffness,damping,sensors,dt,modes=3):
    """Build all calibrated continuous/discrete reduced mechanical matrices."""
    values,phi=eigh(stiffness,mass)
    # Fix an arbitrary modal sign without using any hidden target.
    phi=phi*np.where(phi[-1]<0,-1.,1.)
    omega=np.sqrt(values);gamma=np.diag(np.diag(phi.T@damping@phi))
    P=phi[:,:modes];O=np.diag(values[:modes]);D=gamma[:modes,:modes]
    zero=np.zeros((modes,modes));eye=np.eye(modes)
    Ac=np.block([[zero,eye],[-O,-D]]);Bc=np.vstack((zero,eye))
    Gc=np.concatenate((-sensors@P@O,-sensors@P@D),axis=1);Jc=sensors@P
    Ge=np.concatenate((-P@O,-P@D),axis=1);Je=P
    Ad=expm(Ac*dt);Bd=(Ad-np.eye(2*modes))@np.linalg.inv(Ac)@Bc
    return dict(M=mass,K=stiffness,C=damping,Sa=sensors,Sp=eye,P=eye,Phi_r=P,Omega2_r=O,Gamma_r=D,Transf=np.block([[P,np.zeros_like(P)],[np.zeros_like(P),P]]),f=omega[:modes]/(2*np.pi),zeta=np.diag(D)/(2*omega[:modes]),A_c=Ac,B_c=Bc,G_c=Gc,J_c=Jc,G_e=Ge,J_e=Je,A=Ad,B=Bd,G=Gc,J=Jc,dt=dt,nm=modes,nx=2*modes,ny=len(sensors),nye=len(mass),np=modes)

def matern52(hyperparameters,dt):
    """Return explicit three-state GP SDE and stationary/discrete covariances."""
    alpha,length=np.asarray(hyperparameters,dtype=float)
    if alpha<=0 or length<=0:raise ValueError('Positive GP scales required')
    rate=np.sqrt(5.)/length;q=400*np.sqrt(5.)*alpha**2/(3*length**5)
    Fc=np.array([[0.,1.,0.],[0.,0.,1.],[-rate**3,-3*rate**2,-3*rate]])
    Lc=np.array([[0.],[0.],[1.]]);Hc=np.array([[1.,0.,0.]])
    F=expm(Fc*dt);L=(F-np.eye(3))@np.linalg.inv(Fc)@Lc
    P=solve_continuous_lyapunov(Fc,-Lc*q@Lc.T)
    Q=P-F@P@F.T
    return dict(hp=np.array([alpha,length]),q=q,F_c=Fc,L_c=Lc,H_c=Hc,F=F,L=L,H=Hc,P=P,Q=Q,sigma_w=np.sqrt(q/dt),nz=3)

def augmented_model(model,hyperparameters):
    """Assemble mechanical+GP state, Lyapunov covariance and exact transition."""
    m=model['nm'];nx=model['nx'];dim=3*m;knl=matern52(hyperparameters,model['dt'])
    Fc=np.zeros((dim,dim));Bc=np.zeros((nx,dim));Jc=np.zeros((model['ny'],dim));Je=np.zeros((model['nye'],dim));Qc=np.zeros((dim,dim))
    for i in range(m):
        sl=slice(3*i,3*i+3);Fc[sl,sl]=knl['F_c']
        Bc[:,sl]=model['B_c'][:,i,None]@knl['H_c'];Jc[:,sl]=model['J_c'][:,i,None]@knl['H_c'];Je[:,sl]=model['J_e'][:,i,None]@knl['H_c']
        Qc[sl,sl]=knl['L_c']*knl['q']@knl['L_c'].T
    A=np.block([[model['A_c'],Bc],[np.zeros((dim,nx)),Fc]])
    H=np.concatenate((model['G_c'],Jc),axis=1);He=np.concatenate((model['G_e'],Je),axis=1)
    Q=np.block([[np.zeros((nx,nx)),np.zeros((nx,dim))],[np.zeros((dim,nx)),Qc]])
    P=solve_continuous_lyapunov(A,-Q);F=expm(A*model['dt']);Qd=P-F@P@F.T
    return dict(F_cs=Fc,B_cs=Bc,J_cs=Jc,J_cse=Je,Q_c=Qc,F_ac=A,H_ac=H,H_ae=He,Q_ac=Q,P_0=P,F_ad=F,H_ad=H,Q_ad=Qd,na=len(A))

def hellinger_squared(reference_std,model_std,npoints=500):
    """Native mean squared Hellinger via500-point trapezoid on +/-5 sigma."""
    values=[]
    for r,s in zip(reference_std,model_std):
        if r<=0 or s<=0:raise ValueError('Nondegenerate Gaussian marginals required')
        grid=np.linspace(-5*max(r,s),5*max(r,s),npoints)
        pr=np.exp(-.5*(grid/r)**2)/(np.sqrt(2*np.pi)*r)
        ps=np.exp(-.5*(grid/s)**2)/(np.sqrt(2*np.pi)*s)
        values.append(1-np.trapezoid(np.sqrt(pr*ps),grid))
    return float(np.mean(values))

def hyperparameter_cost(normalized,observations,model):
    """Fit modelled stationary acceleration marginals to all measured samples."""
    hp=LOWER+SPAN*np.asarray(normalized);aug=augmented_model(model,hp)
    covariance=aug['H_ad']@aug['P_0']@aug['H_ad'].T
    # Native data convention excludes rows exactly zero at the first sample.
    active=observations[0]!=0
    empirical=np.cov(observations[:,active],rowvar=False,ddof=1)
    return hellinger_squared(np.sqrt(np.abs(np.diag(empirical))),np.sqrt(np.abs(np.diag(covariance))))

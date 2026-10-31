"""Independent incompressible plane-stress hyperelastic and weak operators."""
import numpy as np
ALPHAS=np.arange(-10.,20.,.2)
BALANCE=20.
def features(F):
    F=np.asarray(F,dtype=np.float64).reshape(-1,2,2)
    J=np.linalg.det(F)
    if np.any(J<=0): raise ValueError('Orientation-preserving F required')
    f=F.reshape(-1,4); cofactor=np.column_stack((f[:,3],-f[:,2],-f[:,1],f[:,0]))
    s=np.sum(f*f,axis=1); I1=s+J**-2; I2=J**2+s*J**-2
    d1=2*f-2*cofactor/J[:,None]**3
    d2=2*J[:,None]*cofactor+2*f/J[:,None]**2-2*s[:,None]*cofactor/J[:,None]**3
    k1=I1-3; k2=I2-3; values=[]; derivatives=[]
    for degree in range(1,6):
        for q in range(degree+1):
            p=degree-q; values.append(k1**p*k2**q)
            d=np.zeros_like(f)
            if p:d+=p*(k1**(p-1)*k2**q)[:,None]*d1
            if q:d+=q*(k1**p*k2**(q-1))[:,None]*d2
            derivatives.append(d)
    values.append(np.log(I2/3)); derivatives.append(d2/I2[:,None])
    U,L,Vh=np.linalg.svd(F); dl=np.einsum('eik,ekj->ekij',U,Vh).reshape(-1,2,4)
    for alpha in ALPHAS:
        values.append(np.sum(L**alpha,axis=1)+J**(-alpha)-3)
        d=alpha*np.einsum('ek,ekj->ej',L**(alpha-1),dl)-alpha*J[:,None]**(-alpha-1)*cofactor
        derivatives.append(d)
    return np.stack(values,axis=1),np.stack(derivatives,axis=1)
def law(F,theta):
    Q,G=features(F)
    return Q@theta,np.einsum('ekj,k->ej',G,theta).reshape(-1,2,2)
def weak_operator(triangles,shape_gradients,areas,feature_gradient,nnode):
    G=feature_gradient.reshape(len(triangles),171,2,2)
    contributions=np.einsum('ekij,enj,e->enik',G,shape_gradients,areas)
    A=np.zeros((nnode,2,171))
    for local in range(3): np.add.at(A,triangles[:,local],contributions[:,local])
    return A
def frame_system(A,free_dofs,boundary_masks,reactions,balance=BALANCE):
    Af=A[free_dofs]; R=np.einsum('rnd,ndk->rk',boundary_masks,A)
    return 2*(Af.T@Af+balance*R.T@R),2*balance*R.T@reactions,balance*float(reactions@reactions)

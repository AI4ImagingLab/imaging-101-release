"""Explicit Williams elastic crack-tip eigenfields.

Adapted from DLR CrackPy, MIT license. No domain library is imported.
Displacements are mm and stresses MPa. Coefficient n has units
MPa mm**(1-n/2); term ordering is recorded in the task contract.
"""
import numpy as np

YOUNG_MODULUS = 72000.
POISSON_RATIO = .33
TERMS = np.array([-1,1,2,3,4,5])


def displacement_basis(r,phi,terms=TERMS,young_modulus=YOUNG_MODULUS,poisson_ratio=POISSON_RATIO):
    """Return x,y basis arrays[2*K,*r.shape], all a terms then all b terms."""
    shear=young_modulus/(2*(1+poisson_ratio))
    kappa=(3-poisson_ratio)/(1+poisson_ratio)
    bx=[];by=[];cx=[];cy=[]
    for n in terms:
        h=n/2;sign=(-1.)**n;radial=r**h/(2*shear)
        t=h*phi;s=(h-2)*phi
        bx.append(radial*((kappa+sign+h)*np.cos(t)-h*np.cos(s)))
        by.append(radial*((kappa-sign-h)*np.sin(t)+h*np.sin(s)))
        cx.append(radial*((-kappa+sign-h)*np.sin(t)+h*np.sin(s)))
        cy.append(radial*((kappa+sign-h)*np.cos(t)+h*np.cos(s)))
    return np.asarray(bx+cx),np.asarray(by+cy)


def stress_field(coefficients,r,phi,terms=TERMS):
    """Return [sigma_xx,sigma_yy,sigma_xy] on the given polar grid in MPa."""
    coefficients=np.asarray(coefficients)
    k=len(terms);a,b=coefficients[:k],coefficients[k:]
    xx=np.zeros_like(r);yy=np.zeros_like(r);xy=np.zeros_like(r)
    for i,n in enumerate(terms):
        h=n/2;sign=(-1.)**n;scale=h*r**(h-1)
        t=(h-1)*phi;s=(h-3)*phi
        xx+=scale*(a[i]*((2+h+sign)*np.cos(t)-(h-1)*np.cos(s))
                   -b[i]*((2+h-sign)*np.sin(t)-(h-1)*np.sin(s)))
        yy+=scale*(a[i]*((2-h-sign)*np.cos(t)+(h-1)*np.cos(s))
                   -b[i]*((2-h+sign)*np.sin(t)+(h-1)*np.sin(s)))
        xy+=scale*(a[i]*((h-1)*np.sin(s)-(h+sign)*np.sin(t))
                   +b[i]*((h-1)*np.cos(s)-(h-sign)*np.cos(t)))
    return np.asarray([xx,yy,xy])


def stress_intensity(coefficients,terms=TERMS):
    """Return [K_I,K_II] in MPa sqrt(m), including the antisymmetric sign."""
    k=len(terms);index=np.flatnonzero(np.asarray(terms)==1)
    if len(index)!=1:raise ValueError('The singular n=1 term is required')
    i=index.item();factor=np.sqrt(2*np.pi)/np.sqrt(1000.)
    return factor*np.array([coefficients[i],-coefficients[k+i]])


def assemble_system(sampled_x,sampled_y,r,phi,terms=TERMS):
    """Build the explicit displacement matrix A and measured target b."""
    bx,by=displacement_basis(r,phi,terms)
    target=np.concatenate((sampled_x.ravel(),sampled_y.ravel()))
    matrix=np.concatenate((bx.reshape(len(bx),-1),by.reshape(len(by),-1)),axis=1).T
    valid=np.isfinite(target)
    return matrix[valid],target[valid],valid

"""Disk point-contact elastic stress superposition and stress-optic law.
Adapted from Jack Featherstone's MIT-licensed pepe (2022). See NOTICE.
"""
import numpy as np
from .preprocessing import CENTER,RADIUS,EFFECTIVE_PIXELS_PER_METRE,FSIGMA,disk_mask

def stress_basis(beta,shape=(181,181),pixels_per_metre=EFFECTIVE_PIXELS_PER_METRE,alpha=None,center=CENTER,radius=RADIUS):
    y,x=np.mgrid[:shape[0],:shape[1]];x=(x-center[1])/pixels_per_metre;y=(y-center[0])/pixels_per_metre;R=radius/pixels_per_metre
    basis=np.zeros((len(beta),3)+shape);mask=disk_mask(shape,center,radius)
    if alpha is None:alpha=np.zeros(len(beta))
    for i,b in enumerate(beta):
        a=alpha[i];b1=-b+np.pi;b2=b1-(-1+2*int(np.sign(a)>-1))*np.pi+2*a
        cx=R*(np.sin(b2)-np.sin(b1));cy=R*(np.cos(b2)-np.cos(b1));chord=np.sqrt(cx*cx+cy*cy)
        if chord==0:return np.zeros_like(basis)
        cx/=chord;cy/=chord
        vx=x-R*np.sin(b1);vy=-y-R*np.cos(b1);r=np.sqrt(vx*vx+vy*vy)
        arg=(vx*cx+vy*cy)/r-1e-10
        # Model is evaluated only within the disk; outside values are immaterial.
        arg=np.where(mask,arg,0.)
        theta=np.sign(vy*cx-vx*cy)*np.arccos(arg)+b-np.pi/2-a
        sr=(-2/np.pi)*arg/r
        basis[i,0]=np.where(mask,sr*np.sin(theta)**2,0)
        basis[i,1]=np.where(mask,sr*np.cos(theta)**2,0)
        basis[i,2]=np.where(mask,.5*sr*np.sin(2*theta),0)
    return basis

def principal_difference(forces,basis):
    stress=np.einsum('j,jcyx->cyx',forces,basis)
    return np.sqrt((stress[0]-stress[1])**2+4*stress[2]**2)

def intensity_from_stress(stress_difference,fsigma=FSIGMA):
    return np.sin(np.pi/fsigma*stress_difference)**2

def forward(forces,basis,fsigma=FSIGMA):
    return intensity_from_stress(principal_difference(forces,basis),fsigma)

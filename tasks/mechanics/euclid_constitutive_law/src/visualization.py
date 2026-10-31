"""Review helpers: calibrated mesh plots and full-observation forward residuals."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from .preprocessing import element_deformation
from .physics_model import features,weak_operator
def mesh_field(ax,data,values,title,units,cmap='viridis'):
    x=data['coordinates'];tri=mtri.Triangulation(x[:,0],x[:,1],data['triangles'])
    image=ax.tripcolor(tri,facecolors=values,shading='flat',cmap=cmap)
    ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    ax.figure.colorbar(image,ax=ax,label=units)
    return image
def forward_records(data,theta):
    predictions=[];free_rms=[]
    for u in data['displacements']:
        F=element_deformation(data['triangles'],data['shape_gradients'],u)
        _,G=features(F);A=weak_operator(data['triangles'],data['shape_gradients'],data['element_areas'],G,len(data['coordinates']))
        forces=np.einsum('nik,k->ni',A,theta)
        predictions.append(np.einsum('rni,ni->r',data['boundary_masks'],forces))
        free_rms.append(np.sqrt(np.mean(forces[data['free_dofs']]**2)))
    return np.asarray(predictions),np.asarray(free_rms)

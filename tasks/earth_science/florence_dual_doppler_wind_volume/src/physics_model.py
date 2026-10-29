"""Explicit PyDDA variational wind equations. BSD-3-Clause, PyDDA authors.
JAX oracle differentiates the same discrete objective; local adjoints are exact
for the one-sided-edge/centered-interior first-difference discretization.
"""
import numpy as np
def fall_speed(reflectivity,z,freezing_height=5000.):
    """Original empirical downward terminal speed, m/s, from dBZ and height m."""
    z=np.broadcast_to(np.asarray(z)[None,:,None,None],reflectivity.shape);A=np.zeros_like(reflectivity);B=np.zeros_like(reflectivity);r=reflectivity
    for condition,a,b in [( (z<freezing_height)&(r<55),-2.6,.0107),((z<freezing_height)&(r>=55)&(r<60),-2.5,.013),((z<freezing_height)&(r>60),-3.95,.0148),((z>=freezing_height)&(r<33),-.817,.0063),((z>=freezing_height)&(r>=33)&(r<49),-2.5,.013),((z>=freezing_height)&(r>49),-3.95,.0148)]:A[condition]=a;B[condition]=b
    return A*10**(r*B)*(1.2/np.exp(-z/10000.))**.4
def projection(azimuth,elevation):
    """Radar components east,north,up; azimuth clockwise from north, radians."""
    return np.stack([np.cos(elevation)*np.sin(azimuth),np.cos(elevation)*np.cos(azimuth),np.sin(elevation)],axis=1)
def forward(winds,data):
    a=projection(data['azimuth'],data['elevation']);terminal=fall_speed(data['reflectivity'],data['z'])
    return np.einsum('rczyx,czyx->rzyx',a,winds)-np.sin(data['elevation'])*abs(terminal)
def derivative(values,spacing,axis):return np.gradient(values,spacing,axis=axis)
def derivative_adjoint(values,spacing,axis):
    """Transpose of np.gradient(edge_order1), not negative-gradient shortcut."""
    v=np.moveaxis(values,axis,-1);r=np.zeros_like(v);r[...,0]-=v[...,0]/spacing;r[...,1]+=v[...,0]/spacing;r[...,:-2]-=v[...,1:-1]/(2*spacing);r[...,2:]+=v[...,1:-1]/(2*spacing);r[...,-2]-=v[...,-1]/spacing;r[...,-1]+=v[...,-1]/spacing
    return np.moveaxis(r,-1,axis)
def divergence(winds,data):
    dz,dy,dx=[np.diff(data[k])[0] for k in ['z','y','x']];z=np.broadcast_to(data['z'][:,None,None],winds.shape[1:]);rho=np.exp(-z/10000.);drho=derivative(rho,dz,0)/rho
    return derivative(winds[0],dx,2)+derivative(winds[1],dy,1)+derivative(winds[2],dz,0)+winds[2]*drho
def objective_gradient(flat_winds,data,mass_weight=256.,prior_weight=1e-4):
    """Original normalized radial + half anelastic + horizontal prior cost."""
    from .preprocessing import normalize_observations
    shape=data['radial_velocity'].shape[1:];v=flat_winds.reshape((3,)+shape);a=projection(data['azimuth'],data['elevation']);pred=forward(v,data);res=pred-data['radial_velocity'];r=res*data['validity']/normalize_observations(data)**2;value=float(np.sum(res*r));gradient=2*np.einsum('rczyx,rzyx->czyx',a,r)
    dz,dy,dx=[np.diff(data[k])[0] for k in ['z','y','x']];z=np.broadcast_to(data['z'][:,None,None],shape);rho=np.exp(-z/10000.);dr=derivative(rho,dz,0)/rho;div=divergence(v,data)
    value+=mass_weight*.5*np.sum(div**2);gradient+=mass_weight*np.stack([derivative_adjoint(div,dx,2),derivative_adjoint(div,dy,1),derivative_adjoint(div,dz,0)+div*dr])
    value+=prior_weight*np.sum(v[:2]**2);gradient[:2]+=2*prior_weight*v[:2]
    # These vertical boundary variables remain zero from initialization.
    gradient[2,0]=0;gradient[2,-1]=0
    return float(value),gradient.ravel()

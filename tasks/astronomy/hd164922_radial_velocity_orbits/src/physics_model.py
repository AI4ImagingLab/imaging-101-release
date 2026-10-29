"""Local Newtonian Doppler forward model. Adapted from RadVel, MIT (see LICENSE)."""
import numpy as np

def solve_kepler(mean_anomaly,eccentricity):
    """Third-order Danby corrections, native initial guess and 1e-12 rad residual."""
    m=np.asarray(mean_anomaly,dtype=np.float64)
    e=np.broadcast_to(np.asarray(eccentricity,dtype=np.float64),m.shape)
    E=m+np.sign(np.sin(m))*.85*e
    f=E-e*np.sin(E)-m
    for _ in range(100):
        active=np.abs(f)>1e-12
        if not np.any(active):return E
        ea=E[active];ee=e[active];ff=f[active]
        fp=1-ee*np.cos(ea);fpp=ee*np.sin(ea);fppp=1-fp
        d1=-ff/fp;d2=-ff/(fp+d1*fpp/2);d3=-ff/(fp+d2*fpp/2+d2*d2*fppp/6)
        E[active]=ea+d3;f=E-e*np.sin(E)-m
    raise RuntimeError('Kepler residual did not converge')

def transit_to_periastron(tc,period,eccentricity,omega):
    f=np.pi/2-omega
    E=2*np.arctan(np.tan(f/2)*np.sqrt((1-eccentricity)/(1+eccentricity)))
    return tc-period/(2*np.pi)*(E-eccentricity*np.sin(E))

def planet_velocity(time,period,tc,eccentricity,omega,amplitude):
    t=np.asarray(time,dtype=np.float64)
    tp=transit_to_periastron(tc,period,eccentricity,omega)
    # Native defensive clamps are inactive throughout the physically permitted prior.
    if eccentricity==0:
        m=2*np.pi*((t-tp)/period-np.floor((t-tp)/period))
        return amplitude*np.cos(m+omega)
    period=max(period,1e-4);eccentricity=np.clip(eccentricity,0,.99)
    m=2*np.pi*((t-tp)/period-np.floor((t-tp)/period))
    E=solve_kepler(m,eccentricity)
    nu=2*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(E/2))
    return amplitude*(np.cos(nu+omega)+eccentricity*np.cos(omega))

def unpack_orbits(vector):
    """Two rows (P days, inferior-conjunction BJD, e, omega radians, K m/s)."""
    v=np.asarray(vector)
    return np.array([[v[j],v[j+1],v[j+2]**2+v[j+3]**2,np.arctan2(v[j+3],v[j+2]),np.exp(v[j+4])] for j in (0,5)])

def components(time,vector):
    return np.stack([planet_velocity(time,*row) for row in unpack_orbits(vector)])

def forward(time,instrument,vector):
    return components(time,vector).sum(axis=0)+np.asarray(vector)[10:13][instrument]

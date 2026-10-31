"""Thin-plate local energy balance extracted from HFITS, GPL-3.0.

Adapted from P. Dehghani and M. J. DiDomizio's inverse_heat_transfer and
calculate_gradients. The selected source mode uses Engineering Toolbox steel
properties, constant convection on both faces, and NumPy time derivatives.
"""
import numpy as np

EMISSIVITY=.94
THICKNESS=.000793
AMBIENT=293.15
H_EXPOSED=10.
H_UNEXPOSED=10.
STEFAN_BOLTZMANN=5.67e-8
SPACING=.0127
TIME_STEP=1.

def thermal_properties(temperature):
    """Return k [W/m/K], cp [J/kg/K], rho [kg/m³] at Kelvin input array."""
    t=np.asarray(temperature)
    return 8.1119*np.log(t)-30.377,1000*(.1414*np.log(t)-.3534),8050.

def time_derivative(temperature,time_step=TIME_STEP):
    """Second-order central/one-sided derivative along time axis0 [K/s]."""
    return np.gradient(temperature,axis=0,edge_order=2)/time_step

def laplacian(temperature,spacing=SPACING):
    """Five-point square-grid Laplacian on interior pixels [K/m²].

    Input (time,y,x) includes a measured one-pixel collar; output trims spatial
    collar only. No boundary extrapolation is used at any target pixel.
    """
    t=np.asarray(temperature);c=t[:,1:-1,1:-1]
    return ((t[:,2:,1:-1]-c)+(t[:,:-2,1:-1]-c)+(t[:,1:-1,2:]-c)+(t[:,1:-1,:-2]-c))/spacing**2

def energy_terms(temperature,time_step=TIME_STEP,spacing=SPACING,thickness=THICKNESS,emissivity=EMISSIVITY,ambient=AMBIENT,h_exposed=H_EXPOSED,h_unexposed=H_UNEXPOSED):
    """Return storage, convection, radiation, conduction, each interior W/m².

    k(T) multiplies the local discrete Laplacian, preserving HFITS's local-
    conductivity approximation rather than silently replacing it by div(k gradT).
    Radiation is loss from both faces minus incoming ambient IR on the back face.
    """
    t=np.asarray(temperature);c=t[:,1:-1,1:-1];k,cp,rho=thermal_properties(c)
    storage=cp*rho*thickness*time_derivative(t,time_step)[:,1:-1,1:-1]
    convection=(h_exposed+h_unexposed)*(c-ambient)
    radiation=2*emissivity*STEFAN_BOLTZMANN*(c**4-.5*ambient**4)
    conduction=k*thickness*laplacian(t,spacing)
    return {'storage':storage,'convection':convection,'radiation':radiation,'conduction':conduction}

def predicted_storage(temperature,incident_radiation,**parameters):
    """Forward balance storage term [W/m²] for a trial incident field [kW/m²]."""
    terms=energy_terms(temperature,**parameters)
    emissivity=parameters.get('emissivity',EMISSIVITY)
    return 1000*emissivity*incident_radiation-terms['convection']-terms['radiation']+terms['conduction']

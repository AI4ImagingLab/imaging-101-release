"""Full PT canopy/soil inverse with the standard LAI-zero OSEB branch."""
import numpy as np
from . import physics_model as p
from .preprocessing import load_observations,prepare_forcing
def reconstruct(path='data/raw_data.npz',nominal_height=.3):
    inputs=load_observations(path);f=prepare_forcing(inputs);mask=f['canopy_mask'];bare=~mask;n=len(mask)
    # Height sensitivity is a labeled supplemental physical assumption experiment.
    height=np.full(n,nominal_height);z0=f['z0'].copy();d0=f['d0'].copy()
    if nominal_height!=.3:z0[mask],d0[mask]=p.res_calc_roughness(f['LAI'][mask],height[mask])
    pt=p.TSEB_TSEB_PT(inputs['radiometric_temperature'][mask],0.,inputs['air_temperature'][mask],inputs['wind_speed'][mask],f['ea'][mask],inputs['surface_pressure'][mask]/100,f['SnC'],f['SnS'],inputs['thermal_radiation'][mask],f['LAI'][mask],height[mask],.98,.95,z0[mask],d0[mask],10.,2.)
    soil=p.TSEB_OSEB(inputs['radiometric_temperature'][bare],inputs['air_temperature'][bare],inputs['wind_speed'][bare],f['ea'][bare],inputs['surface_pressure'][bare]/100,f['soil_Sn'],inputs['thermal_radiation'][bare],.95,z0[bare],d0[bare],10.,2.)
    names=['flag','Ts','Tc','Tac','LnS','LnC','LEc','Hc','LEs','Hs','G','Rs','Rx','Ra','ustar','L','iterations','alpha_PT']
    d={'pt_'+k:v for k,v in zip(names,pt)};d.update({'soil_'+k:v for k,v in zip(['flag','LnS','LEs','Hs','G','Ra','ustar','L','iterations'],soil)});d.update(f)
    if np.any(~np.isin(d['pt_flag'],[0,3,5])) or np.any(~np.isin(d['soil_flag'],[10,15])):raise RuntimeError('Original inverse reported an invalid or out-of-range pixel; no fitted pixel is masked')
    flux=np.zeros((n,4));flux[mask]=np.stack([d['pt_'+k] for k in ['LEc','Hc','LEs','Hs']],axis=-1);flux[bare,2:]=np.stack([d['soil_LEs'],d['soil_Hs']],axis=-1)
    ts=np.zeros(n);ts[mask]=d['pt_Ts'];ts[bare]=inputs['radiometric_temperature'][bare]
    g=np.zeros(n);g[mask]=d['pt_G'];g[bare]=d['soil_G']
    radiation=np.zeros((n,4));radiation[mask]=np.stack([f['SnC'],f['SnS'],d['pt_LnC'],d['pt_LnS']],axis=-1);radiation[bare,1]=f['soil_Sn'];radiation[bare,3]=d['soil_LnS']
    result=dict(reconstruction=flux[None],soil_temperature=ts[None],canopy_temperature=d['pt_Tc'].astype('float64')[None],soil_heat_flux=g[None],radiation=radiation[None])
    if any(not np.isfinite(a).all() for a in result.values()):raise RuntimeError('Nonfinite reconstruction; no output clipping/masking is permitted')
    return result,d

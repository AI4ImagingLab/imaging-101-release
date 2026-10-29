"""Measured Landsat/ERA5 inputs and explicitly nominal physical forcing."""
from pathlib import Path
import numpy as np
from . import physics_model as p
KEYS=('NDVI','radiometric_temperature','dewpoint_temperature','surface_pressure','air_temperature','wind_speed','solar_radiation','thermal_radiation')
def load_observations(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k][0].copy() for k in (*KEYS,'pixel_index')}
    if any(a.shape!=(110935,) for a in arrays.values()):raise ValueError('Complete110935valid-pixel input required')
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite public observation')
    return arrays
def prepare_forcing(inputs):
    ndvi=inputs['NDVI'];lai=-np.log(1-np.clip(ndvi-.05,0,1))/.5
    canopy=lai>0;bare=~canopy;n=len(lai)
    zenith,_=p.met_calc_sun_angles(30.25,38.25,45.,111,10.98);zenith=np.full(n,float(np.asarray(zenith)))
    diffvis,diffnir,fvis,fnir=p.rad_calc_difuse_ratio(inputs['solar_radiation'],zenith,press=inputs['surface_pressure']/100)
    diffuse=diffvis*fvis+diffnir*fnir;direct=inputs['solar_radiation']*(1-diffuse);scattered=inputs['solar_radiation']*diffuse
    ea=p.met_calc_vapor_pressure(inputs['dewpoint_temperature']);height=np.full(n,.3);z0=np.full(n,.01);d0=np.zeros(n)
    z0[canopy],d0[canopy]=p.res_calc_roughness(lai[canopy],height[canopy],w_C=np.ones(canopy.sum()),landcover=np.full(canopy.sum(),p.res_CROP),f_c=np.ones(canopy.sum()))
    omega0=p.CI_calc_omega0_Kustas(lai[canopy],np.ones(canopy.sum()),isLAIeff=True);omega=p.CI_calc_omega_Kustas(omega0,zenith[canopy])
    snc,sns=p.rad_calc_Sn_Campbell(lai[canopy],zenith[canopy],direct[canopy],scattered[canopy],fvis[canopy],fnir[canopy],*[np.full(canopy.sum(),v) for v in [.07,.08,.32,.33,.15,.25]],LAI_eff=lai[canopy]*omega)
    soil_sn=(1-(fvis[bare]*.15+fnir[bare]*.25))*inputs['solar_radiation'][bare]
    return dict(LAI=lai,canopy_mask=canopy,zenith=zenith,ea=ea,z0=z0,d0=d0,SnC=snc,SnS=sns,soil_Sn=soil_sn,height=height)

"""Published CCM89 extinction, preserving original observed precision."""
import numpy as np

C_HBETA=.047
R_V=3.1
# Archived original float32 conversion of supplied cHbeta to E(B-V).
# Preserve its value explicitly rather than changing rounding under NumPy2.
E_BV=0.0325583778321743011

def extinction(wavelength):
    x=1e4/np.asarray(wavelength,dtype=float)
    a=np.zeros_like(x);b=np.zeros_like(x)
    mask=(x>.3)&(x<=1.1)
    a[mask]=.574*x[mask]**1.61;b[mask]=-.527*x[mask]**1.61
    mask=(x>1.1)&(x<=3.3);y=x[mask]-1.82
    a[mask]=1+.17699*y-.50447*y**2-.02427*y**3+.72085*y**4+.01979*y**5-.7753*y**6+.32999*y**7
    b[mask]=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-.622510*y**5+5.3026*y**6-2.09002*y**7
    if np.any(x>3.3):raise ValueError('This task uses only optical and IR CCM89 branches')
    return R_V*a+b

def deredden(flux,wavelength):
    # The original Observation.correctData uses absolute extinction correction,
    # not Hbeta-renormalized getCorrHb. MidIRoutsideCCMrange is zero extinction.
    # Original observation parser stores cHbeta in float32 and its NumPy1.26
    # E(B-V) conversion performs both arithmetic stages in float32.
    return np.asarray(flux,dtype=float)*10**(.4*E_BV*extinction(wavelength))

LINE_WAVELENGTHS={'S4_10.5m': 105000.0, 'Ne2_12.8m': 128000.0, 'Ne3_15.6m': 156000.0, 'S3_18.7m': 187000.0, 'O2_3726A': 3726.0, 'O2_3729A': 3729.0, 'Ne3_3869A': 3869.0, 'Ne3_3968A': 3968.0, 'S2_4069A': 4069.0, 'S2_4076A': 4076.0, 'O3_4363A': 4363.0, 'O3_5007A': 5007.0, 'N2_5755A': 5755.0, 'S3_6312A': 6312.0, 'O1_6300A': 6300.0, 'O1_6364A': 6364.0, 'N2_6548A': 6548.0, 'N2_6584A': 6584.0, 'S2_6716A': 6716.0, 'S2_6731A': 6731.0, 'Ar3_7136A': 7136.0, 'O2_7319A+': 7319.0, 'O2_7330A+': 7330.0}

def load_observations(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as a:
        raw={key:float(a['flux_'+key][0,0]) for key in LINE_WAVELENGTHS}
    return raw,{key:float(deredden(value,LINE_WAVELENGTHS[key])) for key,value in raw.items()}

def observation_ratios(corrected):
    return {'NII':corrected['N2_5755A']/corrected['N2_6584A'],
            'OII':corrected['O2_3726A']/corrected['O2_3729A'],
            'OIII':corrected['O3_4363A']/corrected['O3_5007A'],
            'SII':corrected['S2_6731A']/corrected['S2_6716A'],
            'SIIblue':(corrected['S2_4069A']+corrected['S2_4076A'])/(corrected['S2_6716A']+corrected['S2_6731A']),
            'SIII':corrected['S3_6312A']/corrected['S3_18.7m'],
            'NeIII':(corrected['Ne3_3869A']+corrected['Ne3_3968A'])/corrected['Ne3_15.6m']}

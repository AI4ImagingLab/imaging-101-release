"""Standalone trusted physical evaluator, GPL3; no solver imports."""
"""Explicit published isothermal transmission physics, no domain-library calls.

Adapted from PLATON v4, GPL-3.0, Michael Zhang and contributors. Full source
revision, adaptations and license are recorded in test/validation.md and LICENSE.
"""
import numpy as np
from scipy.interpolate import interpn,RectBivariateSpline,UnivariateSpline


K_B=1.38064852e-23
AMU=1.660539040e-27
G=6.67408e-11
M_SUN=1.98848e30
R_SUN=6.957e8
R_JUP=7.1492e7
M_JUP=1.89819e27
T_STAR=6091.
REF_PRESSURE=1e5
N_PROFILE=250

class AtmosphereError(ValueError):
    """Physically unbound atmosphere or invalid atmospheric state."""

def shell_paths(radii):
    """Two-sided distances through shells, descending radii, metres."""
    r_mesh,r_prime_mesh=np.meshgrid(radii,radii)
    square=r_prime_mesh[:,:-1]**2-r_mesh[:,1:]**2;square[square<0]=0
    lengths=2*np.sqrt(square)
    return lengths[:-1]-lengths[1:]

def line_of_sight_tau(absorption_coeff,radii):
    """Midpoint absorption integrated along chord paths; output wavelength by ray."""
    if not np.allclose(radii,np.sort(radii)[::-1]):raise AtmosphereError('Radii must descend')
    midpoint=.5*(absorption_coeff[:-1]+absorption_coeff[1:])
    return np.dot(midpoint.T,shell_paths(radii))

def integrate_radii(log_pressures,planet_mass,planet_radius,temperature_spline,mu_spline):
    """Explicit hydrostatic increments in reciprocal radius, starting at 1 bar."""
    mu=(mu_spline(log_pressures[1:])+mu_spline(log_pressures[:-1]))/2.
    temperature=(temperature_spline(log_pressures[1:])+temperature_spline(log_pressures[:-1]))/2.
    delta=np.diff(log_pressures)*K_B*temperature/(G*planet_mass*mu*AMU)
    inverse=1./planet_radius+np.cumsum(delta)
    return np.append(planet_radius,1./inverse)

def hydrostatic_radii(pressure,temperature,mu,planet_mass,planet_radius,star_radius,above):
    """Original spherical hydrostatic discretization and Hill-radius guard."""
    hill=star_radius*(T_STAR/temperature[0])**2*(planet_mass/(3*M_SUN))**(1./3)
    estimate=1./(1./planet_radius+K_B*np.mean(temperature)*np.log(pressure[0]/pressure[-1])/(G*planet_mass*np.mean(mu)*AMU))
    if estimate<0 or estimate>hill:raise AtmosphereError('Atmosphere unbound: height > Hill radius')
    ms=UnivariateSpline(np.log(pressure),mu,s=0);ts=UnivariateSpline(np.log(pressure),temperature,s=0)
    below=np.append(REF_PRESSURE,pressure[pressure>REF_PRESSURE]);upper=np.append(REF_PRESSURE,pressure[pressure<=REF_PRESSURE][::-1])
    rb=integrate_radii(np.log(below),planet_mass,planet_radius,ts,ms);ra=integrate_radii(np.log(upper),planet_mass,planet_radius,ts,ms)
    radii=np.append(ra.flatten()[1:][::-1],rb.flatten()[1:]);radii=radii[above]
    return radii,-np.diff(radii)

class TransmissionModel:
    """All active v4 hydrostatic, gas, Rayleigh, collision and band operators."""
    def __init__(self,raw):
        self.raw=raw;self.wave=raw['wavelength_grid'];self.pressure=raw['pressure_grid'];self.temperature=raw['temperature_grid']
        self.absorber_map={int(s):i for i,s in enumerate(raw['absorber_species'])}
        self.pmesh,self.tmesh,_=np.meshgrid(self.pressure,self.temperature,self.wave)
        self.bin_masks=[(self.wave>=lo)&(self.wave<hi) for lo,hi in raw['wavelength_bins']]

    def abundances(self,metallicity):
        """Bilinear log(Z),log(C/O) chemical-equilibrium calibration interpolation."""
        values=10**interpn((self.raw['metallicity_grid'],np.log10(self.raw['co_grid'])),self.raw['log_abundance_grid'],[metallicity,np.log10(.53)])[0]
        values[values<1e-99]=1e-99
        return values

    def atmospheric_profiles(self,temperature,abundances,planet_mass,planet_radius,star_radius,cloudtop):
        pressure=np.logspace(np.log10(self.pressure[0]),np.log10(self.pressure[-1]),N_PROFILE)
        thermal=np.ones(len(pressure))*temperature;above=pressure<cloudtop;mu=np.zeros(len(pressure));composition=[]
        for species,grid in enumerate(abundances):
            interpolator=RectBivariateSpline(self.temperature,np.log10(self.pressure),np.log10(grid),kx=1,ky=1)
            abundance=10**interpolator.ev(thermal,np.log10(pressure));composition.append(abundance)
            mu+=abundance*self.raw['species_mass'][species]
        radii,dr=hydrostatic_radii(pressure,thermal,mu,planet_mass,planet_radius,star_radius,above)
        return pressure[above],thermal[above],radii,dr,mu,np.array(composition)[:,above]

    def absorption(self,abundances,temperature_condition,pressure_condition,scattering_factor,gas=True,rayleigh=True,collision=True):
        tc=temperature_condition;pc=pressure_condition;shape=(np.sum(tc),np.sum(pc),len(self.wave));gas_coeff=np.zeros(shape)
        if gas:
            for species,grid in enumerate(abundances):
                if species in self.absorber_map:gas_coeff+=self.raw['opacity_grid'][self.absorber_map[species]][tc][:,pc]*grid[tc][:,pc,np.newaxis]
        rayleigh_coeff=np.zeros(shape)
        if rayleigh:
            polar_sum=np.zeros((np.sum(tc),np.sum(pc)))
            for species,grid in enumerate(abundances):
                polar=self.raw['species_polarizability'][species]
                if polar!=0:polar_sum+=grid[tc,:][:,pc]*polar**2
            density=self.pmesh[tc][:,pc]/(K_B*self.tmesh[tc][:,pc])
            rayleigh_coeff=scattering_factor*(128./3*np.pi**5)*density*polar_sum[:,:,np.newaxis]/self.wave.reshape(1,1,-1)**4
        collision_coeff=np.zeros(shape)
        if collision:
            density=self.pressure[np.newaxis,pc]/(K_B*self.temperature[tc,np.newaxis])
            for pair,coefficients in zip(self.raw['collision_species'],self.raw['collision_coefficients']):
                a,b=pair;n1=abundances[a][tc,:][:,pc]*density;n2=abundances[b][tc,:][:,pc]*density
                data=coefficients.reshape(len(self.temperature),1,len(self.wave))[tc]
                collision_coeff+=data*(n1*n2)[:,:,np.newaxis]
        total=np.zeros(shape);total+=gas_coeff;total+=rayleigh_coeff;total+=collision_coeff
        return total,dict(gas=gas_coeff,rayleigh=rayleigh_coeff,collision=collision_coeff)

    def forward(self,parameters,full_output=False,gas=True,rayleigh=True,collision=True):
        """Eight parameters: Rs,mass,Rp,T,log10 scatter,log10 Z,log10 cloud Pa,error scale."""
        star,mass,radius,temp,log_scatter,metal,log_cloud,_=parameters;cloud=10.**log_cloud
        if star<=0 or mass<=0:raise AtmosphereError('Nonpositive stellar radius or planet mass')
        if not (200<=temp<=3000):raise AtmosphereError('Temperature outside calibration')
        if not (self.pressure[0]<cloud<=self.pressure[-1]):raise AtmosphereError('Cloud pressure outside calibration')
        abundances=self.abundances(metal);pressure,thermal,radii,dr,mu,composition=self.atmospheric_profiles(temp,abundances,mass,radius,star,cloud)
        tc=bracket_grid(thermal,self.temperature);pc=bracket_grid(pressure,self.pressure,cloud)
        coeff,components=self.absorption(abundances,tc,pc,10.**log_scatter,gas,rayleigh,collision)
        if len(self.temperature[tc])==1:atmospheric=interpn((self.pressure[pc],),coeff[0],pressure)
        else:atmospheric=interpn((self.temperature[tc],self.pressure[pc]),coeff,np.array([thermal,pressure]).T)
        tau=line_of_sight_tau(atmospheric,radii);fraction=1-np.exp(-tau)
        depth=(np.min(radii)/star)**2+2/star**2*fraction.dot(radii[1:]*dr)
        weights=self.raw['stellar_weights'];binned=np.array([np.average(depth[condition],weights=weights[condition]) for condition in self.bin_masks])
        if full_output:return binned,dict(absorption_coeff_atm=atmospheric,tau_los=tau,radii=radii,P_profile=pressure,T_profile=thermal,mu_profile=mu,atm_abundances=composition,unbinned_depths=depth,unbinned_wavelengths=self.wave,stellar_spectrum=weights,components=components)
        return binned


def bracket_grid(target,grid,max_cutoff=np.inf):
    """Original inclusive enclosing calibration nodes, including cutoff boundary."""
    condition=np.zeros(len(grid),dtype=bool);start=None;end=None
    for i in range(len(grid)):
        if start is None:
            if grid[i]>np.min(target):start=i-1
            if grid[i]==np.min(target):start=i
        if end is None:
            if grid[i]>=np.max(target) or grid[i]>=max_cutoff:end=i+1
    condition[start:end]=True
    return condition


from pathlib import Path
import argparse,json,sys
ASSETS=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,8),'best_fit':(1,8),'transit_depth':(1,42)}
GUESS=np.array([1.19*R_SUN,.73*M_JUP,1.4*R_JUP,1200.,0.,0.,4.,1.])
SD=np.array([.02*R_SUN,.04*M_JUP])
def validate(path):
    with np.load(path,allow_pickle=False) as z:
        result={}
        for k,shape in SHAPES.items():
            if k not in z:raise ValueError('Missing required key: '+k)
            a=z[k]
            if a.shape!=shape:raise ValueError('Wrong shape: '+k)
            if a.dtype!=np.dtype('float64'):raise ValueError('Expected real float64: '+k)
            if not np.isfinite(a).all():raise ValueError('Nonfinite: '+k)
            result[k]=a
        return result
def prior(p):
    low=np.array([0,300,0,-1,-.99,0]);high=np.array([np.inf,3000,5,3,5,np.inf])
    if p[0]<=0 or p[1]<=0 or not ((p[2:]>low)&(p[2:]<high)).all():return -np.inf
    return float(-.5*np.sum(((p[:2]-GUESS[:2])/SD)**2)-np.log(np.sqrt(2*np.pi)*SD).sum())
def evaluate(prediction,assets=ASSETS):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        a=validate(prediction);ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        b=validate(ref);rules=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k][0] for k in z.files}
        scale=np.array(rules['parameter_scales'])
        if scale.shape!=(8,) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid parameter comparison scales')
        if not np.isfinite(prior(a['reconstruction'][0])) or not np.isfinite(prior(a['best_fit'][0])):raise ValueError('Parameter values outside declared physical support')
        model=TransmissionModel(raw);spectrum=model.forward(a['best_fit'][0]);residual=spectrum-raw['transit_depth'];errors=raw['uncertainty']*a['best_fit'][0,7]
        measured={'median_standardized_max':float(np.max(abs(a['reconstruction'][0]-b['reconstruction'][0])/scale)),'best_standardized_max':float(np.max(abs(a['best_fit'][0]-b['best_fit'][0])/scale)),'spectrum_max_ppm':float(np.max(abs(a['transit_depth'][0]-b['transit_depth'][0]))*1e6),'unscaled_chi2':float(np.sum((residual/raw['uncertainty'])**2)),'log_posterior_min':float(prior(a['best_fit'][0])-.5*np.sum((residual/errors)**2+np.log(2*np.pi*errors**2))),'spectrum_consistency_ppm':float(np.max(abs(a['transit_depth'][0]-spectrum))*1e6)}
        thresholds={k:dict(comparison='>=' if k=='log_posterior_min' else '<=',boundary=rules[k+'_boundary']) for k in measured}
        if not np.isfinite(list(measured.values())).all() or not all(np.isfinite(v['boundary']) for v in thresholds.values()):raise ValueError('Undefined metric configuration')
        passed=all(v>=thresholds[k]['boundary'] if k=='log_posterior_min' else v<=thresholds[k]['boundary'] for k,v in measured.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds);return report,0 if passed else 1
    except Exception as e:report['errors']=[str(e)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();r,c=evaluate(Path(a.prediction))
    try:
        dest=Path(a.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return c
if __name__=='__main__':sys.exit(main())

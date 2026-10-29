"""Scientific review of measured ellipsometry and freshly computed local fits.

All data and reference measurements are supplied by the active caller. Plotting
never imports an upstream optical solver or selects a Matplotlib backend.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import cauchy_index, predict_rho, reflection

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
PARAMETER_LABELS = ['n0', 'n1', 'n2', 'd']


def observed_rho(data):
    """Measured rho=tan(Psi)*exp(-i*Delta), with angles in degrees."""
    return np.tan(np.deg2rad(data['psi_deg'])) * np.exp(-1j*np.deg2rad(data['delta_deg']))


def rho_angles(rho):
    """Return Psi and signed Delta in degrees, preserving circular phase."""
    rho = np.asarray(rho)
    return np.rad2deg(np.arctan(np.abs(rho))), (-np.rad2deg(np.angle(rho))+180)%360-180


def comparison_metrics(data, rho):
    observed = observed_rho(data)
    psi, delta = rho_angles(rho)
    circular = (delta-data['delta_deg']+180)%360-180
    return dict(relative_complex_rho_rms=float(np.linalg.norm(rho-observed)/np.linalg.norm(observed)),
                psi_rms_deg=float(np.sqrt(np.mean((psi-data['psi_deg'])**2))),
                circular_delta_rms_deg=float(np.sqrt(np.mean(circular**2))))


def export_figure(fig, destination, arrays, metadata):
    """Write matched PNG/SVG/PDF, exact numeric NPZ and provenance JSON."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(destination.with_suffix('.'+suffix), dpi=180, bbox_inches='tight', facecolor='white')
    np.savez_compressed(destination.with_suffix('.npz'), **arrays)
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')


def acquisition_figure(data):
    wavelength = data['wavelength_nm']; rho = observed_rho(data)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout='constrained')
    for ax, values, label in zip(axes.ravel(), [data['psi_deg'], data['delta_deg'], rho.real, rho.imag],
                                 ['Psi [deg]', 'Signed Delta [deg]', 'Real rho', 'Imaginary rho']):
        ax.plot(wavelength, values, color=COLORS[0], lw=1)
        ax.set(xlabel='Wavelength [nm]', ylabel=label, title='Measured: all 926 spectral points')
    fig.suptitle('Real TiO2 / SiO2 / Si acquisition: reported incidence 70.06 degrees', fontsize=13)
    return fig, dict(wavelength_nm=wavelength, psi_deg=data['psi_deg'], delta_deg=data['delta_deg'], rho_observed=rho)


def spectrum_figure(data, fresh_rho, source, initial_rho, source4x4):
    wavelength=data['wavelength_nm']; psi, delta=rho_angles(fresh_rho)
    initial_psi,initial_delta=rho_angles(initial_rho)
    fig, axes=plt.subplots(1, 3, figsize=(16, 4.7), layout='constrained')
    for ax, measured, original, fresh, initial, alternate, ylabel in zip(axes[:2],
            [data['psi_deg'],data['delta_deg']], [source['psi_fitted'],source['delta_fitted']],
            [psi,delta], [initial_psi,initial_delta], [source4x4['psi_fitted'],source4x4['delta_fitted']], ['Psi [deg]','Signed Delta [deg]']):
        ax.plot(wavelength, measured, '.', color='#666666', ms=2, label='Real observation')
        ax.plot(wavelength, initial, ':', color=COLORS[2], lw=1.1, label='Initial parameters: local forward')
        ax.plot(wavelength, original, color=COLORS[1], lw=1.6, label='Original source 2x2 fit')
        ax.plot(wavelength, alternate, 'x', markevery=70, color=COLORS[3], ms=3, label='Original source 4x4 fit')
        ax.plot(wavelength, fresh, '--', color=COLORS[0], lw=1.1, label='Fresh local fit')
        ax.set(xlabel='Wavelength [nm]',ylabel=ylabel,title='Identical complete spectral window')
    obs=observed_rho(data)
    axes[2].plot(obs.real,obs.imag,'.',color='#666666',ms=2,label='Real observation')
    axes[2].plot(initial_rho.real,initial_rho.imag,':',color=COLORS[2],lw=1,label='Initial parameters: local forward')
    axes[2].plot(source['rho_fitted'].real,source['rho_fitted'].imag,color=COLORS[1],lw=1.6,label='Original source fit')
    axes[2].plot(source4x4['rho_fitted'].real,source4x4['rho_fitted'].imag,'x',markevery=70,color=COLORS[3],ms=3,label='Original source 4x4 fit')
    axes[2].plot(fresh_rho.real,fresh_rho.imag,'--',color=COLORS[0],lw=1.1,label='Fresh local fit')
    axes[2].set(xlabel='Real rho',ylabel='Imaginary rho',title='Complex reflection-ratio trajectory')
    axes[0].legend(fontsize=8)
    return fig, dict(wavelength_nm=wavelength,rho_observed=obs,source_rho=source['rho_fitted'],fresh_rho=fresh_rho,
                     psi_observed=data['psi_deg'],delta_observed=data['delta_deg'],source_psi=source['psi_fitted'],
                     source_delta=source['delta_fitted'],fresh_psi=psi,fresh_delta=delta,initial_rho=initial_rho,initial_psi=initial_psi,
                     initial_delta=initial_delta,source4x4_rho=source4x4['rho_fitted'],source4x4_parameters=source4x4['parameters'])


def residual_figure(data, fresh_rho, source_rho):
    wavelength=data['wavelength_nm']; obs=observed_rho(data)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained'); arrays={'wavelength_nm':wavelength}
    for label,rho,color,style in [('Original',source_rho,COLORS[1],'-'),('Fresh',fresh_rho,COLORS[0],'--')]:
        psi,delta=rho_angles(rho)
        residuals=[psi-data['psi_deg'],(delta-data['delta_deg']+180)%360-180,(rho-obs).real,(rho-obs).imag]
        for ax,values,name in zip(axes.ravel(),residuals,['Psi residual [deg]','Circular Delta residual [deg]','Real-rho residual','Imaginary-rho residual']):
            ax.plot(wavelength,values,style,color=color,lw=1,label=label)
            ax.set(xlabel='Wavelength [nm]',ylabel=name)
            ax.axhline(0,color='#555555',lw=.6)
            arrays[label+'_'+name.split(' [')[0].replace(' ','_')]=values
    metrics=comparison_metrics(data,fresh_rho)
    fig.suptitle(f"Fresh residuals: relative rho RMS {metrics['relative_complex_rho_rms']:.4%}; "
                 f"Psi {metrics['psi_rms_deg']:.3f} deg; circular Delta {metrics['circular_delta_rms_deg']:.3f} deg",fontsize=12)
    axes[0,0].legend(fontsize=8)
    return fig, arrays, metrics


def scaled_jacobian(data, parameters, angle_deg=70.):
    """Centered descriptive sensitivity, with parameter-relative columns."""
    parameters=np.asarray(parameters,dtype=float); scale=np.maximum(abs(parameters),1.)
    columns=[]
    for j in range(4):
        step=1e-5*scale[j]; plus=parameters.copy();minus=parameters.copy()
        plus[j]+=step;minus[j]-=step
        change=(predict_rho(data,plus,angle_deg=angle_deg)-predict_rho(data,minus,angle_deg=angle_deg))/(2*step)*scale[j]
        columns.append(np.r_[change.real,change.imag])
    return np.asarray(columns).T


def properties_figure(data, parameters, source_parameters, source_covariance, source_singular_values):
    wavelength=data['wavelength_nm']; n=cauchy_index(wavelength,parameters);original_n=cauchy_index(wavelength,source_parameters)
    jac=scaled_jacobian(data,parameters);singular=np.linalg.svd(jac,compute_uv=False)
    covariance=np.asarray(source_covariance);sd=np.sqrt(np.diag(covariance));correlation=covariance/np.outer(sd,sd)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    axes[0,0].plot(wavelength,original_n,color=COLORS[1],lw=2,label='Original source')
    axes[0,0].plot(wavelength,n,'--',color=COLORS[0],label='Fresh local')
    axes[0,0].set(xlabel='Wavelength [nm]',ylabel='TiO2 refractive index n',title='Transparent-film dispersion, k = 0');axes[0,0].legend(fontsize=8)
    axes[0,1].bar(['Original source','Fresh local'],[source_parameters[3],parameters[3]],color=[COLORS[1],COLORS[0]])
    axes[0,1].set(ylabel='TiO2 thickness [nm]',title='Absolute thickness: no fitted rescaling')
    for x,value in enumerate([source_parameters[3],parameters[3]]):axes[0,1].text(x,value*.5,f'{value:.8f} nm',ha='center',rotation=90,color='white')
    axes[1,0].semilogy(np.arange(1,5),source_singular_values,'o-',color=COLORS[1],label='Independent source diagnostic')
    axes[1,0].semilogy(np.arange(1,5),singular,'x--',color=COLORS[0],label='Fresh local diagnostic')
    axes[1,0].set(xlabel='Ordered sensitivity mode',ylabel='Singular value',title=f'Scaled Jacobian: rank {np.linalg.matrix_rank(jac)}, condition {singular[0]/singular[-1]:.1f}');axes[1,0].set_xticks([1,2,3,4]);axes[1,0].legend(fontsize=7)
    im=axes[1,1].imshow(correlation,vmin=-1,vmax=1,cmap='RdBu_r');axes[1,1].set_xticks(range(4),PARAMETER_LABELS);axes[1,1].set_yticks(range(4),PARAMETER_LABELS)
    axes[1,1].set(title='Original source conditional parameter correlation')
    for i in range(4):
        for j in range(4):axes[1,1].text(j,i,f'{correlation[i,j]:.2f}',ha='center',va='center',color='white' if abs(correlation[i,j])>.6 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes[1,1],label='Correlation coefficient')
    return fig,dict(wavelength_nm=wavelength,fresh_index=n,original_index=original_n,fresh_parameters=parameters,source_parameters=source_parameters,
                    fresh_scaled_jacobian=jac,fresh_singular_values=singular,source_singular_values=source_singular_values,source_covariance=covariance,source_correlation=correlation)


def multistart_figure(data, starts, solutions, labels, objectives, evaluations):
    starts=np.asarray(starts);solutions=np.asarray(solutions);wavelength=data['wavelength_nm']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    axes[0,0].scatter(starts[:,3],solutions[:,3],s=55,c=COLORS[:len(labels)])
    for i,label in enumerate(labels):
        rightmost=starts[i,3]==np.max(starts[:,3])
        axes[0,0].annotate(label,(starts[i,3],solutions[i,3]),xytext=(-5 if rightmost else 4,5),
                           textcoords='offset points',ha='right' if rightmost else 'left',fontsize=8)
    axes[0,0].set(xlabel='Initial thickness [nm]',ylabel='Fitted thickness [nm]',title='Actually computed alternate initializations')
    padding=max(.05,float(np.ptp(solutions[:,3]))*.15)
    axes[0,0].set_ylim(float(solutions[:,3].min())-padding,float(solutions[:,3].max())+padding)
    axes[0,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0,1].bar(labels,objectives,color=COLORS[:len(labels)]);axes[0,1].set(ylabel='Sum of squared complex-rho residuals',title='Same unweighted selected objective')
    indices=[]
    for parameters,label,color in zip(solutions,labels,COLORS):
        index=cauchy_index(wavelength,parameters);indices.append(index);axes[1,0].plot(wavelength,index,label=label,color=color,lw=1)
    axes[1,0].set(xlabel='Wavelength [nm]',ylabel='Refractive index n',title='All fitted dispersion curves');axes[1,0].legend(fontsize=8)
    axes[1,1].bar(labels,evaluations,color=COLORS[:len(labels)]);axes[1,1].set(ylabel='Reported function evaluations',title='Evaluation counts, not optimizer iterations')
    return fig,dict(wavelength_nm=wavelength,starts=starts,solutions=solutions,labels=np.asarray(labels,dtype='U32'),objective_sums=np.asarray(objectives),function_evaluations=np.asarray(evaluations),indices=np.asarray(indices))


def sensitivity_figure(data, parameters, angles_deg, source_rho):
    wavelength=data['wavelength_nm'];angles=np.asarray(angles_deg);standard=predict_rho(data,parameters,angle_deg=70.)
    actual=predict_rho(data,parameters,angle_deg=70.06);errors=[]
    for angle in angles:errors.append(comparison_metrics(data,predict_rho(data,parameters,angle_deg=float(angle)))['relative_complex_rho_rms'])
    psi0,delta0=rho_angles(standard);psi1,delta1=rho_angles(actual)
    rs,rp=reflection(wavelength,parameters,data['silicon_index'],angle_deg=70.)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    axes[0,0].plot(angles,np.asarray(errors)*100,color=COLORS[0]);axes[0,0].axvline(70,color=COLORS[1],ls='--',label='Original model: 70 deg');axes[0,0].axvline(70.06,color=COLORS[2],ls=':',label='Reported acquisition: 70.06 deg')
    axes[0,0].set(xlabel='Forward-model incidence [deg]',ylabel='Relative complex-rho RMS [%]',title='Held fitted parameters: no refit or new data');axes[0,0].legend(fontsize=7)
    axes[0,1].plot(wavelength,psi1-psi0,color=COLORS[0],label='Psi change');axes[0,1].plot(wavelength,(delta1-delta0+180)%360-180,color=COLORS[1],label='Circular Delta change')
    axes[0,1].set(xlabel='Wavelength [nm]',ylabel='70.06 minus 70 degree prediction [deg]',title='Effect of the documented angle discrepancy');axes[0,1].legend(fontsize=8)
    axes[1,0].plot(wavelength,abs(rs)**2,color=COLORS[0],label='s polarization');axes[1,0].plot(wavelength,abs(rp)**2,color=COLORS[1],label='p polarization')
    axes[1,0].set(xlabel='Wavelength [nm]',ylabel='Reflectance',title='Fresh passive-film reflection');axes[1,0].set_ylim(0,1);axes[1,0].legend(fontsize=8)
    difference=standard-source_rho
    axes[1,1].plot(wavelength,difference.real,color=COLORS[0],label='Real part');axes[1,1].plot(wavelength,difference.imag,color=COLORS[1],label='Imaginary part')
    axes[1,1].set(xlabel='Wavelength [nm]',ylabel='Fresh minus original rho',title='Implementation parity at the selected 70 deg');axes[1,1].legend(fontsize=8)
    return fig,dict(wavelength_nm=wavelength,angles_deg=angles,relative_rho_rms=np.asarray(errors),rho_at_model_angle=standard,rho_at_reported_angle=actual,
                    psi_change_deg=psi1-psi0,circular_delta_change_deg=(delta1-delta0+180)%360-180,reflectance_s=abs(rs)**2,reflectance_p=abs(rp)**2,source_rho=source_rho,parity_difference=difference)

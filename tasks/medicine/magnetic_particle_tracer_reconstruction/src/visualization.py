"""Full-grid MPI comparisons in calibrated physical coordinates."""
import numpy as np
import matplotlib.pyplot as plt
EXTENTS=[[-19,19,-19,19],[-19,19,-9.5,9.5],[-19,19,-9.5,9.5]]
AXES=[('x (mm)','y (mm)'),('x (mm)','z (mm)'),('y (mm)','z (mm)')]

def image_panel(ax,image,extent,vmax,title):
    artist=ax.imshow(image,origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='viridis',aspect='equal',interpolation='nearest')
    ax.set_title(title);return artist

def figures(canonical,upstream,measurement,predicted,frequency,channel,experiments):
    """Return four distinct figures and all exact plotted numeric arrays."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    x=canonical[0];ref=upstream[0];fs=[];data={'fresh_reconstruction':canonical,'upstream_reconstruction':upstream,'measurement':measurement,'predicted':predicted,'frequency_Hz':frequency,'channel':channel}
    fig,axes=plt.subplots(2,3,figsize=(11.5,7),layout='constrained');vmax=x.max()
    for j in range(3):
        im=image_panel(axes[0,j],x.max(axis=j),EXTENTS[j],vmax,['MIP along z','MIP along y','MIP along x'][j]);axes[0,j].set(xlabel=AXES[j][0],ylabel=AXES[j][1])
        sl=[x[9,:,:],x[:,9,:],x[:,:,9]][j];image_panel(axes[1,j],sl,EXTENTS[j],vmax,['Central z=0 mm','Central y=0 mm','Central x=0 mm'][j]);axes[1,j].set(xlabel=AXES[j][0],ylabel=AXES[j][1])
    fig.colorbar(im,ax=axes,label='Tracer amount (0.4 µmol Fe / unit)',shrink=.75);fs.append(('full_grid_tracer_projections',fig))
    fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained');row=np.arange(len(measurement))
    for ax,part,name in [(axes[0,0],np.real,'Real'),(axes[0,1],np.imag,'Imaginary')]:
        ax.plot(row,part(measurement),'.',ms=2,label='Measured average',color='#0072B2');ax.plot(row,part(predicted),'.',ms=1.5,alpha=.6,label='Fresh prediction',color='#D55E00');ax.set(xlabel='Selected channel/frequency row',ylabel=name+' amplitude (a.u.)',title=name+' component of acquisition');ax.legend(fontsize=8)
        for edge in np.flatnonzero(np.diff(channel))+1:ax.axvline(edge,color='gray',ls=':',lw=.7)
    axes[1,0].scatter(ref.ravel(),x.ravel(),s=3,alpha=.4,color='#009E73');axes[1,0].plot([0,x.max()],[0,x.max()],'k--',lw=1);axes[1,0].set(xlabel='Independent original amount (delta units)',ylabel='Fresh amount (delta units)',title=f'All 6859 voxels: max difference {np.max(abs(x-ref)):.2g}')
    residual=predicted-measurement
    bins=np.histogram_bin_edges(np.r_[residual.real,residual.imag],bins=70)
    axes[1,1].hist(residual.real,bins=bins,alpha=.55,label='Real residual');axes[1,1].hist(residual.imag,bins=bins,alpha=.55,label='Imaginary residual');axes[1,1].set_yscale('log');axes[1,1].axvline(0,color='black',lw=.8);axes[1,1].set(xlabel='Signed complex component residual (a.u.)',ylabel='Selected-row count (log scale)',title='All measured rows; no fitted rescaling');axes[1,1].legend()
    fs.append(('original_parity_complex_data_fidelity',fig));data['complex_residual']=residual
    fig,axes=plt.subplots(2,3,figsize=(12,7.2),layout='constrained');hist=experiments['sweep_images'];vmax=hist[[0,2,9]].max()
    for j,k in enumerate([0,2,9]):
        im=image_panel(axes[0,j],hist[k].max(axis=0),EXTENTS[0],vmax,f'{k+1} complete sweep(s) / XY MIP');axes[0,j].set(xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axes[0,:],label='Amount (delta units)',shrink=.8)
    axes[1,0].plot(np.arange(1,11),experiments['sweep_forward_residual'],'o-');axes[1,0].set(xlabel='Complete sweeps',ylabel='Complex forward relative L2',title='Finite-sweep / projection behavior')
    axes[1,1].semilogx(experiments['lambda_multipliers'],experiments['lambda_forward_residual'],'o-',color='#D55E00');axes[1,1].set(xlabel='Damping / canonical damping',ylabel='Complex forward relative L2',title='Computed damping sensitivity')
    axes[1,2].semilogx(experiments['lambda_multipliers'],experiments['lambda_nrmse'],'o-',color='#009E73');axes[1,2].set(xlabel='Damping / canonical damping',ylabel='NRMSE versus original method',title='Method sensitivity, not physical truth')
    fs.append(('finite_sweeps_damping_sensitivity',fig))
    fig,axes=plt.subplots(2,3,figsize=(12,7.2),layout='constrained');avg=experiments['average_images'];nobg=experiments['no_measurement_background_image'];vmax=max(avg.max(),nobg.max())
    for j,k in enumerate([0,1,3]):
        im=image_panel(axes[0,j],avg[k].max(axis=0),EXTENTS[0],vmax,f'{experiments["average_counts"][k]} foreground cycle(s) / XY MIP');axes[0,j].set(xlabel='x (mm)',ylabel='y (mm)')
    image_panel(axes[1,0],nobg.max(axis=0),EXTENTS[0],vmax,'1000 cycles / measurement BG omitted');axes[1,0].set(xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=[*axes[0,:],axes[1,0]],label='Amount (delta units); shared full range',shrink=.8)
    axes[1,1].semilogx(experiments['average_counts'],experiments['average_nrmse'],'o-',color='#0072B2');axes[1,1].set(xlabel='Foreground cycles averaged',ylabel='NRMSE versus original1000cycle method',title='All 1000 background cycles retained')
    coord=(np.arange(19)-9)*2;axes[1,2].plot(coord,x[9,9],label='Full background corrected');axes[1,2].plot(coord,nobg[9,9],label='Measurement BG omitted');axes[1,2].set(xlabel='x at y=z=0 (mm)',ylabel='Amount (delta units)',title='Independent-background effect');axes[1,2].legend(fontsize=8)
    fs.append(('cycle_averaging_background_controls',fig))
    for key,value in experiments.items():data[key]=value
    return fs,data

import numpy as np
import matplotlib.pyplot as plt
def rms(values):return float(np.sqrt(np.mean(np.asarray(values)**2)))
def make_figures(data,fresh,original,sensitivity):
    positions=data['sock_vertices'];electrodes=data['torso_vertices'][data['electrode_indices'].ravel()]
    fig=plt.figure(figsize=(14,5),layout='constrained')
    ax=fig.add_subplot(131,projection='3d');im=ax.scatter(*electrodes.T,c=data['potentials'].ravel(),cmap='RdBu_r',s=16);fig.colorbar(im,ax=ax,shrink=.6,label='Native potential unit');ax.set(title='All191recorded jacket potentials',xlabel='x(mm)',ylabel='y(mm)',zlabel='z(mm)');ax.set_box_aspect(np.ptp(electrodes,axis=0))
    limit=np.max(np.abs(original))
    for panel,values,title in [(132,original.ravel(),'Independent original MFS reference'),(133,fresh['reconstruction'].ravel(),'Fresh independent-equation MFS')]:
        ax=fig.add_subplot(panel,projection='3d');im=ax.scatter(*positions.T,c=values,cmap='RdBu_r',vmin=-limit,vmax=limit,s=20);fig.colorbar(im,ax=ax,shrink=.6,label='Native potential unit');ax.set(title=title,xlabel='x(mm)',ylabel='y(mm)',zlabel='z(mm)');ax.set_box_aspect(np.ptp(positions,axis=0))
    fig2,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');n=191;prediction=fresh['predicted_cauchy'];observed=data['potentials'].ravel()
    axes[0,0].plot(observed,label='All measured jacket potentials');axes[0,0].plot(prediction[:n],'--',label='Fresh model prediction');axes[0,0].set(title='Cauchy data consistency',xlabel='Electrode index',ylabel='Native potential unit');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(prediction[:n]-observed);axes[0,1].set(title=f'Measured-boundary residual RMS {rms(prediction[:n]-observed):.3g}',xlabel='Electrode index',ylabel='Native potential unit')
    axes[1,0].plot(prediction[n:]);axes[1,0].axhline(0,color='black',lw=.7);axes[1,0].set(title='All191zero-normal-flux constraints',xlabel='Electrode index',ylabel='Native potential unit / mm')
    axes[1,1].plot(fresh['reconstruction'].ravel()-original.ravel());axes[1,1].set(title='Fresh minus independently run reference',xlabel='Sock electrode index',ylabel='Native potential difference')
    fig3,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    axes[0].semilogx(fresh['lambdas'],fresh['creso']/np.max(fresh['creso']));axes[0].set_yscale('symlog',linthresh=.1);axes[0].axvline(fresh['selected_lambda'],color='red',ls='--',label='Complete CRESO maximum');axes[0].set(title='All10000original-grid candidates',xlabel='Regularization parameter lambda',ylabel='CRESO / maximum (symmetric log)');axes[0].legend(fontsize=7)
    axes[1].semilogy(fresh['singular_values']);axes[1].axhline(fresh['selected_lambda'],color='red',ls='--');axes[1].set(title='Ill-conditioned Cauchy continuation',xlabel='Singular-value index',ylabel='Singular value')
    for factor,values in sensitivity.items():axes[2].plot(values,label=f'Fresh lambda x{factor:g}')
    axes[2].set(title='Regularization dependence, no cardiac GT',xlabel='Sock electrode index',ylabel='Native potential unit');axes[2].legend(fontsize=7)
    return [fig,fig2,fig3]

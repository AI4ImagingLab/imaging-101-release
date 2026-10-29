"""Published native pseudoinverse, explicit local math; no domain reconstruction API."""
import numpy as np
from .physics_model import design_matrix,coefficients_to_fields,components,relative_residual
from .preprocessing import prepare_intensity
from .generate_data import add_gaussian_perturbation

def reconstruct(intensity,polarizer_psg,polarizer_psa):
    """Full native least squares; returns fields and original linear coefficient maps."""
    images=prepare_intensity(intensity)
    W=design_matrix(polarizer_psg,polarizer_psa)
    if len(W)!=len(images) or np.linalg.matrix_rank(W)<5:raise ValueError('Rank-five acquisition required')
    inverse=np.linalg.pinv(W)
    # Preserve original's(row,column,frame) layout and multiplication order.
    X=np.tensordot(inverse,np.moveaxis(images,0,-1).astype(np.float64),axes=(1,-1))
    return {'reconstruction':coefficients_to_fields(X),'coefficients':X,'design':W,'pseudoinverse':inverse}

def unrestricted_mueller(intensity,polarizer_psg,polarizer_psa):
    """Review-only9basis control; does not replace the native5field answer."""
    W=design_matrix(polarizer_psg,polarizer_psa,complete=True)
    if np.linalg.matrix_rank(W)<9:raise ValueError('Rank-nine acquisition required')
    X=np.tensordot(np.linalg.pinv(W),np.moveaxis(prepare_intensity(intensity),0,-1).astype(np.float64),axes=(1,-1))
    return {'coefficients':X,'design':W}

def review_comparisons(data,fresh):
    """Actively compute all-full-image scientific controls; never used by standard CLI."""
    y=data['intensity'];g=data['polarizer_psg'];c=data['polarizer_psa'];native=fresh['reconstruction']
    selection={5:[5,7,9,10,14],8:list(range(8)),12:list(range(12)),16:list(range(16))}
    support_fields={16:native};support_metrics={}
    for count,indices in selection.items():
        fields=native if count==16 else reconstruct(y[indices],g[indices],c[indices])['reconstruction']
        support_fields[count]=fields
        support_metrics[count]={'channel_relative_l2':[float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a,b in zip(fields,native)],
            'all_state_prediction_relative_l2':relative_residual(fields,y,g,c)['relative_l2'],'selected_indices':indices,'condition':float(np.linalg.cond(design_matrix(g[indices],c[indices])))}
    complete=unrestricted_mueller(y,g,c);rms9=np.empty(y.shape[1:]);e9=np.zeros(16);n9=np.zeros(16)
    for k in range(0,y.shape[1],64):
        observed=np.clip(y[:,k:k+64],0,None).astype(np.float64)
        residual=np.tensordot(complete['design'],complete['coefficients'][:,k:k+64],axes=(1,0))-observed
        rms9[k:k+64]=np.sqrt(np.mean(residual**2,axis=0));e9+=(residual**2).sum(axis=(1,2));n9+=(observed**2).sum(axis=(1,2))
    del complete
    frame_rms=np.sqrt(np.mean(y.astype(np.float64)**2,axis=(1,2)));noise_metrics={};noise_amplitudes={}
    for fraction in (.005,.01,.02):
        noisy=add_gaussian_perturbation(y,frame_rms,fraction,seed=31917)
        clipped_fraction=float(np.mean(noisy<0))
        fields=reconstruct(noisy,g,c)['reconstruction']
        noise_amplitudes[fraction]=components(fields)['sum_amplitude']
        noise_metrics[fraction]={'channel_relative_l2':[float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a,b in zip(fields,native)],
            'source_frame_RMS':list(frame_rms),'newly_lower_clipped_fraction':clipped_fraction,'noise_std_fraction_of_frame_RMS':fraction,
            'prediction_error_against_unperturbed_real_observations':relative_residual(fields,y,g,c)['relative_l2']}
        del noisy,fields
    return {'support_fields':support_fields,'support_metrics':support_metrics,'rms9':rms9,'frame_relative9':np.sqrt(e9/n9),
            'relative9':float(np.sqrt(e9.sum()/n9.sum())),'noise_metrics':noise_metrics,'noise_sum_amplitudes':noise_amplitudes}

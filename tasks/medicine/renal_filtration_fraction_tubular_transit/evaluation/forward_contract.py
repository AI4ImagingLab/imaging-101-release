"""Independent calibrated GRE/filtration observation model, no solver imports."""
import numpy as np

def exponential_average(values,time,transit):
    if transit==0:return values.copy()
    output=np.zeros_like(values,dtype=float)
    for i,dt in enumerate(np.diff(time)):
        x=dt/transit;e=np.exp(-x);one=-np.expm1(-x)
        output[i+1]=e*output[i]+values[i]*one+(values[i+1]-values[i])*(1-one/x)
    return output

def predicted_signal(parameters,data,index):
    length=int(data['lengths'][0,index]);time=data['time'][0,index,:length]
    observed=data['measurements'][0,index,:length];aorta=data['arterial_signal'][0,index,:length]
    n0=int(data['baseline_count'][0,index]);tr=float(data['tr'][0,index]);angle=np.deg2rad(data['flip_angle'][0,index])
    r10=1/float(data['blood_t1'][0,index]);relaxivity=float(data['relaxivity'][0,index]);cosine=np.cos(angle)
    baseline_e=np.exp(-tr*r10);normalized=aorta/np.mean(aorta[:n0])*(1-baseline_e)/(1-cosine*baseline_e)
    saturated=normalized>=1
    if np.any(saturated):
        if np.all(saturated):raise ValueError('No invertible arterial calibration values')
        indices=np.arange(length);normalized[saturated]=np.interp(indices[saturated],indices[~saturated],normalized[~saturated])
    ratio=(1-normalized)/(1-cosine*normalized)
    rate=np.where(ratio==0,0,-np.log(ratio)/tr)
    arterial=(rate-r10)/relaxivity/.55
    flow,volume,fraction,transit=parameters
    if flow==0:plasma=tubular=np.zeros(length)
    elif volume==0:
        plasma=np.zeros(length);tubular=transit*exponential_average(flow*fraction/(1+fraction)*arterial,time,transit)
    else:
        plasma_time=volume/(flow*(1+fraction))
        plasma=plasma_time*exponential_average(flow*arterial,time,plasma_time)
        tubular=transit*exponential_average(flow*fraction*plasma/volume,time,transit)
    tissue_r10=1/float(data['tissue_t1'][0,index]);rate=tissue_r10+relaxivity*(plasma+tubular)
    e=np.exp(-tr*rate);e0=np.exp(-tr*tissue_r10)
    return np.mean(observed[:n0])*np.abs((1-e)/(1-cosine*e))/((1-e0)/(1-cosine*e0))

def physical_metrics(prediction,data,budgets):
    p=prediction['reconstruction'][0];s=prediction['signal'][0]
    if np.any(p[:,:3]<0) or np.any(p[:,:3]>[.05,.3,.3]) or np.any(p[:,3]<30):raise ValueError('Parameters outside declared physical analysis domain')
    identity=[];quality=[];relative=[];padding=[]
    for i,length in enumerate(data['lengths'][0]):
        model=predicted_signal(p[i],data,i);observed=data['measurements'][0,i,:length]
        norm=np.linalg.norm(observed)
        identity.append(np.linalg.norm(s[i,:length]-model)/np.linalg.norm(model))
        residual=np.linalg.norm(model-observed)/norm
        quality.append(residual**2/budgets[i]);relative.append(residual)
        padding.extend(abs(s[i,length:]))
    return dict(signal_model_identity_max_relative_rmse=float(max(identity)),maximum_fit_quality_budget_ratio=float(max(quality)),signal_observation_max_relative_rmse=float(max(relative)),padding_max=float(max(padding,default=0.)))

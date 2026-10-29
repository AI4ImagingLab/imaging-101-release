"""Standalone evaluator; never imports solver or submitted code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = (1,128,128,128)


def read_volume(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z: raise ValueError('Missing reconstruction key')
        a = z['reconstruction']
    if a.shape!=SHAPE: raise ValueError('Expected shape '+str(SHAPE))
    if a.dtype!=np.dtype('float32'): raise ValueError('Expected real float32 dtype')
    if not np.isfinite(a).all(): raise ValueError('Non-finite reconstruction')
    return a


def agreement(estimate, reference):
    x = estimate.astype(np.float64).ravel();r = reference.astype(np.float64).ravel()
    dr = np.ptp(r);nx = np.linalg.norm(x);nr = np.linalg.norm(r)
    if dr<=0 or nr<=0: raise ValueError('Degenerate evaluator reference')
    ncc = float(np.dot(x,r)/(nx*nr)) if nx else 0.
    return {'ncc':ncc,'nrmse':float(np.sqrt(np.mean((x-r)**2))/dr),
            'max_error_over_reference_range':float(np.max(abs(x-r))/dr)}


def observation_arrays(estimate, raw_path):
    """Independent explicit forward evaluation using only public calibration.

    Its homogeneous-coordinate construction and separable Gaussian directly
    evaluate the declared operator; it imports no reference reconstruction code.
    """
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    device = 'cpu'
    with np.load(raw_path,allow_pickle=False) as z:
        windows = torch.tensor(z['energy_windows'][0],dtype=torch.float32,device=device)
        mu = torch.tensor(z['attenuation'][0],device=device)
        phis = (270-torch.tensor(z['angles'][0],device=device)).tolist()
        radii = z['radii'][0]
    dx = .47951998710631993;size=182
    image = F.pad(torch.tensor(estimate[0],device=device),[0,0,27,27,27,27])
    mu = F.pad(mu,[0,0,27,27,27,27])
    kernels = {}
    for radius in np.unique(radii):
        d = np.maximum(radius+((size-1)/2-np.arange(size))*dx,0)
        sig = np.sqrt((.03161982752665455*d+.1248503046423388)**2+.16137114205472364**2)/dx
        k = int(2*np.ceil(3*max(sig))+1)
        coordinates = torch.arange(-(k//2),k//2+1,device=device)[None,None].repeat(size,1,1)
        sig = torch.tensor(sig,dtype=torch.float32,device=device).reshape(size,1,1)
        kernel = torch.exp(-coordinates**2/(2*sig**2+1e-11))
        kernels[radius] = kernel/kernel.sum(-1).unsqueeze(-1)
    predictions=[]
    for phi,radius in zip(phis,radii):
        angle = torch.tensor([-phi],device=device)*torch.tensor(np.pi,dtype=torch.float32,device=device)/180
        c,s = torch.cos(angle),torch.sin(angle)
        shift = torch.eye(3,device=device)[None].clone();shift[:,:2,2]=90.5
        unshift = torch.eye(3,device=device)[None].clone();unshift[:,:2,2]=-90.5
        rotation = torch.eye(3,device=device)[None].clone()
        rotation[:,0,0]=c;rotation[:,0,1]=s;rotation[:,1,0]=-s;rotation[:,1,1]=c
        matrix = shift@rotation@torch.eye(3,device=device)[None]@unshift
        normal = torch.tensor([[[1.,0.,-1.],[0.,1.,-1.],[0.,0.,1.]]])
        normal[:,0,0]=normal[:,0,0]*2/181;normal[:,1,1]=normal[:,1,1]*2/181
        normal=normal.to(device)
        grid=F.affine_grid(torch.inverse(normal@(matrix@torch.linalg.inv(normal)))[:,:2,:],[1,1,182,182],align_corners=True)
        def rotate(a):
            return F.grid_sample(a.permute(2,0,1)[None],grid,align_corners=True)[0].permute(1,2,0)
        rotated_mu=rotate(mu)*dx
        path=torch.flip(torch.cumsum(torch.flip(rotated_mu,[0]),0),[0])-rotated_mu/2
        u=rotate(image)*torch.exp(-path)
        u=F.conv1d(u.permute(2,0,1),kernels[radius],groups=size,padding='same')
        u=F.conv1d(u.permute(2,1,0),kernels[radius],groups=size,padding='same')
        u=u.permute(2,1,0).permute(1,2,0)
        predictions.append(u.sum(0)[27:-27])
    widths=[228.80000305176-187.19999694824,187.19999694824-166.39999389648,249.60000610352-228.80000305176]
    scatter=(windows[1]/widths[1]*.5+windows[2]/widths[2]*.5)*widths[0]
    return torch.stack(predictions).double().cpu().numpy(),scatter.double().cpu().numpy(),windows[0].double().cpu().numpy()


def poisson_deviance(predicted,observed):
    if np.any(predicted<0) or not np.isfinite(predicted).all():raise ValueError('Invalid predicted counts')
    # Exact zero-observed convention; positive observations and zero predictions
    # receive a finite, rejecting log-tiny penalty for valid JSON reporting.
    positive=observed>0
    terms=predicted-observed
    terms[positive]+=observed[positive]*(np.log(observed[positive])-np.log(np.maximum(predicted[positive],np.finfo(float).tiny)))
    return float(2*np.mean(terms)/np.mean(observed))


def observation_metrics(estimate,raw_path):
    signal,scatter,observed=observation_arrays(estimate,raw_path)
    predicted=signal+scatter
    return {'poisson_deviance':poisson_deviance(predicted,observed),
            'forward_relative_L2':float(np.linalg.norm(predicted-observed)/np.linalg.norm(observed))}


def observation_error(estimate,raw_path):
    return observation_metrics(estimate,raw_path)['forward_relative_L2']


def evaluate(prediction,reference_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        folder=Path(reference_dir);cfg=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        for key in ['poisson_deviance_boundary','forward_relative_L2_boundary']:
            v=cfg[key]
            if type(v) not in (int,float) or not np.isfinite(v) or v<0:raise ValueError('Invalid metric boundary '+key)
        if type(cfg['negative_count_boundary']) is not int or cfg['negative_count_boundary']!=0:raise ValueError('Physical negative-count boundary must equal zero')
        if (folder/'ground_truth.npz').exists():raise ValueError('Unexpected actual truth requires explicit truth-quality protocol certification')
        ref=read_volume(folder/'baseline_reference.npz')
        if np.any(ref<0):raise ValueError('Nonphysical negative evaluator reference')
        a=read_volume(prediction);negative=int(np.count_nonzero(a<0))
        if negative:
            report.update(status='fail',passed=False,metrics={'negative_count':negative},thresholds={'negative_count':dict(comparison='<=',boundary=0)})
            return report,1
        measured=observation_metrics(a,folder.parent/'data/raw_data.npz');measured['negative_count']=negative
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Non-finite measured metric')
        thresholds={k:dict(comparison='<=',boundary=cfg[k+'_boundary']) for k in measured}
        passed=all(measured[k]<=d['boundary'] for k,d in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];report['metrics']={};return report,2


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code


if __name__=='__main__':
    raise SystemExit(main())

"""Standalone complete signed-field and public-observation consistency evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np

def load_field(path):
    if not Path(path).is_file():raise ValueError('Missing NPZ '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if 'reconstruction' not in f:raise ValueError('Missing reconstruction key')
        x=f['reconstruction']
        if x.shape!=(1,144,144) or x.dtype.kind!='f' or x.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64(1,144,144)')
        if not np.isfinite(x).all():raise ValueError('Nonfinite reconstruction')
        return x.astype(np.float64)

def config(path):
    c=json.loads(Path(path).read_text(encoding='utf8'))
    for key in ['ncc_boundary','nrmse_boundary','forward_range_rms_boundary']:
        v=c[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v):raise ValueError('Invalid threshold '+key)
        if key=='ncc_boundary' and not -1<=v<=1:raise ValueError('NCC boundary outside[-1,1]')
        if key!='ncc_boundary' and v<0:raise ValueError('Negative error boundary')
    return c

def reference_metrics(x,y):
    sx=float(np.max(np.abs(x)));sy=float(np.max(np.abs(y)));span=float(np.ptp(y))
    if sy==0 or span==0 or not np.isfinite(span):raise ValueError('Degenerate reference')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        cosine=0. if sx==0 else float(np.vdot(x/sx,y/sy).real/(np.linalg.norm(x/sx)*np.linalg.norm(y/sy)))
        error=float(np.sqrt(np.mean(((x-y)/span)**2)))
    if not np.isfinite(cosine) or not np.isfinite(error):raise ValueError('Nonfinite metric')
    return float(np.clip(cosine,-1,1)),error

def public_observation_model(path):
    """Independent explicit calibration; never import submitted/local solver code."""
    with np.load(path,allow_pickle=False) as f:
        wave=f['wavelength'];flux=f['spectrum'];ph=f['orbital_phase']
    if wave.shape!=(1,28,2048) or flux.shape!=wave.shape or ph.shape!=(1,28) or any(a.dtype.kind!='f' or not np.isfinite(a).all() for a in [wave,flux,ph]):raise ValueError('Malformed evaluator observations')
    if np.any(np.diff(wave,axis=-1)<=0):raise ValueError('Nonincreasing evaluator wavelength')
    common=wave[0,0];spectra=np.stack([flux[0,0]]+[np.interp(common,wave[0,i],flux[0,i]) for i in range(1,28)])
    b=36./299800.;rest=common/((1+b)*np.sqrt(1-b*b));select=(rest>6527.8)&(rest<6597.8);continuum=((rest>6500)&(rest<6537))|((rest>6591)&(rest<6620))
    w=rest[select];ratio=(w/6562.8)**2;v=(ratio-1)*299700/(1+ratio);uniform=np.linspace(v[0],v[-1],len(v));lines=[]
    for row in spectra:
        q=np.polyfit(rest[continuum],row[continuum],3);line=np.interp(uniform,v,row[select]-np.polyval(q,w));lines.append([float(f'{a:.5f}') for a in line])
    observed=np.array(lines).T.ravel();velocity=np.array([float(f'{a*1e5:.5e}') for a in uniform]);phase=np.array([float(f'{float(np.float32(a)):.6f}') for a in ph[0]])
    return observed,velocity,phase

def forward_range_rms(image,raw_path):
    data,velocity,cycles=public_observation_model(raw_path);nph=28;nvp=len(velocity);nv=144
    pi2=float(8*np.arctan(np.float32(1.)));phase=cycles*pi2;width=.03930*pi2;va=max(abs(velocity[0]),velocity[-1]);dv=2*va/143
    axis=np.linspace(-va,va,nv);vx,vy=np.meshgrid(axis,axis,indexing='ij');support=vx*vx+vy*vy<(va+dv/10)**2;ix,iy=np.where(support);vx=vx[support];vy=vy[support]
    edges=np.r_[velocity[0]-(velocity[1]-velocity[0])/2,(velocity[:-1]+velocity[1:])/2,velocity[-1]+(velocity[-1]-velocity[-2])/2]
    rs=[];cs=[];ws=[]
    def F(t):return np.where(t<0,0.,np.where(t<.5,2*t*t,np.where(t<1,1-2*(1-t)**2,1.)))
    for i,p in enumerate(phase):
        a=vx*np.sin(p-width/2)-vy*np.cos(p-width/2);b=vx*np.sin(p+width/2)-vy*np.cos(p+width/2);lo=np.minimum(a,b)-dv/2;hi=np.maximum(a,b)+dv/2
        begin=np.clip(np.searchsorted(edges,lo,'left')-1,0,nvp-1);end=np.clip(np.searchsorted(edges,hi,'left')-1,0,nvp-1)
        for j in range(int(np.max(end-begin))+1):
            c=np.flatnonzero(begin+j<=end);k=begin[c]+j;weight=(F((edges[k+1]-lo[c])/(hi[c]-lo[c]))-F((edges[k]-lo[c])/(hi[c]-lo[c]))).astype(np.float32)
            rs.append((i+k*nph).astype(np.int32));cs.append(c.astype(np.int32));ws.append(weight)
    rows=np.concatenate(rs);cols=np.concatenate(cs);weights=np.concatenate(ws);totals=np.bincount(cols,weights=weights.astype(np.float64),minlength=len(vx));weights=(weights.astype(np.float64)/totals[cols]).astype(np.float32).astype(np.float64)
    flat=(vx*vx+vy*vy<va*va).astype(float)
    response=np.bincount(rows,weights=weights*flat[cols],minlength=nph*nvp)
    baseline=float(np.float32(.1))*sum(data)/len(data);center=response[(nvp//2)*nph-1];background=flat*baseline/center;background_data=baseline*response/center
    data+=background_data;data[data<baseline/3]=float(np.float32(1e-5))*baseline
    folded=data.reshape(nvp,nph);blank=(cycles>.95)|(cycles<.05);flux=folded.sum(axis=0);flux[blank]=0;average=sum(flux)/np.count_nonzero(~blank);relative=flux/average
    factors=np.ones(nph);valid=relative!=0;factors[valid]=1/relative[valid];folded*=factors
    denu=(velocity[-1]-velocity[0])/(nvp-1)/6562.8*1e8
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        field=image[0,iy,ix]*dv**2/denu+background
        predicted=np.bincount(rows,weights=weights*field[cols],minlength=nph*nvp)
        result=float(np.sqrt(np.mean(((predicted-data)/np.ptp(data))**2)))
    if not np.isfinite(result):raise ValueError('Nonfinite observation-consistency metric')
    return result

def evaluate(prediction,evaluation_dir=None):
    root=Path(evaluation_dir) if evaluation_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        c=config(root/'metrics.json');reference=root/'ground_truth.npz'
        if not reference.exists():reference=root/'baseline_reference.npz'
        y=load_field(reference);x=load_field(prediction);cosine,error=reference_metrics(x,y);forward=forward_range_rms(x,root.parent/'data/raw_data.npz')
        measured=dict(ncc=cosine,nrmse=error,forward_range_rms=forward);thresholds={name:dict(comparison='>=' if name=='ncc' else '<=',boundary=c[name+'_boundary']) for name in measured}
        passed=cosine>=c['ncc_boundary'] and error<=c['nrmse_boundary'] and forward<=c['forward_range_rms_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
    except Exception as e:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(e)])
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failure: '+str(e),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())

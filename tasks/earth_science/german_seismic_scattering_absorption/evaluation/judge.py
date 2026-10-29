"""Independent standalone evaluator; Qopen MIT conventions, see LICENSE."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy import signal,fftpack
SAMPLE_RATE=20.;DENSITY=2700.;FREE_SURFACE=4.;VELOCITY=3400.
def load_data(path="data/raw_data.npz"):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
def smooth(values,samples=20,pad=True):
    w=np.bartlett(samples);w/=w.sum()
    if pad:values=np.r_[np.zeros((samples-1)//2),values,np.zeros(samples//2)]
    return np.convolve(w,values,'valid')
def filter_width(band):
    b,a=signal.iirfilter(2,np.array(band)/(SAMPLE_RATE*.5),btype='band',ftype='butter',output='ba')
    w,h=signal.freqz(b,a)
    return (w[1]-w[0])/2/np.pi*SAMPLE_RATE*np.sum(np.abs(h)**4)
def observed_energy(velocity,band):
    sos=signal.iirfilter(2,np.array(band)/(SAMPLE_RATE*.5),btype='band',ftype='butter',output='sos')
    energies=[]
    for row in velocity:
        detrended=signal.detrend(row,type='linear')
        filtered=signal.sosfilt(sos,signal.sosfilt(sos,detrended)[::-1])[::-1]
        hilbert=fftpack.hilbert(filtered)
        energies.append(DENSITY*(filtered**2+hilbert**2)/2/filter_width(band)/FREE_SURFACE)
    return np.sum(energies,axis=0)
def sample_index(time,start):
    value=(time-start)*SAMPLE_RATE
    return int(np.floor(value+.5))
def prepare_pair(velocity,band,distance,start,sonset):
    energy=observed_energy(velocity,band)
    noise=min(np.mean(energy[sample_index(a,start):sample_index(b,start)+1]) for a,b in [(-10,-5),(-5,0)])
    corrected=energy-noise;corrected[corrected<noise/100]=noise/100
    first=sample_index(sonset-2,start);last=sample_index(sonset+20,start)
    bulk=corrected[first:last+1];tbgrid=np.arange(len(bulk))/SAMPLE_RATE+distance/VELOCITY-2
    eb=np.mean(bulk);tb=np.sum(bulk*tbgrid)/np.sum(bulk)
    smoothed=smooth(corrected);cs=sample_index(sonset+20,start)
    crossing=np.flatnonzero(smoothed[cs:]<3*noise)
    offset=int(crossing[0]) if len(crossing) else len(smoothed)-cs
    end=min(sonset+150,sonset+20+offset/SAMPLE_RATE)
    if end-(sonset+20)<.05:return None
    ce=sample_index(end,start);ec=smoothed[cs:ce+1]
    tc=np.arange(len(ec))/SAMPLE_RATE+distance/VELOCITY+20
    return dict(energy=smoothed,unsmoothed=corrected,tcoda=tc,Ecoda=ec,tbulk=tb,Ebulk=eb,bulk_interval=(tbgrid[0],tbgrid[-1]),bulk_weight=len(bulk),distance=distance,noise=noise)
def prepare_band(data,event,band_index):
    pairs=[]
    for j in np.flatnonzero(data['valid'][event]):
        p=prepare_pair(data['velocity'][event,j],data['frequency_bands'][band_index],data['distance'][event,j],data['start_time'][event,j],data['s_onset'][event,j])
        if p is not None:p['station_slot']=j;pairs.append(p)
    return pairs
def coda_green(distance,time,scattering):
    r=distance*scattering;t=time*VELOCITY*scattering
    a=1-r**2/t**2
    return a**.125/(4*np.pi*t/3)**1.5*np.exp(t*(a**.75-1))*np.sqrt(1+2.026/(t*a**.75))*scattering**3
def green(distance,time,scattering):
    time=np.asarray(time);g=np.zeros(len(time));dt=float(time[1]-time[0]);i=np.count_nonzero(VELOCITY*time-distance<0)
    g[i+1:]=coda_green(distance,time[i+1:],scattering)
    if 0<i<len(g):
        arrival=distance/VELOCITY
        g[i]=np.exp(-VELOCITY*arrival*scattering)/(4*np.pi*VELOCITY**2*arrival**2)/dt/VELOCITY
    return g
def smoothed_green(distance,time,scattering):
    dt=time[1]-time[0];samples=int(round(1./dt));left=(samples-1)//2;right=samples//2
    extended=np.r_[time[0]-left*dt+np.arange(left)*dt,time,time[-1]+dt+np.arange(right)*dt]
    return smooth(green(distance,extended,scattering),samples,pad=False)
def green_observations(pairs,scattering):
    coda=[smoothed_green(p['distance'],p['tcoda'],scattering) for p in pairs]
    bulk=[np.mean(green(p['distance'],np.linspace(*p['bulk_interval'],1000),scattering)) for p in pairs]
    return coda,bulk

SHAPES={'reconstruction':(1,5,5,2),'source_energy':(1,5,5),'site_amplification':(1,5,5,5)}
def read_output(path,mask):
    with np.load(path,allow_pickle=False) as archive:
        out={}
        for key,shape in SHAPES.items():
            if key not in archive:raise ValueError('Missing required key: '+key)
            a=archive[key]
            if a.shape!=shape:raise ValueError('Wrong shape: '+key)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64: '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite values: '+key)
            out[key]=a
    for key in ('reconstruction','source_energy'):
        if np.any(out[key]<=0):raise ValueError('Nonpositive physical quantity: '+key)
    sites=out['site_amplification'][0]
    if np.any(sites[mask]<=0) or np.any(sites[~mask]!=0):raise ValueError('Site positivity/padding violation')
    if np.max(abs(np.sum(np.where(mask,np.log(np.where(mask,sites,1)),0),axis=-1)/mask.sum(axis=-1)))>1e-5:
        raise ValueError('Site geometric-mean gauge violation')
    return out

def forward_errors(out,data):
    values=[]
    for e in range(5):
        row=[]
        for f in range(5):
            pairs=prepare_band(data,e,f);g,b=out['reconstruction'][0,e,f]
            gc,gb=green_observations(pairs,g);residual=[];weights=[]
            for j,p in enumerate(pairs):
                amplitude=out['source_energy'][0,e,f]*out['site_amplification'][0,e,f,p['station_slot']]
                residual.extend(np.log(p['Ecoda'])-np.log(amplitude*gc[j])+b*p['tcoda'])
                weights.extend(np.ones(len(p['tcoda'])))
            for j,p in enumerate(pairs):
                amplitude=out['source_energy'][0,e,f]*out['site_amplification'][0,e,f,p['station_slot']]
                residual.append(np.log(p['Ebulk'])-np.log(amplitude*gb[j])+b*p['tbulk']);weights.append(p['bulk_weight'])
            residual=np.asarray(residual)
            value=np.sqrt(np.dot(weights,residual**2)/(len(residual)-len(pairs)-1))
            if not np.isfinite(value):raise ValueError('Undefined forward residual')
            row.append(value)
        values.append(row)
    return np.array(values)

def evaluate(prediction,evaluation_dir=None):
    folder=Path(evaluation_dir) if evaluation_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        data=load_data(folder.parent/'data/raw_data.npz');mask=data['fit_valid']
        reference=folder/'ground_truth.npz'
        if not reference.exists():reference=folder/'baseline_reference.npz'
        try:ref=read_output(reference,mask)
        except Exception as exc:raise ValueError('Invalid hidden reference: '+str(exc)) from exc
        pred=read_output(prediction,mask)
        for key in SHAPES:
            a=pred[key];b=ref[key]
            if key=='site_amplification':a=a[0][mask];b=b[0][mask]
            metric=key+'_max_abs_log_ratio';report['metrics'][metric]=float(np.max(abs(np.log(a/b))))
            bound=config[metric+'_boundary']
            if not np.isfinite(bound) or bound<=0:raise ValueError('Invalid metric boundary')
            report['thresholds'][metric]={'comparison':'<=','boundary':bound}
        error=forward_errors(pred,data);bounds=np.array(config['forward_log_error_boundary'])
        if bounds.shape!=(5,5) or not np.isfinite(bounds).all() or np.any(bounds<=0):raise ValueError('Invalid forward boundaries')
        report['metrics']['forward_error_boundary_ratio']=float(np.max(error/bounds))
        report['thresholds']['forward_error_boundary_ratio']={'comparison':'<=','boundary':1.0}
        report['passed']=all(report['metrics'][k]<=v['boundary'] for k,v in report['thresholds'].items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)])
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())

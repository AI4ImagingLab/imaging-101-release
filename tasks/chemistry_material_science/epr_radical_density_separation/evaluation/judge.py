"""Independent strict evaluator; never imports solver or submitted code."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np

SHAPE=(1,2,100,100)
ASSET_DIR=Path(__file__).resolve().parent

def _number(value,label):
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError(f'{label} must be a finite JSON number')
 return float(value)

def _load_prediction(path):
 path=Path(path)
 if not path.is_file():raise ValueError('Prediction file is missing')
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!={'reconstruction'}:raise ValueError('Prediction must contain exactly reconstruction')
  x=z['reconstruction']
  if x.shape!=SHAPE:raise ValueError(f'Prediction shape must be {SHAPE}')
  if x.dtype!=np.dtype('float64'):raise ValueError('Prediction dtype must be real float64')
  if not np.isfinite(x).all():raise ValueError('Prediction contains NaN or Infinity')
 return x

def predict_observations(images,field,gradients,spectra,delta=.03):
 """Direct, per-projection Fourier quadrature, implemented independently.

    The measurement is sum integral u(x)h(B+g.x)dx. A positive shift
    in h(B+s) multiplies its Fourier transform by exp(+i omega*s).
    Frequencies beyond the spatial Nyquist bound are exactly excluded.
 """
 nb=len(field);alpha=np.arange(nb//2+1);omega=2*np.pi*alpha/(nb*(field[1]-field[0]));hf=np.fft.rfft(spectra,axis=-1)
 x=(np.arange(images.shape[2])-images.shape[2]//2)*delta;y=(np.arange(images.shape[1])-images.shape[1]//2)*delta
 result=np.zeros((gradients.shape[1],nb))
 for p in range(gradients.shape[1]):
  gx,gy=gradients[:,p];keep=(np.hypot(gx,gy)*alpha<.5*nb*(field[1]-field[0])/delta)&(alpha<.5*nb)
  frequencies=np.flatnonzero(keep);wx=omega[keep]*gx;wy=omega[keep]*gy
  ex=np.exp(1j*wx[:,None]*x[None,:]);ey=np.exp(1j*wy[:,None]*y[None,:]);coeff=np.zeros(nb//2+1,dtype=complex)
  for j,image in enumerate(images):coeff[frequencies]+=delta**2*hf[j,keep]*np.einsum('fy,yx,fx->f',ey,image,ex,optimize=True)
  result[p]=np.fft.irfft(coeff,n=nb)
 return result

def evaluate(prediction,asset_dir=None):
 """Return report and exit code; reference directory defaults to this file."""
 assets=Path(asset_dir) if asset_dir is not None else ASSET_DIR
 report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
 try:
  x=_load_prediction(prediction)
  config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
  ncc=_number(config['ncc_boundary'],'ncc_boundary');mass=_number(config['mass_relative_error_boundary'],'mass_relative_error_boundary')
  if not 0<ncc<=1 or mass<=0:raise ValueError('NCC and mass boundaries are out of range')
  nrmse=config['nrmse_boundary_per_species']
  if not isinstance(nrmse,list) or len(nrmse)!=2:raise ValueError('Exactly two NRMSE boundaries are required')
  nrmse=[_number(v,'nrmse_boundary_per_species') for v in nrmse]
  if any(v<=0 for v in nrmse):raise ValueError('NRMSE boundaries must be positive')
  lo=_number(config['forward_noise_rms_lower_boundary'],'forward_noise_rms_lower_boundary');hi=_number(config['forward_noise_rms_upper_boundary'],'forward_noise_rms_upper_boundary');corr=_number(config['absolute_residual_lag1_boundary'],'absolute_residual_lag1_boundary')
  if not 0<=lo<hi or not 0<corr<1:raise ValueError('Invalid forward/noise boundaries')
  ref=assets/'ground_truth.npz'
  if not ref.is_file():ref=assets/'baseline_reference.npz'
  if not ref.is_file():raise ValueError('Hidden reference asset is unavailable')
  with np.load(ref,allow_pickle=False) as z:
   if 'reconstruction' not in z.files:raise ValueError('Hidden reference reconstruction is missing')
   truth=z['reconstruction']
   if truth.shape!=SHAPE or truth.dtype!=np.float64 or not np.isfinite(truth).all():raise ValueError('Hidden reference shape, dtype or finite values invalid')
  values={};thresholds={}
  for j,label in enumerate(('TAM','TEMPO')):
   estimate=x[0,j];target=truth[0,j];norm=float(np.linalg.norm(target));dynamic=float(np.ptp(target));total=float(target.sum())
   if norm<=0 or dynamic<=0 or total<=0:raise ValueError('Reference norm, dynamic range or integrated amount is undefined')
   xn=float(np.linalg.norm(estimate));values[f'{label}_ncc']=float(np.vdot(estimate,target)/(xn*norm)) if xn>0 else 0.
   values[f'{label}_nrmse']=float(np.sqrt(np.mean((estimate-target)**2))/dynamic)
   values[f'{label}_mass_relative_error']=float(abs(estimate.sum()-total)/abs(total))
   thresholds[f'{label}_ncc']={'comparison':'>=','boundary':ncc};thresholds[f'{label}_nrmse']={'comparison':'<=','boundary':nrmse[j]};thresholds[f'{label}_mass_relative_error']={'comparison':'<=','boundary':mass}
  with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as d:
   field=d['magnetic_field_G'];g=d['gradients_G_cm'];h=d['reference_spectra'];observed=d['projections']
   if field.shape!=(1,1200) or g.shape!=(1,2,40) or h.shape!=(1,2,1200) or observed.shape!=(1,40,1200):raise ValueError('Public forward calibration shape is invalid')
   if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (field,g,h,observed)):raise ValueError('Public forward calibration dtype or finite values invalid')
   residual=predict_observations(x[0],field[0],g[0],h[0])-observed[0]
  values['forward_noise_rms_over_sigma']=float(np.sqrt(np.mean(residual**2))/5000.)
  a=residual[:,:-1].ravel();b=residual[:,1:].ravel();a=a-a.mean();b=b-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b))
  if den<=0:raise ValueError('Residual lag-one correlation is undefined')
  values['absolute_residual_lag1']=float(abs(np.vdot(a,b)/den))
  # Two comparisons for the same measured RMS are explicitly both required.
  values['forward_noise_rms_minimum_check']=values['forward_noise_rms_over_sigma'];values['forward_noise_rms_maximum_check']=values['forward_noise_rms_over_sigma']
  thresholds['forward_noise_rms_minimum_check']={'comparison':'>=','boundary':lo};thresholds['forward_noise_rms_maximum_check']={'comparison':'<=','boundary':hi};thresholds['absolute_residual_lag1']={'comparison':'<=','boundary':corr}
  if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite metric encountered')
  passed=all(values[k]>=rule['boundary'] if rule['comparison']=='>=' else values[k]<=rule['boundary'] for k,rule in thresholds.items())
  report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds=thresholds)
  return report,0 if passed else 1
 except Exception as error:
  report['errors']=[f'{type(error).__name__}: {error}'];return report,2

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True,type=Path);parser.add_argument('--report',required=True,type=Path);args=parser.parse_args();report,code=evaluate(args.prediction)
 try:
  args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 except Exception as error:print(f'Report write failed: {error}',file=sys.stderr);return 2
 return code

if __name__=='__main__':raise SystemExit(main())

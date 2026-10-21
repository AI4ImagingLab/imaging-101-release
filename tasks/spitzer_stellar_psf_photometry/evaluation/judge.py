"""Standalone deterministic parameter and public-observation evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline

ASSETS=Path(__file__).resolve().parent
SHAPES={'positions':(1,4,2),'flux':(1,4),'background':(1,4)}

def load_arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in SHAPES}
    for key,value in arrays.items():
        if value.shape!=SHAPES[key] or value.dtype!=np.dtype('float64'):
            raise ValueError(f'{key}: expected shape {SHAPES[key]} and float64')
        if not np.isfinite(value).all():
            raise ValueError(f'{key}: nonfinite values')
    if np.any(arrays['flux']<0) or np.any(arrays['positions']<0) or np.any(arrays['positions']>16):
        raise ValueError('Unphysical flux or out-of-patch position')
    return arrays

def observation_error(arrays, raw_path):
    with np.load(raw_path,allow_pickle=False) as data:
        images=data['images'][0]
        flight=data['flight_psf'][0]
    grid=np.arange(81.)
    coordinates=40+(grid-40)*.3/.30325
    effective=RectBivariateSpline(grid,grid,flight.T)(coordinates,coordinates).T
    effective *=16/effective.sum()
    spline=RectBivariateSpline(grid,grid,effective.T)
    y,x=np.indices((17,17),dtype=float)
    models=[]
    for i in range(4):
        px,py=arrays['positions'][0,i]
        a=4*(x-px)+40;b=4*(y-py)+40
        q=spline(a,b,grid=False)
        q[(a<0)|(a>80)|(b<0)|(b>80)]=0
        models.append(arrays['flux'][0,i]*q+arrays['background'][0,i])
    return float(np.linalg.norm(np.asarray(models)-images)/np.linalg.norm(images))

def evaluate(prediction,assets=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        names=('position_error','flux_relative_error','background_absolute_error','observation_relative_l2')
        thresholds={name:{'comparison':'<=','boundary':config[name+'_boundary']} for name in names}
        if any(not isinstance(v['boundary'],(int,float)) or not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()):
            raise ValueError('Invalid metric boundary')
        report['thresholds']=thresholds
        truth=assets/'ground_truth.npz'
        reference=load_arrays(truth if truth.exists() else assets/'baseline_reference.npz')
        estimate=load_arrays(prediction)
        if np.any(reference['flux']<=0):
            raise ValueError('Invalid zero reference flux')
        metrics={
          'position_error':float(np.max(np.linalg.norm(estimate['positions']-reference['positions'],axis=-1))),
          'flux_relative_error':float(np.max(np.abs(estimate['flux']-reference['flux'])/reference['flux'])),
          'background_absolute_error':float(np.max(np.abs(estimate['background']-reference['background']))),
          'observation_relative_l2':observation_error(estimate,assets.parent/'data/raw_data.npz')}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError('Undefined metric')
        report['metrics']=metrics
        report['passed']=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as error:
        report['errors']=[f'{type(error).__name__}: {error}']
        report['metrics']={}
        return report,2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report,code=evaluate(Path(args.prediction))
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as error:
        print(f'Report write failed: {error}',file=sys.stderr)
        return 2
    return code

if __name__=='__main__':
    sys.exit(main())

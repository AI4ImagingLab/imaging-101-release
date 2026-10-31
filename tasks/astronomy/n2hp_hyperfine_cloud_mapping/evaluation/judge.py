"""Standalone evaluator: no reference solver imports or submission execution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
KEYS = ('excitation_temperature','optical_depth','velocity','dispersion')
HERE = Path(__file__).resolve().parent

def load_maps(path):
    if not Path(path).is_file():
        raise ValueError('Missing file: '+str(path))
    with np.load(path,allow_pickle=False) as data:
        result={}
        for key in KEYS:
            if key not in data:
                raise ValueError('Missing required key: '+key)
            value=data[key]
            if value.shape != (1,3,3):
                raise ValueError('Invalid shape for '+key)
            if value.dtype not in (np.dtype('float32'),np.dtype('float64')):
                raise ValueError('Expected real float32/float64 dtype for '+key)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite values for '+key)
            result[key]=value
    return result

def relative_max(estimate,reference):
    if np.any(reference == 0):
        raise ValueError('Zero reference invalid for relative map error')
    return float(np.max(np.abs((estimate-reference)/reference)))

def reduced_chi_square(maps,data_path):
    with np.load(data_path,allow_pickle=False) as data:
        frequency=data['frequency'][0]; observed=data['brightness'][0]
    if frequency.shape != (501,) or observed.shape != (501,3,3):
        raise ValueError('Invalid public calibration dimensions')
    if not np.isfinite(frequency).all() or not np.isfinite(observed).all():
        raise ValueError('Nonfinite public calibration')
    c=299792.458
    offsets=np.array([-7.993]*3+[-.6112]*3+[0,.9533,.9533]+[5.5371]*3+[5.9704]*2+[6.9238])
    weights=np.array([.025957,.065372,.019779,.004376,.034890,.071844,.259259,.156480,.028705,.041361,.013309,.056442,.156482,.028705,.037038])
    centers=93173763700.*(1-offsets/c)
    radio_velocity=c*(1-frequency/93176265000.)
    noise=observed[(radio_velocity>=20)&(radio_velocity<=28)].std(axis=0)
    if not np.isfinite(noise).all() or np.any(noise<=0):
        raise ValueError('Invalid noise calibration')
    largest=0.
    for y,x in np.ndindex(3,3):
        tex,tau,velocity,width=[float(maps[key][0,y,x]) for key in KEYS]
        if tex <= 0 or tau < 0 or width <= 0:
            raise ValueError('Nonphysical parameter domain')
        optical=np.zeros(501)
        for offset,center,strength in zip(offsets,centers,weights):
            line=center*(1-offset/c)
            optical += tau*strength/sum(weights)*np.exp(-.5*((frequency+velocity*line/c-center)/(width*line/c))**2)
        energy=(6.62607015e-34/1.380649e-23)*frequency
        with np.errstate(over='ignore',under='ignore'):
            model=(1-np.exp(-optical))*energy*(1/np.expm1(energy/tex)-1/np.expm1(energy/2.73))
        score=float(np.sum(((model-observed[:,y,x])/noise[y,x])**2)/497)
        if not np.isfinite(score):
            raise ValueError('Nonfinite observation metric')
        largest=max(largest,score)
    return largest

def evaluate(prediction, asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        asset_dir=Path(asset_dir)
        config=json.loads((asset_dir/'metrics.json').read_text(encoding='utf8'))
        bound=float(config['parameter_relative_max_boundary']); chi_bound=float(config['reduced_chi_square_boundary'])
        if not np.isfinite([bound,chi_bound]).all() or bound<=0 or chi_bound<=0:
            raise ValueError('Invalid metric configuration')
        ref_path=asset_dir/'ground_truth.npz'
        if not ref_path.exists():ref_path=asset_dir/'baseline_reference.npz'
        reference=load_maps(ref_path)
        for key in KEYS:
            if np.any(reference[key]==0):raise ValueError('Zero reference map')
        estimate=load_maps(prediction)
        # Validate the physical domain before emitting any scores.
        if np.any(estimate[KEYS[0]]<=0) or np.any(estimate[KEYS[1]]<0) or np.any(estimate[KEYS[3]]<=0):
            raise ValueError('Nonphysical parameter domain')
        chi=reduced_chi_square(estimate,asset_dir.parent/'data/raw_data.npz')
        measured={key+'_relative_max':relative_max(estimate[key],reference[key]) for key in KEYS}
        measured['maximum_reduced_chi_square']=chi
        thresholds={key+'_relative_max':{'comparison':'<=','boundary':bound} for key in KEYS}
        thresholds['maximum_reduced_chi_square']={'comparison':'<=','boundary':chi_bound}
        passed=all(measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as error:
        report['errors']=[str(error)]
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:
        print('Could not write report: '+str(error),file=sys.stderr)
        return 2
    return code

if __name__=='__main__':
    raise SystemExit(main())

"""Standalone submission and observation-consistency evaluator; no solver imports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy import fft

HERE = Path(__file__).resolve().parent
KEYS = {'reconstruction':np.dtype('float32'),'phase':np.dtype('complex64'),'mtf':np.dtype('float32')}
SHAPE = (1,400,400)


def aligned_metrics(image, reference):
    """Fixed circular translation; no amplitude fitting or intensity rescaling."""
    correlation = fft.ifft2(fft.fft2(reference)*fft.fft2(image).conj()).real
    index = np.unravel_index(correlation.argmax(),correlation.shape)
    estimate = np.roll(image,index,axis=(0,1))
    estimate = estimate.astype(np.float64)
    reference = reference.astype(np.float64)
    norm = float(np.linalg.norm(estimate)*np.linalg.norm(reference))
    ncc = float(np.clip(np.sum(estimate*reference)/norm,-1,1)) if norm > 0 else 0.0
    span = float(np.ptp(reference))
    if span <= 0:
        raise ValueError('Evaluator reference has zero dynamic range')
    return ncc,float(np.sqrt(np.mean((estimate.astype(np.float64)-reference)**2))/span)


def observation_metrics(prediction, raw_path):
    """Independent SciPy implementation of covariance and paper output checks."""
    with np.load(raw_path,allow_pickle=False) as source:
        x = source['camera_frames'][0].astype(np.float64)
    wy = np.hanning(400)
    x = x*wy[None,:,None]*wy[None,None,:]
    x -= x.mean(0)
    spectra = fft.fftshift(fft.fft2(x))
    expected_mtf = np.sqrt(np.mean(np.abs(spectra)**2,axis=0))
    expected_mtf /= expected_mtf.max()
    phase = prediction['phase'][0]
    mtf = prediction['mtf'][0]
    corrected = spectra*phase[None]
    frames = fft.ifft2(fft.ifftshift(corrected/(mtf[None]+0.01),axes=(-2,-1)))
    image = np.sqrt(np.mean(np.abs(frames)**2,axis=0))
    image /= image.max()
    consistency = float(np.linalg.norm(prediction['reconstruction'][0]-image)/np.linalg.norm(image))
    mtf_error = float(np.linalg.norm(mtf-expected_mtf)/np.linalg.norm(expected_mtf))
    matrix = corrected.transpose(2,1,0).reshape(160000,150)
    matrix /= np.linalg.norm(matrix)
    diagonals = fft.ifft(np.sum(np.abs(fft.fft(matrix,axis=0))**2,axis=1))
    coherence = float(np.sum(np.abs(diagonals)**2))
    return {'image_consistency':consistency,'mtf_consistency':mtf_error,
            'covariance_coherence':coherence}


def evaluate(prediction_path, evaluation_dir=HERE, raw_path=None):
    """Return a finite JSON-compatible report; malformed inputs have no metrics."""
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        evaluation_dir=Path(evaluation_dir)
        with (evaluation_dir/'metrics.json').open(encoding='utf-8') as handle:
            config=json.load(handle)
        boundaries=config['thresholds']
        report['thresholds']=boundaries
        for key,item in boundaries.items():
            if item['comparison'] not in ['>=','<='] or not np.isfinite(item['boundary']):
                raise ValueError('Invalid evaluator metric definition: '+key)
        path=Path(prediction_path)
        if not path.is_file():
            raise ValueError('Prediction file does not exist: '+str(path))
        with np.load(path,allow_pickle=False) as archive:
            prediction={}
            for key,dtype in KEYS.items():
                if key not in archive:
                    raise ValueError('Missing required output key: '+key)
                value=archive[key]
                if value.shape != SHAPE or value.dtype != dtype:
                    raise ValueError('Invalid shape/dtype for '+key)
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite values in '+key)
                prediction[key]=value
        if (prediction['reconstruction']<0).any() or (prediction['mtf']<0).any():
            raise ValueError('Image and MTF must be nonnegative')
        if np.max(np.abs(np.abs(prediction['phase'])-1))>1e-4:
            raise ValueError('phase must be a unit-magnitude correction phasor')
        with np.load(evaluation_dir/'baseline_reference.npz',allow_pickle=False) as archive:
            reference=archive['reconstruction']
        if reference.shape != SHAPE or reference.dtype != np.float32 or not np.isfinite(reference).all():
            raise ValueError('Invalid hidden evaluator reference')
        ncc,nrmse=aligned_metrics(prediction['reconstruction'][0],reference[0])
        if raw_path is None:
            raw_path=evaluation_dir.parent/'data/raw_data.npz'
        values={'ncc_vs_ref':ncc,'nrmse_vs_ref':nrmse}
        values.update(observation_metrics(prediction,raw_path))
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite evaluation metric')
        report['metrics']=values
        passed=all(values[key]>=item['boundary'] if item['comparison']=='>='
                   else values[key]<=item['boundary'] for key,item in boundaries.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed))
    except Exception as exc:
        report['metrics']={}
        report['errors']=[str(exc)]
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        output=Path(args.report)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print('Cannot write Judge report: '+str(exc),file=sys.stderr)
        return 2
    return 0 if report['passed'] else (1 if report['status']=='fail' else 2)


if __name__=='__main__':
    sys.exit(main())

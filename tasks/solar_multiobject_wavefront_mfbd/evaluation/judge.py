"""Standalone deterministic evaluator; never imports reconstruction code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPES = {'reconstruction': (1,2,476,476), 'objects': (1,2,121,88,88),
          'wavefront_coefficients': (1,121,12,54)}


def load_arrays(path):
    if not Path(path).is_file():
        raise ValueError(f'Missing NPZ file: {path}')
    with np.load(path, allow_pickle=False) as data:
        output = {}
        for key, shape in SHAPES.items():
            if key not in data:
                raise ValueError(f'Missing required key: {key}')
            value = data[key]
            if value.shape != shape:
                raise ValueError(f'{key}: expected shape {shape}, got {value.shape}')
            if value.dtype != np.float32:
                raise ValueError(f'{key}: expected float32, got {value.dtype}')
            if not np.isfinite(value).all():
                raise ValueError(f'{key}: nonfinite values')
            output[key] = value
    return output


def intensity_metrics(estimate, reference):
    estimate, reference = estimate.astype('float64').ravel(), reference.astype('float64').ravel()
    scale = np.ptp(reference)
    norm = np.linalg.norm(estimate)*np.linalg.norm(reference)
    if scale <= 0 or np.linalg.norm(reference) <= 0:
        raise ValueError('Degenerate hidden intensity reference')
    ncc = float(np.dot(estimate,reference)/norm) if norm > 0 else 0.0
    return ncc, float(np.sqrt(np.mean((estimate-reference)**2))/scale)


def relative_error(estimate, reference):
    numerator=np.sum((estimate.astype('float64')-reference.astype('float64'))**2)
    denominator=np.sum(reference.astype('float64')**2)
    if denominator <= 0:
        raise ValueError('Zero-norm comparison reference')
    return float(np.sqrt(numerator/denominator))


def merge_objects(objects):
    edge=np.hanning(60)
    line=np.ones(76)
    line[:30],line[-30:]=edge[:30],edge[-30:]
    weight=np.outer(line,line).astype('float32')
    merged=np.zeros((2,476,476),dtype='float32')
    normalization=np.zeros_like(merged)
    for index in range(121):
        row,column=divmod(index,11)
        region=(slice(None),slice(40*row,40*row+76),slice(40*column,40*column+76))
        merged[region]+=objects[:,index,6:82,6:82]*weight
        normalization[region]+=weight
    return merged/(normalization+1e-10)


def psf_and_observation_metrics(coefficients, objects, reference_psfs, pupil, basis, data_path):
    with np.load(data_path,allow_pickle=False) as handle:
        observed=handle['observations'][0].copy()
    observed/=np.mean(observed,axis=(-1,-2),keepdims=True)
    line=np.ones(88)
    edge=np.hanning(12)
    line[:6],line[-6:]=edge[:6],edge[-6:]
    window=np.outer(line,line).astype('float32')
    psf_squared_error=0.0
    psf_reference_power=0.0
    residual_power=0.0
    observation_power=0.0
    for index in range(121):
        phase=np.einsum('jm,mrc->jrc',coefficients[index],basis,optimize=False)
        amplitude=np.fft.fft2(pupil[None]*np.exp(1j*phase),norm='ortho')
        power=(amplitude.conj()*amplitude).real
        psf=power/np.sum(power,axis=(-1,-2),keepdims=True)
        ref=reference_psfs[index]
        psf_squared_error+=np.sum((psf.astype('float64')-ref)**2)
        psf_reference_power+=np.sum(ref.astype('float64')**2)
        transfer=np.fft.fft2(psf)
        row,column=divmod(index,11)
        for channel in range(2):
            frames=observed[channel,:,40*row:40*row+88,40*column:40*column+88]
            mean=np.mean(frames,axis=(-1,-2),keepdims=True)
            target=(frames-mean)*window+mean
            degraded=np.fft.ifft2(np.fft.fft2(objects[channel,index])[None]*transfer).real
            residual_power+=np.sum((degraded.astype('float64')-target)**2)
            observation_power+=np.sum(target.astype('float64')**2)
    if psf_reference_power<=0 or observation_power<=0:
        raise ValueError('Degenerate PSF or observation reference')
    return float(np.sqrt(psf_squared_error/psf_reference_power)),float(np.sqrt(residual_power/observation_power))


def evaluate(prediction, directory=HERE):
    directory=Path(directory)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        metrics_path=directory/'metrics.json'
        reference_path=directory/'baseline_reference.npz'
        for path in (metrics_path,reference_path):
            if not path.is_file():
                raise ValueError(f'Unavailable evaluator asset: {path}')
        config=json.loads(metrics_path.read_text(encoding='utf-8'))
        thresholds=config['thresholds']
        required={'reconstruction_ncc_min','reconstruction_nrmse_max','objects_ncc_min',
                  'objects_nrmse_max','wavefront_psf_relative_error','observation_relative_error',
                  'mosaic_relative_error','translation_gauge_max','null_slot_max'}
        if set(thresholds)!=required:
            raise ValueError('Undefined or missing metric configuration')
        for key,value in thresholds.items():
            if value.get('comparison') not in ('>=','<=') or not np.isfinite(value.get('boundary',np.nan)):
                raise ValueError(f'Invalid threshold: {key}')
        reference=load_arrays(reference_path)
        with np.load(reference_path,allow_pickle=False) as hidden:
            psfs=hidden['reference_psfs'][0]
            pupil=hidden['pupil'][0]
            basis=hidden['phase_basis'][0]
        if psfs.shape!=(121,12,88,88) or pupil.shape!=(88,88) or basis.shape!=(54,88,88):
            raise ValueError('Malformed hidden optics/reference assets')
        if not all(np.isfinite(x).all() for x in (psfs,pupil,basis)):
            raise ValueError('Nonfinite hidden optics/reference assets')
        submitted=load_arrays(prediction)
        metrics={}
        for quantity in ('reconstruction','objects'):
            pairs=[intensity_metrics(submitted[quantity][0,k],reference[quantity][0,k]) for k in range(2)]
            metrics[quantity+'_ncc_min']=min(x[0] for x in pairs)
            metrics[quantity+'_nrmse_max']=max(x[1] for x in pairs)
        modes=submitted['wavefront_coefficients'][0]
        objects=submitted['objects'][0]
        metrics['wavefront_psf_relative_error'],metrics['observation_relative_error']=psf_and_observation_metrics(
            modes,objects,psfs,pupil,basis,directory.parent/'data/raw_data.npz')
        merged=merge_objects(objects)
        if np.linalg.norm(merged.astype('float64')) == 0:
            metrics['mosaic_relative_error']=float(np.sqrt(np.mean(submitted['reconstruction'][0].astype('float64')**2)))
        else:
            metrics['mosaic_relative_error']=relative_error(submitted['reconstruction'][0],merged)
        metrics['translation_gauge_max']=float(np.max(np.abs(modes[:,0,:2])))
        metrics['null_slot_max']=float(np.max(np.abs(modes[:,:,-1])))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError('Nonfinite measured metric')
        passed=all(metrics[key]>=entry['boundary'] if entry['comparison']=='>=' else
                   metrics[key]<=entry['boundary'] for key,entry in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exception:
        report['errors']=[f'{type(exception).__name__}: {exception}']
        return report,2


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,exit_code=evaluate(args.prediction)
    try:
        destination=Path(args.report)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exception:
        print(f'Cannot write report: {exception}',file=sys.stderr)
        return 2
    print(json.dumps(report,indent=2,allow_nan=False))
    return exit_code


if __name__=='__main__':
    sys.exit(main())

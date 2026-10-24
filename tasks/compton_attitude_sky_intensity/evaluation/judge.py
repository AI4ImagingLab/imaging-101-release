"""Standalone NPZ evaluator; no submitted code or reconstruction imports."""
from pathlib import Path
import argparse,json,sys,zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPE=(1,3072)

def _field(path):
    path=Path(path)
    if not path.is_file():raise ValueError('Missing NPZ file: '+str(path))
    with np.load(path,allow_pickle=False) as data:
        if 'reconstruction' not in data:raise ValueError('Missing reconstruction key: '+str(path))
        array=data['reconstruction']
    if array.shape!=SHAPE:raise ValueError('reconstruction must have shape (1,3072)')
    if array.dtype.kind!='f' or array.dtype.itemsize not in (4,8):raise ValueError('reconstruction must have real float32 or float64 dtype')
    if not np.isfinite(array).all():raise ValueError('reconstruction contains nonfinite values')
    if np.any(array<0):raise ValueError('reconstruction contains negative intensity')
    return array.astype(np.float64).ravel()

def _observation_sum(archive,key):
    with archive.open(key+'.npy') as handle:
        version=np.lib.format.read_magic(handle)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(handle)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(handle)
        else:raise ValueError('Invalid public observation NPY format')
        if shape!=(1,278,1,60,3072) or fortran or dtype!=np.dtype('uint8'):raise ValueError('Invalid public count contract: '+key)
        count=0;total=0
        while True:
            block=handle.read(2**20)
            if not block:break
            count+=len(block);total+=int(np.frombuffer(block,dtype=np.uint8).sum(dtype=np.uint64))
        if count!=int(np.prod(shape)):raise ValueError('Truncated public count array: '+key)
    return total

def evaluate(prediction,evaluation_dir=None,public_data_path=None):
    directory=HERE if evaluation_dir is None else Path(evaluation_dir)
    public=HERE.parent/'data/raw_data.npz' if public_data_path is None else Path(public_data_path)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        with (directory/'metrics.json').open(encoding='utf-8') as handle:config=json.load(handle)
        directions={'ncc_boundary':'>=','nrmse_boundary':'<=','source_count_relative_error_boundary':'<='}
        for key,direction in directions.items():
            value=config.get(key)
            if type(value) not in (int,float) or not np.isfinite(value):raise ValueError('Invalid numeric metric configuration: '+key)
            if key=='ncc_boundary' and not 0<value<=1:raise ValueError('NCC boundary must lie in (0,1]')
            if key!='ncc_boundary' and value<=0:raise ValueError('Error boundaries must be positive')
            report['thresholds'][key.removesuffix('_boundary')]=dict(comparison=direction,boundary=value)
        reference_path=directory/'ground_truth.npz'
        if not reference_path.exists() and not reference_path.is_symlink():reference_path=directory/'baseline_reference.npz'
        reference=_field(reference_path)
        span=float(np.ptp(reference));norm=float(np.linalg.norm(reference))
        if not np.isfinite([span,norm]).all() or span<=0 or norm<=0:raise ValueError('Undefined image metrics for nonfinite, constant or zero reference scale')
        estimate=_field(prediction)
        with np.load(public,allow_pickle=False) as data:exposure=data['exposure_cm2_s_sr']
        if exposure.shape!=SHAPE or exposure.dtype!=np.float32 or not np.isfinite(exposure).all() or np.any(exposure<=0):raise ValueError('Invalid public exposure calibration')
        with zipfile.ZipFile(public) as archive:source_count=_observation_sum(archive,'events')-_observation_sum(archive,'background')
        if source_count<=0:raise ValueError('Undefined source-count closure')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            estimate_norm=float(np.linalg.norm(estimate))
            ncc=0. if estimate_norm==0 else float(np.dot(estimate,reference)/(estimate_norm*norm))
            nrmse=float(np.sqrt(np.mean((estimate-reference)**2))/span)
            closure=float(abs(np.dot(exposure.ravel().astype(np.float64),estimate)-source_count)/source_count)
        if not np.isfinite([estimate_norm,ncc,nrmse,closure]).all():raise ValueError('Nonfinite derived metric or estimate norm')
        ncc=float(np.clip(ncc,-1,1))
        report['metrics']=dict(ncc=ncc,nrmse=nrmse,source_count_relative_error=closure)
        passed=ncc>=config['ncc_boundary'] and nrmse<=config['nrmse_boundary'] and closure<=config['source_count_relative_error_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed))
        return report,0 if passed else 1
    except Exception as error:
        report['metrics']={};report['errors']=[type(error).__name__+': '+str(error)]
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as error:
        print('Cannot write judge report: '+str(error),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())

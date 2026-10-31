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

def available_memory():
    """Available host RAM; resource guard is unrelated to scientific acceptance."""
    import os
    if os.name=='nt':
        import ctypes
        class State(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('totalpage',ctypes.c_ulonglong),('availablepage',ctypes.c_ulonglong),('totalvirtual',ctypes.c_ulonglong),('availablevirtual',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
        state=State();state.length=ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError('Cannot read available memory')
        return int(state.available)
    try:
        fields=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(fields['MemAvailable'].split()[0])*1024
    except Exception: return 2**63-1


def public_context(path):
    from scipy.sparse import coo_matrix
    with np.load(path,allow_pickle=False) as z:
        counts=z['events'];background=z['background'];coordinates=z['attitude_coordinates'];weights=z['attitude_weights_s'];exposure=z['exposure_cm2_s_sr'];area=z['response_area_cm2']
    for name,arr in [('events',counts),('background',background)]:
        if arr.shape!=(1,278,1,60,3072) or arr.dtype!=np.uint8:raise ValueError('Invalid public count contract: '+name)
    if coordinates.shape!=(1,3,3416064) or coordinates.dtype!=np.int32 or np.any(coordinates<1):raise ValueError('Invalid attitude coordinates')
    c=coordinates[0].astype(np.int64)-1
    if np.any(c[0]>=278) or np.any(c[1:]>=3072):raise ValueError('Out-of-range attitude coordinates')
    if weights.shape!=(1,3416064) or weights.dtype!=np.float32 or not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Invalid attitude weights')
    if area.shape!=(1,1) or area.dtype!=np.float32 or not np.isfinite(area).all() or area.item()<=0:raise ValueError('Invalid calibrated response area')
    if exposure.shape!=SHAPE or exposure.dtype!=np.float32 or not np.isfinite(exposure).all() or np.any(exposure<=0):raise ValueError('Invalid exposure calibration')
    rotation=coo_matrix((weights[0].astype(float),(c[0]*3072+c[2],c[1])),shape=(278*3072,3072)).tocsr()
    return dict(counts=counts.reshape(278,184320),background=background.reshape(278,184320),rotation=rotation,area=float(area.item()),exposure=exposure.ravel().astype(float))


def factorized_blocks(fields,public_path,context,block_columns=2048):
    """Complete independent float64 forward; integer response stays disk mapped.

    Every dot product uses all 3072 incoming directions and all 278 attitudes.
    This generator never imports the reconstruction operator or reads its cache.
    """
    import tempfile,shutil
    fields=np.asarray(fields,dtype=float).reshape(-1,3072)
    rotation=context['rotation'];local=np.stack([(rotation@x).reshape(278,3072) for x in fields])
    local*=context['area']*(4*np.pi/3072)
    with tempfile.TemporaryDirectory(prefix='compton_judge_') as temporary:
        target=Path(temporary)/'response.npy'
        with zipfile.ZipFile(public_path) as archive,archive.open('response_counts.npy') as source,target.open('wb') as output:
            shutil.copyfileobj(source,output,length=2**20)
        mapped=np.load(target,mmap_mode='r',allow_pickle=False)
        try:
            if mapped.shape!=(1,3072,1,1,60,3072) or mapped.dtype!=np.uint16:raise ValueError('Invalid integer response contract')
            response=mapped.reshape(3072,184320)
            for start in range(0,184320,block_columns):
                free=available_memory()
                if free<6*1024**3+160*1024**2:raise MemoryError('Evaluation needs 6 GiB available host headroom plus a small response tile')
                stop=min(start+block_columns,184320)
                tile=np.array(response[:,start:stop],dtype=float,order='C')
                # Stack fields only along the leading batch; no complete float
                # response or 51-million-bin float64 expectation is allocated.
                prediction=(local.reshape(-1,3072)@tile).reshape(len(fields),278,stop-start)
                if not np.isfinite(prediction).all() or np.any(prediction<0):raise ValueError('Invalid predicted source counts')
                yield start,stop,prediction,free
        finally:mapped._mmap.close()


def deviance_sum(observed,mean):
    observed=np.asarray(observed,dtype=float);mean=np.asarray(mean,dtype=float)
    if not np.isfinite(mean).all() or np.any(mean<0):raise ValueError('Nonphysical or nonfinite expected counts')
    positive=observed>0
    unsupported=int(np.count_nonzero(positive&(mean==0)))
    terms=mean-observed
    terms[positive]+=observed[positive]*(np.log(observed[positive])-np.log(np.maximum(mean[positive],np.finfo(float).tiny)))
    return float(2*np.sum(terms,dtype=float)),unsupported


def forward_statistics(fields,public_path,context=None,callback=None):
    if context is None:context=public_context(public_path)
    count=len(np.asarray(fields).reshape(-1,3072));sums=np.zeros(count);unsupported=np.zeros(count,dtype=np.int64);source=np.zeros(count)
    minimum=available_memory();bins=0
    for start,stop,predictions,free in factorized_blocks(fields,public_path,context):
        observed=context['counts'][:,start:stop];background=context['background'][:,start:stop]
        for i,prediction in enumerate(predictions):
            score,invalid=deviance_sum(observed,prediction+background)
            sums[i]+=score;unsupported[i]+=invalid;source[i]+=float(np.sum(prediction,dtype=float))
        if callback is not None:callback(start,stop,predictions)
        minimum=min(minimum,free,available_memory());bins+=observed.size
    total=int(context['counts'].sum(dtype=np.uint64))
    if total<=0 or bins!=278*184320:raise ValueError('Incomplete or degenerate full-data reduction')
    return dict(normalized_poisson_deviance=sums/total,unsupported_positive_bins=unsupported,forward_source_counts=source,observation_bins=bins,total_events=total,minimum_available_memory=minimum)


def evaluate(prediction,evaluation_dir=None,public_data_path=None):
    directory=HERE if evaluation_dir is None else Path(evaluation_dir)
    public=HERE.parent/'data/raw_data.npz' if public_data_path is None else Path(public_data_path)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        for key in ['normalized_poisson_deviance_boundary','source_count_relative_error_boundary']:
            value=config.get(key)
            if type(value) not in (int,float) or not np.isfinite(value) or value<=0:raise ValueError('Invalid positive metric boundary '+key)
        if type(config.get('unsupported_positive_bins_boundary')) is not int or config['unsupported_positive_bins_boundary']!=0:raise ValueError('Unsupported positive-count bins must have boundary zero')
        truth=directory/'ground_truth.npz'
        if truth.exists() or truth.is_symlink():raise ValueError('Unexpected actual truth requires explicit truth-quality protocol certification')
        estimate=_field(prediction)
        # Reject finite-arithmetic overflow before loading the large response.
        with np.load(public,allow_pickle=False) as z:exposure=z['exposure_cm2_s_sr']
        if exposure.shape!=SHAPE or exposure.dtype!=np.float32 or not np.isfinite(exposure).all() or np.any(exposure<=0):raise ValueError('Invalid exposure calibration')
        with np.errstate(over='raise',invalid='raise'):
            folded_total=float(exposure.ravel().astype(float)@estimate)
        if not np.isfinite(folded_total):raise ValueError('Nonfinite source count')
        context=public_context(public)
        source_count=int(context['counts'].sum(dtype=np.uint64))-int(context['background'].sum(dtype=np.uint64))
        if source_count<=0:raise ValueError('Undefined source-count closure')
        stats=forward_statistics(estimate[None],public,context)
        metrics=dict(normalized_poisson_deviance=float(stats['normalized_poisson_deviance'][0]),source_count_relative_error=abs(folded_total-source_count)/source_count,unsupported_positive_bins=int(stats['unsupported_positive_bins'][0]))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite derived metric')
        thresholds={k:dict(comparison='<=',boundary=config[k+'_boundary']) for k in metrics}
        passed=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds,evaluation_details=dict(observation_bins=int(stats['observation_bins']),minimum_available_host_bytes=int(stats['minimum_available_memory']),response_storage='uint16 read-only memmap; float64 response tiles only'))
        return report,0 if passed else 1
    except Exception as error:report['metrics']={};report['errors']=[type(error).__name__+': '+str(error)];return report,2


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

"""Independent NPZ evaluator: actual modulus truth and public mechanical model."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ASSET_DIRECTORY=Path(__file__).resolve().parent
SHAPE=(1,800)

def _field(path: Path,role: str) -> np.ndarray:
    if not path.is_file():raise ValueError(f'Missing {role} file: {path.name}')
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'reconstruction'}:raise ValueError(f'{role} requires exactly reconstruction key')
        value=data['reconstruction']
    if value.shape!=SHAPE:raise ValueError(f'{role} shape must be {SHAPE}')
    if value.dtype!=np.dtype('float64'):raise ValueError(f'{role} dtype must be real float64')
    if not np.isfinite(value).all():raise ValueError(f'{role} contains NaN or infinity')
    if np.any(value<=0):raise ValueError(f'{role} must contain positive moduli')
    return value[0]

def _calibration(path: Path):
    schema={'nodes':((1,441,2),'float64'),'triangles':((1,800,3),'int64'),
            'displacement':((1,441,2),'float64'),'nodal_force':((1,441,2),'float64'),
            'fixed_dofs':((1,441,2),'bool')}
    result={}
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=set(schema):raise ValueError('Malformed public calibration keys')
        for name,(shape,dtype) in schema.items():
            value=data[name]
            if value.shape!=shape or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all():
                raise ValueError(f'Malformed public calibration {name}')
            result[name]=value[0]
    if result['triangles'].min()<0 or result['triangles'].max()>=441:
        raise ValueError('Public calibration has invalid connectivity')
    if np.linalg.norm(result['displacement'])==0:raise ValueError('Zero-norm observed displacement')
    return result

def _reference(asset_directory: Path):
    truth=asset_directory/'ground_truth.npz'
    path=truth if truth.exists() else asset_directory/'baseline_reference.npz'
    reference=_field(path,'hidden reference')
    if np.ptp(reference)<=0 or np.linalg.norm(reference)==0:
        raise ValueError('Hidden reference has undefined dynamic-range/NCC metrics')
    return reference

def _finite_json_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')

def _configuration(path: Path):
    with path.open(encoding='utf8') as handle:config=json.load(handle,parse_constant=_finite_json_constant)
    if not isinstance(config,dict):raise ValueError('Metric configuration must be an object')
    required={'ncc_boundary':('ncc_vs_ref','>='),
              'nrmse_boundary':('nrmse_vs_ref','<='),
              'relative_displacement_error_boundary':('relative_displacement_error','<='),
              'maximum_abs_principal_strain_boundary':('maximum_abs_principal_strain','<=')}
    thresholds={}
    for key,(metric,comparison) in required.items():
        boundary=config.get(key)
        if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary):
            raise ValueError(f'Missing or undefined metric boundary: {key}')
        if key=='ncc_boundary':
            if not -1<=boundary<=1:raise ValueError('NCC boundary outside[-1,1]')
        elif boundary<0:raise ValueError('Negative metric error boundary')
        thresholds[metric]={'comparison':comparison,'boundary':float(boundary)}
    return thresholds

def array_metrics(estimate: np.ndarray,reference: np.ndarray):
    """Cosine NCC and RMS error divided by the unaltered reference range."""
    return {'ncc_vs_ref':float(estimate@reference/(np.linalg.norm(estimate)*np.linalg.norm(reference))),
            'nrmse_vs_ref':float(np.sqrt(np.mean((estimate-reference)**2))/np.ptp(reference))}

def mechanical_metrics(modulus: np.ndarray,calibration: dict):
    """Independent affine-basis FEM, penalty boundary and physical strain.

    Basis coefficients come from a batched coordinate Vandermonde inverse,
    independently from the solver's explicit signed-coordinate B formula.
    """
    triangles=calibration['triangles'];points=calibration['nodes'][triangles]
    V=np.concatenate([np.ones((800,3,1)),points],axis=2)
    inverse=np.linalg.inv(V)
    dx=inverse[:,1,:];dy=inverse[:,2,:]
    area=.5*np.abs(np.linalg.det(V))
    if np.any(area<=0):raise ValueError('Degenerate public geometry')
    B=np.zeros((800,3,6),dtype=np.float64)
    B[:,0,::2]=dx;B[:,1,1::2]=dy;B[:,2,::2]=dy;B[:,2,1::2]=dx
    nu=.35;thickness=.05;penalty_multiplier=100000.
    tensor=np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    constitutive=modulus[:,None,None]*tensor[None,:,:]/(1-nu*nu)
    element=area[:,None,None]*thickness*(B.transpose(0,2,1)@constitutive@B)
    dofs=np.stack([2*triangles,2*triangles+1],axis=-1).reshape(800,6)
    rows=np.broadcast_to(dofs[:,:,None],(800,6,6)).ravel()
    columns=np.broadcast_to(dofs[:,None,:],(800,6,6)).ravel()
    K=coo_matrix((element.ravel(),(rows,columns)),shape=(882,882)).tocsc()
    diagonal=K.diagonal();maximum=diagonal.max()
    diagonal[calibration['fixed_dofs'].reshape(-1)]=penalty_multiplier*maximum
    K.setdiag(diagonal)
    predicted=spsolve(K,calibration['nodal_force'].reshape(-1))
    observed=calibration['displacement'].reshape(-1)
    relative=float(np.linalg.norm(predicted-observed)/np.linalg.norm(observed))
    strain=np.einsum('eij,ej->ei',B,predicted[dofs])
    center=(strain[:,0]+strain[:,1])/2
    radius=np.sqrt(((strain[:,0]-strain[:,1])/2)**2+(strain[:,2]/2)**2)
    maximum_strain=float(np.max(np.maximum(abs(center+radius),abs(center-radius))))
    return {'relative_displacement_error':relative,'maximum_abs_principal_strain':maximum_strain}

def evaluate(prediction: str | Path,asset_directory: str | Path=ASSET_DIRECTORY):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_directory)
        estimate=_field(Path(prediction),'prediction')
        reference=_reference(assets)
        calibration=_calibration(assets.parent/'data/raw_data.npz')
        thresholds=_configuration(assets/'metrics.json')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            metrics=array_metrics(estimate,reference)
            metrics.update(mechanical_metrics(estimate,calibration))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError('Undefined numerical evaluation metric')
        passed=all((metrics[key]>=value['boundary'] if value['comparison']=='>=' else
                    metrics[key]<=value['boundary']) for key,value in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),
                      metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile,FloatingPointError,np.linalg.LinAlgError) as error:
        report['errors']=[str(error)]
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except (OSError,TypeError,ValueError) as error:
        print(f'Unable to write Judge report: {error}',file=sys.stderr)
        return 2
    return code

if __name__=='__main__':raise SystemExit(main())

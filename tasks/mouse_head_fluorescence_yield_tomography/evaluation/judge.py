"""Independent standalone yield/observation Judge; no submitted-code imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
HERE=Path(__file__).resolve().parent
SHAPE=(1,5707)

def _load(path):
    if not Path(path).is_file():raise ValueError('NPZ file does not exist: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={'reconstruction'}:raise ValueError('Require exactly reconstruction')
        x=f['reconstruction']
    if x.shape!=SHAPE or x.dtype!=np.float64:raise ValueError('Require real float64 shape(1,5707)')
    if not np.isfinite(x).all():raise ValueError('Nonfinite yield')
    if np.any(x<0):raise ValueError('Yield must be nonnegative')
    return x[0]

def field_metrics(x,y):
    if not np.isfinite(y).all() or np.linalg.norm(y)==0 or np.ptp(y)==0:raise ValueError('Degenerate reference')
    norm=np.linalg.norm(x)
    return {'yield_ncc':float((x@y)/norm/np.linalg.norm(y)) if norm else 0.,'yield_nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y)),'yield_max_abs':float(np.max(np.abs(x-y)))}

def observation_context():
    # Independently assembled exact P1 matrices, not hidden forward arrays.
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:
        raw={k:f[k][0] for k in f.files}
    expected={'nodes':(5707,3),'tetrahedra':(29448,4),'boundary_vertices':(5707,),'source_positions':(8,3),'detector_positions':(8,3),'links':(56,2),'fluorescence':(56,),'excitation':(56,),'coarse_nodes':(3323,3),'coarse_tetrahedra':(17350,4)}
    if set(raw)!=set(expected):raise ValueError('Invalid public observation keys')
    for k,a in raw.items():
        if a.shape!=expected[k] or not np.isfinite(a).all():raise ValueError('Invalid public observation '+k)
    nodes=raw['nodes'];e=raw['tetrahedra'];n=len(nodes)
    if e.dtype!=np.int64 or e.min()<0 or e.max()>=n:raise ValueError('Invalid tetrahedra')
    if raw['links'].dtype!=np.int64 or raw['links'].min()<0 or raw['links'].max()>7:raise ValueError('Invalid links')
    if raw['boundary_vertices'].dtype!=np.bool_:raise ValueError('Invalid boundary flags')
    if np.any(raw['fluorescence']<=0) or np.any(raw['excitation']<=0):raise ValueError('Nonpositive observations')
    p=nodes[e];affine=np.concatenate((np.ones((*p.shape[:2],1)),p),axis=2);grad=np.linalg.inv(affine)[:,1:,:];v=np.abs(np.linalg.det(p[:,1:]-p[:,:1]))/6
    if np.any(v<=0):raise ValueError('Degenerate tetrahedra')
    M=v[:,None,None]*(np.ones((4,4))+np.eye(4))[None]/20
    K=v[:,None,None]*np.einsum('eki,ekj->eij',grad,grad)
    row=np.broadcast_to(e[:,:,None],M.shape).ravel();col=np.broadcast_to(e[:,None,:],M.shape).ravel()
    def matrix(values):return coo_matrix((values.ravel(),(row,col)),shape=(n,n)).tocsc()
    mass=matrix(M)
    faces=np.concatenate([np.sort(e[:,q],axis=1) for q in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]);faces,count=np.unique(faces,axis=0,return_counts=True);faces=faces[count==1]
    q=nodes[faces];area=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)/2
    nref=1.33;r0=((nref-1)/(nref+1))**2;c=np.abs(np.cos(np.arcsin(1/nref)));ksi=1/(2*((2/(1-r0)-1+c**3)/(1-c*c)))
    B=area[:,None,None]*ksi*(np.ones((3,3))+np.eye(3))[None]/12
    robin=coo_matrix((B.ravel(),(np.broadcast_to(faces[:,:,None],B.shape).ravel(),np.broadcast_to(faces[:,None,:],B.shape).ravel())),shape=(n,n)).tocsc()
    Ax=matrix(.251953*K+.009*M)+robin;Am=matrix(.26062*K+.006*M)+robin
    inverse=np.linalg.inv(np.transpose(p[:,1:]-p[:,:1],(0,2,1)))
    def point_values(points,boundary_only=False):
        R=np.zeros((len(points),n))
        for j,point in enumerate(points):
            xyz=np.einsum('eij,ej->ei',inverse,point-p[:,0]);b=np.column_stack((1-xyz.sum(axis=1),xyz));valid=np.flatnonzero(b.min(axis=1)>=-1e-10)
            if not len(valid):raise ValueError('Optode outside geometry')
            i=valid[0];w=b[i].copy()
            if boundary_only:
                for k in range(4):
                    if not raw['boundary_vertices'][e[i,k]]:w[k]=0;w/=w.sum()
            R[j,e[i]]=w
        return R
    phix=splu(Ax.tocsc()).solve(point_values(raw['source_positions']).T*np.cos(.15))
    detector=point_values(raw['detector_positions'],True);links=raw['links'];observed=raw['fluorescence']*(detector@phix)[links[:,1],links[:,0]]/raw['excitation']
    return phix,mass,splu(Am.tocsc()),detector,links,observed

def forward_error(x,context):
    phix,mass,lu,detector,links,observed=context
    field=lu.solve(np.asarray(mass@(x[:,None]*phix)))
    predicted=np.abs((detector@field)[links[:,1],links[:,0]])
    return float(np.linalg.norm(predicted-observed)/np.linalg.norm(observed))

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        names=['ncc_boundary','nrmse_boundary','max_abs_boundary','forward_relative_l2_boundary']
        for k in names:
            a=cfg[k]
            if isinstance(a,bool) or not isinstance(a,(int,float)) or not np.isfinite(a):raise ValueError('Undefined threshold '+k)
        if not -1<=cfg[names[0]]<=1 or any(cfg[k]<0 for k in names[1:]):raise ValueError('Invalid threshold range')
        truth=HERE/'ground_truth.npz';y=_load(truth if truth.exists() else HERE/'baseline_reference.npz');field_metrics(y,y)
        context=observation_context()
    except Exception as exc:report['errors']=['Evaluator error: '+str(exc)];return report
    report['thresholds']={k:{'comparison':op,'boundary':cfg[b]} for k,op,b in [('yield_ncc','>=',names[0]),('yield_nrmse','<=',names[1]),('yield_max_abs','<=',names[2]),('forward_relative_L2','<=',names[3])]}
    try:
        x=_load(prediction);metrics=field_metrics(x,y);metrics['forward_relative_L2']=forward_error(x,context)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Metric overflow')
    except Exception as exc:report['errors']=['Invalid prediction: '+str(exc)];return report
    passed=all(v>=report['thresholds'][k]['boundary'] if report['thresholds'][k]['comparison']=='>=' else v<=report['thresholds'][k]['boundary'] for k,v in metrics.items())
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics);return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv);report=evaluate(args.prediction)
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())

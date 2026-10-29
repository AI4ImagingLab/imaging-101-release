"""Explicit gauged Young-Dupre inversion and variational Laplace pressure solve.

Adapted from foambryo, Sacha Ichbiah and contributors, GPL-3.0-or-later.
"""
import numpy as np
from scipy import linalg
from .preprocessing import prepare_mesh
from .physics_model import junction_geometry,area_volume_operators,tension_system,interface_curvatures

def solve_tensions(matrix,rhs,mean_tension=1.,logarithmic=False):
    """Solve homogeneous physics least squares with the exact mean-tension gauge.

    A power-of-two gauge-row weight balances median physics row norms. After
    dense SVD least squares, rescale the homogeneous solution to exact mean.
    This retains the author's solution, normalization and selected dense path.
    """
    physics=matrix[:-1];n=matrix.shape[1]
    norms=np.linalg.norm(physics,axis=1);norms=norms[norms>0]
    w=float(2.**np.round(np.log2(np.median(norms)/np.sqrt(n)))) if len(norms) else 1.
    A=np.vstack([physics,np.full((1,n),w)])
    b=np.r_[rhs[:-1],w*n*(np.log(mean_tension) if logarithmic else mean_tension)]
    x=linalg.lstsq(A,b)[0]
    if logarithmic:
        x+=(n*np.log(mean_tension)-x.sum())/n
        return np.exp(x)
    if x.sum()==0 or not np.isfinite(x).all():raise ArithmeticError('Undetermined tension gauge')
    return x*(n*mean_tension/x.sum())

def solve_pressures(D,V,tensions):
    """Solve (Vcell Vcell^T)p=Vcell D^T gamma with medium pressure zero.

    The author's row-normalized Gram blocks cancel analytically in this
    product; use the same dense inverse convention with explicit unnormalized
    blocks. Pressure units are mean tension per author coordinate voxel.
    """
    cells=V[1:]
    G=(cells@cells.T).toarray();H=(cells@D.T).toarray()
    return np.r_[0.,np.linalg.inv(G)@(H@tensions)]

def solve_variational_tensions(D,V,route='gram'):
    """Paper Gram-Schur route, or the source's later direct-operator comparison.

    Gram preserves the original per-row normalization before constrained
    least squares. Operator instead minimizes vertex-force residual directly,
    with an explicit Householder elimination of the mean-tension constraint.
    They are separate noisy-data estimators and are never relabeled as equal.
    """
    nm=D.shape[0];cells=V[1:]
    if route=='gram':
        F=(D@D.T).toarray();H=(D@cells.T).toarray();G=(cells@cells.T).toarray()
        M=np.block([[F,-H],[H.T,-G]])
        rows=M.sum(axis=1);nonzero=rows!=0;M[nonzero]/=rows[nonzero,None]
        Fg=M[:nm,:nm];Fp=-M[:nm,nm:];Gg=M[nm:,:nm];Gp=-M[nm:,nm:]
        schur=Fg-Fp@np.linalg.inv(Gp)@Gg
        return solve_tensions(np.vstack([schur,np.ones(nm)]),np.r_[np.zeros(nm),nm])
    if route!='operator':raise ValueError('Unknown variational route')
    n=nm+cells.shape[0]
    operator=np.empty((D.shape[1],n));operator[:,:nm]=D.T.toarray();operator[:,nm:]=-cells.T.toarray()
    anchor=np.r_[np.full(nm,1/nm),np.zeros(n-nm)];unit=anchor/np.linalg.norm(anchor)
    reflector=unit-np.eye(n)[0];rnorm=np.linalg.norm(reflector)
    basis=np.eye(n)[:,1:] if rnorm<1e-14 else (np.eye(n)-2*np.outer(reflector/rnorm,reflector/rnorm))[:,1:]
    particular=anchor/(anchor@anchor)
    reduced=operator@basis;rhs=-(operator@particular)
    answer=particular+basis@linalg.lstsq(reduced,rhs)[0]
    return answer[:nm]

def solve_laplace_pressures(mesh,geometry,tensions,interfaces,cells,areas,weighted_curvature=False):
    """Area-weighted interface Laplace relations, exterior pinned by elimination."""
    h=interface_curvatures(mesh,geometry,weighted=weighted_curvature)
    ids=np.searchsorted(cells,interfaces);valid=np.isfinite(h)
    A=np.zeros((np.count_nonzero(valid),len(cells)))
    weight=areas[valid]/areas.sum();rows=np.arange(len(weight));pairs=ids[valid]
    A[rows,pairs[:,0]]=weight;A[rows,pairs[:,1]]=-weight
    b=-2*h[valid]*tensions[valid]*weight
    neighbors={i:set() for i in range(len(cells))}
    for a,c in pairs:neighbors[int(a)].add(int(c));neighbors[int(c)].add(int(a))
    connected={0};stack=[0]
    while stack:
        for c in neighbors[stack.pop()]:
            if c not in connected:connected.add(c);stack.append(c)
    columns=np.array(sorted(connected-{0}),dtype=int)
    p=np.full(len(cells),np.nan);p[0]=0
    if len(columns):p[columns]=linalg.lstsq(A[:,columns],b)[0]
    return p

def compare_formulations(result,include_operator=True):
    """Actively compute the paper families on the exact same measured geometry."""
    outputs={}
    for name in ['young_dupre','projection','lami','inverse_lami','log_lami']:
        M,b=tension_system(result['geometry'],result['interfaces'],method=name)
        gamma=solve_tensions(M,b,logarithmic=name=='log_lami')
        outputs[name]={'tensions':gamma,'pressures':solve_pressures(result['D'],result['V'],gamma)}
    for route in ['gram']+(['operator'] if include_operator else []):
        gamma=solve_variational_tensions(result['D'],result['V'],route=route)
        outputs['variational_'+route]={'tensions':gamma,'pressures':solve_pressures(result['D'],result['V'],gamma)}
    for weighted in [False,True]:
        outputs['laplace_weighted_curvature' if weighted else 'laplace']={'pressures':solve_laplace_pressures(result['mesh'],result['geometry'],result['tensions'],result['interfaces'],result['cells'],result['areas'],weighted_curvature=weighted)}
    M,b=tension_system(result['geometry'],result['interfaces'],weighted=False)
    gamma=solve_tensions(M,b)
    outputs['unweighted_young_dupre']={'tensions':gamma,'pressures':solve_pressures(result['D'],result['V'],gamma)}
    return outputs

def reconstruct(observations,weighted=True):
    """Recover all357 interface tensions and all65 region pressures from mesh."""
    mesh=prepare_mesh(**observations)
    geometry=junction_geometry(mesh)
    D,V,interfaces,cells,areas=area_volume_operators(mesh)
    M,b=tension_system(geometry,interfaces,weighted=weighted)
    tensions=solve_tensions(M,b)
    pressures=solve_pressures(D,V,tensions)
    return {'tensions':tensions,'pressures':pressures,'interfaces':interfaces,'cells':cells,
            'mesh':mesh,'geometry':geometry,'D':D,'V':V,'areas':areas,'M':M,'b':b}

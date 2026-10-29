"""Explicit triangular geometry, Young-Dupre balance and variational operators.

Adapted from foambryo, Sacha Ichbiah and contributors, GPL-3.0-or-later.
The native author's chord angle and junction-length conventions are retained.
"""
import numpy as np
from scipy import sparse

def triangle_geometry(mesh):
    """Return triangle coordinates, outward unit normals and voxel-squared areas."""
    p=mesh.vertices[mesh.triangles]
    n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,1])
    length=np.linalg.norm(n,axis=1)
    if np.any(length==0):raise ValueError('Degenerate observed triangle')
    return p,n/length[:,None],length/2

def junction_geometry(mesh):
    """Read chord wedge angles and historical 6-times line lengths.

    Each geometric edge is grouped by its two vertex indices. A valid triple
    junction has exactly three incident triangles separating exactly three
    distinct pairs of three regions. Other edge valences carry no measured
    angle; malformed three-region edges are explicitly returned for review.
    For parity, visit each incident half edge and append each angle twice,
    as the selected author code does. No geometry is silently modified.
    """
    _,normals,_=triangle_geometry(mesh)
    t=mesh.triangles
    half=np.stack([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]],axis=1).reshape(-1,2)
    groups={}
    for k,e in enumerate(np.sort(half,axis=1)):
        key=(int(e[0]),int(e[1]));groups.setdefault(key,[]).append(k)
    angles={};lengths={};malformed=[];valid_edges=[];junction_vertices=[]
    for edge,indices in groups.items():
        faces=np.array(indices)//3;materials=mesh.region_labels[faces]
        if len(indices)>2:junction_vertices.extend(edge)
        nregions=len(np.unique(materials));npairs=len(np.unique(materials,axis=0))
        valid=len(indices)==3 and nregions==3 and npairs==3
        if nregions==3 and not valid:malformed.append(edge)
        if not valid:continue
        valid_edges.append(edge)
        length=np.linalg.norm(mesh.vertices[edge[1]]-mesh.vertices[edge[0]])
        for current in indices:
            visit=[current]+[i for i in indices if i!=current]
            f=np.array(visit)//3
            regions=mesh.region_labels[f].copy();n=normals[f].copy()
            for j in [1,2]:
                if regions[0,0]==regions[j,0] or regions[0,1]==regions[j,1]:
                    regions[j]=regions[j,::-1];n[j]*=-1
            for left,right in [(0,1),(1,2),(2,0)]:
                angle=np.arccos(np.clip(np.dot(n[left],n[right]),-1,1))
                if regions[left,1]==regions[right,0]:
                    e,f,g=regions[left,0],regions[left,1],regions[right,1]
                elif regions[left,0]==regions[right,1]:
                    e,f,g=regions[right,0],regions[right,1],regions[left,1]
                else:raise ValueError('Invalid material cycle at triple junction')
                key=(int(min(e,g)),int(f),int(max(e,g)))
                angles.setdefault(key,[]).extend([angle,angle])
                lengths[key]=lengths.get(key,0.)+length+length
    return {'angles':{k:float(np.mean(v)) for k,v in angles.items()},'lengths':lengths,
            'valid_edges':np.asarray(valid_edges,dtype=np.int64).reshape(-1,2),
            'junction_vertices':np.unique(junction_vertices),
            'malformed_edges':np.asarray(malformed,dtype=np.int64).reshape(-1,2)}

def _scatter(triangles,rows,slots,nrows,nvertices):
    values=np.stack(slots,axis=1)
    rr=np.broadcast_to(rows[:,None,None],values.shape).ravel()
    cc=(3*triangles[:,:,None]+np.arange(3)[None,None,:]).ravel()
    return sparse.coo_matrix((values.ravel(),(rr,cc)),shape=(nrows,3*nvertices)).tocsr()

def area_volume_operators(mesh):
    """Return D=dA/dvertices and V=dVolume/dvertices as explicit sparse matrices.

    Rows follow lexicographically sorted interface pairs and sorted region IDs.
    Columns follow vertex-major (x,y,z). V includes the medium row first.
    """
    p,n,faceareas=triangle_geometry(mesh)
    x,y,z=p[:,0],p[:,1],p[:,2]
    interfaces,rows=np.unique(mesh.region_labels,axis=0,return_inverse=True)
    cells=np.unique(mesh.region_labels)
    a=np.cross(n,z-y)/2;b=np.cross(n,x-z)/2;c=np.cross(n,y-x)/2
    D=_scatter(mesh.triangles,rows,(a,b,c),len(interfaces),len(mesh.vertices))
    a=np.cross(y,z)/6;b=np.cross(z,x)/6;c=np.cross(x,y)/6
    plus=np.searchsorted(cells,mesh.region_labels[:,0]);minus=np.searchsorted(cells,mesh.region_labels[:,1])
    V=_scatter(mesh.triangles,plus,(a,b,c),len(cells),len(mesh.vertices))
    V-= _scatter(mesh.triangles,minus,(a,b,c),len(cells),len(mesh.vertices))
    areas=np.bincount(rows,weights=faceareas,minlength=len(interfaces))
    return D,V,interfaces,cells,areas

def tension_system(geometry,interfaces,weighted=True,method='young_dupre'):
    """Assemble selected 3-cosine force balance plus last mean-tension row.

    One three-row block per unordered triple of regions. Junction angles are
    mean chord angles; each block's weight is its line length/total line length.
    The source's uniform 6-times length accounting cancels in this ratio.
    """
    if method not in ['young_dupre','projection','lami','inverse_lami','log_lami']:raise ValueError('Unknown tension formulation')
    angles=geometry['angles'];lengths=geometry['lengths']
    triples=sorted({tuple(sorted(k)) for k in angles},key=lambda k:(k[2],k[1],k[0]))
    if method in ['lami','log_lami']:
        eps=np.sqrt(8*np.finfo(float).eps)
        triples=[k for k in triples if all(eps<angles[key]<np.pi-eps for key in [(k[0],k[1],k[2]),(k[1],k[0],k[2]),(k[0],k[2],k[1])])]
    kept={tuple(sorted(k)) for k in triples}
    total=sum(v for k,v in lengths.items() if tuple(sorted(k)) in kept)/3
    lookup={tuple(k):i for i,k in enumerate(interfaces)}
    nr=2 if method=='projection' else 3
    M=np.zeros((nr*len(triples)+1,len(interfaces)));b=np.zeros(len(M));b[-1]=0 if method=='log_lami' else len(interfaces)
    for i,(a,bcell,c) in enumerate(triples):
        ab=angles[(a,bcell,c)];aa=angles[(bcell,a,c)];ac=angles[(a,c,bcell)]
        idx=[lookup[(a,bcell)],lookup[(bcell,c)],lookup[(a,c)]]
        factor=lengths[(a,bcell,c)]/total if weighted else 1.
        if method=='young_dupre':
            block=np.array([[1,np.cos(ab),np.cos(aa)],[np.cos(ab),1,np.cos(ac)],[np.cos(aa),np.cos(ac),1]])
        elif method=='projection':
            block=np.array([[1,np.cos(ab),np.cos(aa)],[0,np.sin(ab),-np.sin(aa)]])
        elif method=='lami':
            block=np.array([[1/np.sin(ac),-1/np.sin(aa),0],[1/np.sin(ac),0,-1/np.sin(ab)],[0,1/np.sin(aa),-1/np.sin(ab)]])
        elif method=='inverse_lami':
            block=np.array([[np.sin(aa),-np.sin(ac),0],[np.sin(ab),0,-np.sin(ac)],[0,np.sin(ab),-np.sin(aa)]])
        else:
            block=np.array([[1,-1,0],[1,0,-1],[0,1,-1]])
            b[3*i:3*i+3]=factor*np.array([np.log(np.sin(ac))-np.log(np.sin(aa)),np.log(np.sin(ac))-np.log(np.sin(ab)),np.log(np.sin(aa))-np.log(np.sin(ab))])
        M[nr*i:nr*i+nr,idx]=factor*block
    M[-1]=1
    return M,b

def interface_curvatures(mesh,geometry,weighted=False,return_diagnostics=False):
    """Original cotangent mean curvature, excluding all junction vertices.

    Vertex curvature uses the whole mesh cotangent Laplacian. Its sign is
    fixed by the area-weighted mean face normal. Each interface averages only
    its vertices away from junctions; an unmeasurable interface returns NaN.
    """
    p,n,areas=triangle_geometry(mesh);t=mesh.triangles
    a=np.linalg.norm(p[:,1]-p[:,2],axis=1);b=np.linalg.norm(p[:,0]-p[:,2],axis=1);c=np.linalg.norm(p[:,0]-p[:,1],axis=1)
    s=(a+b+c)/2;heron=np.sqrt(s*(s-a)*(s-b)*(s-c))
    cot=np.stack([b*b+c*c-a*a,a*a+c*c-b*b,a*a+b*b-c*c],axis=1)/(4*heron[:,None])
    ii=t[:,[1,2,0]];jj=t[:,[2,0,1]]
    L=sparse.coo_matrix((cot.ravel(),(jj.ravel(),ii.ravel())),shape=(len(mesh.vertices),len(mesh.vertices))).tocsr();L=L+L.T
    incident=np.bincount(t.ravel(),weights=np.repeat(heron,3),minlength=len(mesh.vertices))
    inverse_area=1/incident
    lcot=(L@mesh.vertices-mesh.vertices*np.asarray(L.sum(axis=1)))*(.75*inverse_area[:,None])
    # Keep the author's separate Heron expression and summation order for
    # vertex-normal weights. On nearly tangential curvature vectors, a
    # cross-product-area replacement can change the numerical curvature sign.
    sides=p-p[:,[2,0,1]];lengths=np.linalg.norm(sides,axis=2)
    hp=lengths.sum(axis=1)/2;diff=hp[:,None]-lengths
    normal_areas=(hp*diff[:,0]*diff[:,1]*diff[:,2])**.5
    normal_n=np.cross(p[:,0]-p[:,1],p[:,0]-p[:,2]);normal_n*=(1/np.linalg.norm(normal_n,axis=1))[:,None]
    normal=np.zeros_like(mesh.vertices);normal_weight=np.zeros(len(mesh.vertices))
    np.add.at(normal,t.ravel(),np.repeat(normal_n*normal_areas[:,None],3,axis=0))
    np.add.at(normal_weight,t.ravel(),np.repeat(normal_areas,3))
    normal/=normal_weight[:,None]
    curvature=np.sign(np.sum(lcot*normal,axis=1))*np.linalg.norm(lcot,axis=1)
    interior=np.ones(len(mesh.vertices),dtype=bool);interior[geometry['junction_vertices']]=False
    interfaces,inv=np.unique(mesh.region_labels,axis=0,return_inverse=True)
    result=[]
    for i in range(len(interfaces)):
        ids=np.unique(t[inv==i]);ids=ids[interior[ids]]
        if not len(ids):result.append(np.nan)
        elif weighted:result.append(np.sum(curvature[ids]*incident[ids])/np.sum(incident[ids]))
        else:result.append(np.mean(curvature[ids]))
    result=np.array(result)
    if return_diagnostics:
        return result,{'laplacian':lcot,'normals':normal,'inverse_area':inverse_area,'vertex_curvature':curvature}
    return result

def cell_volumes(mesh):
    """Signed enclosed volumes, including negative exterior, in author voxels cubed."""
    p,_,_=triangle_geometry(mesh)
    det=np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2]))/6
    cells=np.unique(mesh.region_labels)
    v=np.zeros(len(cells))
    np.add.at(v,np.searchsorted(cells,mesh.region_labels[:,0]),det)
    np.add.at(v,np.searchsorted(cells,mesh.region_labels[:,1]),-det)
    return cells,v

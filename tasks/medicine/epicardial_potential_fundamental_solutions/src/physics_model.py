"""Independent equation implementation of harmonic fundamental solutions.

Mathematical method: Wang/Rudy2006, Johnston/Tate normal-displacement variant.
No unlicensed upstream program text is included. Coordinates are numerical mm,
equivalently nondimensional x/(1mm); derivatives rescale by1mm for Cauchy fit.
"""
import numpy as np
def vertex_normals(vertices,faces):
    triangles=vertices[faces]
    first=triangles[:,1]-triangles[:,0];second=triangles[:,2]-triangles[:,0]
    cross=np.cross(first,second);unit=cross/np.linalg.norm(cross,axis=1)[:,None]
    weights=np.empty(faces.shape,dtype=float)
    for corner in (0,1):
        a=triangles[:,(corner+1)%3]-triangles[:,corner]
        b=triangles[:,(corner+2)%3]-triangles[:,corner]
        weights[:,corner]=np.arccos(np.einsum('ij,ij->i',a,b)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1))/np.pi
    weights[:,2]=1-weights[:,0]-weights[:,1]
    accum=np.zeros_like(vertices,dtype=float)
    # Triangle order is fixed by acquisition geometry; accumulate each corner.
    for tri,angles,n in zip(faces,weights,unit):
        for index,angle in zip(tri,angles):accum[index]+=angle*n
    return accum/np.linalg.norm(accum,axis=1)[:,None]
def virtual_sources(data,scale=.1):
    inner=data['heart_vertices'];outer=data['torso_vertices']
    ni=vertex_normals(inner,data['heart_faces']);no=vertex_normals(outer,data['torso_faces'])
    inside=inner-scale*np.linalg.norm(np.ptp(inner,axis=0))*ni
    outside=outer+scale*np.linalg.norm(np.ptp(outer,axis=0))*no
    return np.concatenate((outside,inside)),no
def kernel_matrices(points,sources,normals=None):
    delta=np.asarray(points)[:,None,:]-np.asarray(sources)[None,:,:]
    distance=np.linalg.norm(delta,axis=2)
    value=1/(4*np.pi*distance)
    derivative=None if normals is None else -np.einsum('ijk,ik->ij',delta,normals)/(4*np.pi*distance**3)
    return value,derivative
def fundamental_matrix(points,sources,normals):
    k,d=kernel_matrices(points,sources,normals)
    return np.vstack((np.column_stack((np.ones(len(points)),k)),np.column_stack((np.zeros(len(points)),d))))
def evaluate_potential(points,sources,coefficients):
    return coefficients[0]+kernel_matrices(points,sources)[0]@coefficients[1:]

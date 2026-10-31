"""Explicit real-harmonic asteroid geometry and acquisition operators.

Algorithm provenance: ADAM 25810f767b9de349a356df520109f4705f87f7a7,
CC BY 4.0; Viikinkoski, Kaasalainen and Durech (2015).
The octantoid and rotation expressions are evaluated locally with NumPy/SciPy.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.special import lpmv


@lru_cache(maxsize=8)
def sphere_grid(nrows=10):
    """Return native ordered colatitude/longitude and zero-based outward faces."""
    if isinstance(nrows,bool) or int(nrows)!=nrows or nrows<1:
        raise ValueError('nrows must be a positive integer')
    n=int(nrows);theta=[0.];phi=[0.];theta_step=np.pi/(2*n)
    nodes=np.zeros((2*n+1,4*n+1),dtype=np.int64);next_id=1
    for ring in range(1,2*n):
        size=ring if ring<=n else 2*n-ring
        for j in range(4*size):
            theta.append(ring*theta_step if ring<=n else np.pi-size*theta_step)
            phi.append(j*(np.pi/(2*size)))
            nodes[ring,j]=next_id;next_id+=1
        nodes[ring,4*size]=nodes[ring,0]
    theta.append(np.pi);phi.append(0.);nodes[2*n,0]=next_id
    faces=[]
    for ring in range(1,n+1):
        for sector in range(1,5):
            start=(sector-1)*ring
            faces.append([nodes[ring-1,start-sector+1],nodes[ring,start],nodes[ring,start+1]])
            for j in range(start+1,start+ring):
                faces.append([nodes[ring,j],nodes[ring-1,j-sector+1],nodes[ring-1,j-sector]])
                faces.append([nodes[ring-1,j-sector+1],nodes[ring,j],nodes[ring,j+1]])
    for ring in range(n+1,2*n+1):
        for sector in range(1,5):
            start=(sector-1)*(2*n-ring)
            faces.append([nodes[ring,start],nodes[ring-1,start+sector],nodes[ring-1,start+sector-1]])
            for j in range(start+1,start+2*n-ring+1):
                faces.append([nodes[ring,j],nodes[ring-1,j+sector-1],nodes[ring,j-1]])
                faces.append([nodes[ring,j],nodes[ring-1,j+sector],nodes[ring-1,j+sector-1]])
    result=np.asarray(theta),np.asarray(phi),np.asarray(faces,dtype=np.int64)
    for array in result:array.setflags(write=False)
    return result


@lru_cache(maxsize=8)
def harmonic_basis(lmax=9,nrows=10):
    """Real orthonormal harmonics with Condon--Shortley phase and index l(l+1)+m."""
    theta,phi,_=sphere_grid(nrows);basis=np.empty((len(theta),(lmax+1)**2))
    for degree in range(lmax+1):
        for order in range(-degree,degree+1):
            m=abs(order)
            normal=np.sqrt((2*degree+1)*math.factorial(degree-m)/(4*np.pi*math.factorial(degree+m)))
            # General associated-Legendre primitive with the same independent
            # normalization and Condon--Shortley convention as the paper/source.
            value=normal*lpmv(m,degree,np.cos(theta))
            if order>0:value=value*np.sqrt(2)*np.cos(m*phi)
            elif order<0:value=value*np.sqrt(2)*np.sin(m*phi)
            basis[:,degree*(degree+1)+order]=value
    basis.setflags(write=False)
    return basis


def octantoid(coefficients,lmax=9,nrows=10,padding=0):
    """Return mesh(km) and Cartesian-coordinate-major harmonic Jacobian.

    Coefficients are a,b,c harmonic blocks; x=e^a sin(theta)cos(phi),
    y=e^(a+b)sin(theta)sin(phi), z=e^(a+c)cos(theta). Optional padding
    appends an identity block for source joint spin differentiation.
    """
    a=np.asarray(coefficients,dtype=np.float64).reshape(-1);q=(lmax+1)**2
    if a.shape!=(3*q,) or not np.isfinite(a).all():raise ValueError('Invalid harmonic coefficients')
    theta,phi,_=sphere_grid(nrows);basis=harmonic_basis(lmax,nrows);nv=len(theta)
    mesh=np.stack([np.exp(basis@a[:q])*np.sin(theta)*np.cos(phi),
                   np.exp(basis@(a[:q]+a[q:2*q]))*np.sin(theta)*np.sin(phi),
                   np.exp(basis@(a[:q]+a[2*q:]))*np.cos(theta)],axis=1)
    jac=np.zeros((3*nv+padding,3*q+padding))
    for axis in range(3):
        block=mesh[:,axis,None]*basis;jac[axis*nv:(axis+1)*nv,:q]=block
        if axis:jac[axis*nv:(axis+1)*nv,axis*q:(axis+1)*q]=block
    if padding:jac[3*nv:,3*q:]=np.eye(padding)
    return mesh,jac


def rotation_matrix(angles,time,derivatives=False):
    """Inertial-to-body rotation; angles=(colat,longitude,rad/day,phase) in rad."""
    beta,longitude,omega,phase=np.asarray(angles,dtype=np.float64)
    cb,sb=np.cos(beta),np.sin(beta);cl,sl=np.cos(longitude),np.sin(longitude)
    cf,sf=np.cos(omega*time+phase),np.sin(omega*time+phase)
    spin=np.array([[cf,sf,0.],[-sf,cf,0.],[0.,0.,1.]])
    pole=np.array([[cb*cl,cb*sl,-sb],[-sl,cl,0.],[sb*cl,sb*sl,cb]])
    matrix=spin@pole
    if not derivatives:return matrix
    db=np.array([[-sb*cl,-sb*sl,-cb],[0.,0.,0.],[cb*cl,cb*sl,-sb]])
    dl=np.array([[-cb*sl,cb*cl,0.],[-cl,-sl,0.],[-sb*sl,sb*cl,0.]])
    dw=time*np.array([[-sf,cf,0.],[-cf,-sf,0.],[0.,0.,0.]])
    return matrix,np.stack([spin@db,spin@dl,dw@pole])

# BEGIN EMBEDDED SELECTED MATH
CPP_SOURCE = r'''

// Mathematical kernels adapted from ADAM 25810f767b9de349a356df520109f4705f87f7a7.
// Viikinkoski, Kaasalainen and Durech; upstream project CC BY 4.0.
// No upstream parser, data reader, domain library, optimizer, or object framework.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <complex>
#include <new>
#include <stdexcept>
#include <vector>
using Complex=std::complex<double>;
#define PI 3.141592653589793
#define PI2 9.869604401089358
#define EP 1E-8
#define DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define NORM(a) std::sqrt(DOT(a,a))
#define DET(a) ((a)[0]*(a)[4]*(a)[8]+(a)[1]*(a)[5]*(a)[6]+(a)[2]*(a)[3]*(a)[7]-(a)[2]*(a)[4]*(a)[6]-(a)[1]*(a)[3]*(a)[8]-(a)[0]*(a)[5]*(a)[7])
// A typed allocator initializes C++ complex object lifetimes explicitly.
struct Allocation {
 size_t bytes;bool zero;
 template<class T> operator T*() const {
   void* raw=std::malloc(bytes);if(!raw)throw std::bad_alloc();
   T* result=static_cast<T*>(raw);
   for(size_t i=0;i<bytes/sizeof(T);i++) {if(zero)new(result+i)T{};else new(result+i)T;}
   return result;
 }
};
inline Allocation allocate_raw(size_t n){return {n,false};}
inline Allocation allocate_zero(size_t n,size_t width){return {n*width,true};}
static double *const INI_HAPKE=nullptr,*const INI_SET_AO_ALBEDO=nullptr,*const INI_DIA_PARAMS=nullptr;
static constexpr int INI_IGNORE_AO_ALBEDO=0,INI_FIT_AO_ALBEDO=0;
static constexpr double INI_NDCHORD_WEIGHT=1000.0;
static constexpr int INI_LOGEXP=4;
// Signatures for mathematically unselected reflectance branches; compiler
// eliminates these constant-false blocks. They are not linked domain APIs.
double Albedo_Term(int*,double*,int,int,double*,double*,int,double*);
void dhapke_bright_fac(double*,double*,double,double,double*,double,double*,double*,double*);
void dhapke_bright(double*,double*,double,double,double*,double,double*,double*,double*);
void matrix_prod(double *A,int m,int k,double *B,int n,double *C){
 for(int i=0;i<m;i++)for(int j=0;j<n;j++){double value=0;for(int q=0;q<k;q++)value+=A[i*k+q]*B[q*n+j];C[i*n+j]=value;}
}
void matrix_vectorprod(double*A,int m,int n,double*x,double*y,int trans){
 for(int i=0;i<(trans?n:m);i++){double value=0;for(int j=0;j<(trans?m:n);j++)value+=(trans?A[j*n+i]:A[i*n+j])*x[j];y[i]=value;}
}

void AdjFacet(int *tlist,double *vlist,int nfac,int nvert,int *A);
void mult_with_cons(double *A,int m,int n,double C);
double sum_vector(double *vec1,int n);
void set_submatrix(double *A,int m0,int n0,double *B,int m1,int n1,int k,int l);
void  set_el(double *A,int m,int n,double a,int i,int j);
double  get_el(double *A,int m,int n,int i,int j);
void  set_elI(int *A,int m,int n,int a,int i,int j);
int  get_elI(int *A,int m,int n,int i,int j);
void Calculate_Area_and_Normal_Derivative(double *w1,double *w2,double *w3,double *n,double *dndx1,double *dndx2,double *dndx3,double *dndy1,double *dndy2,double *dndy3,double *dndz1,double *dndz2,double *dndz3,double *area,double *dAdx,double *dAdy,double *dAdz);
void Convert_to_Matrix(double *xr,double *yr,double *zr,double R[3][3]);
void mult_mat(double A[3][3],double B[3][3],double C[3][3]);
void transpose(double M[3][3],double Mt[3][3]);
void transpose2(double M[3][3],double *Mt);
void Calculate_Frame_Matrix(double *E,double *up,double R[3][3]);
void find_neighborhood(int *tlist,double *vlist,int nfac,int nvert,int *E,int *N,int *E2,int *A);
void cross(double *A,double *B,double *C);
void mult_vector(double A[3][3],double *y,double *x);
void zero_array(double *array,int count);
double minv(double *vlist,int nvert,int index);
double maxv(double *vlist,int nvert,int index);
void rotate(double beta,double lambda,double omega,double omega0,double t,double M[3][3],double dMb[3][3],double dMl[3][3],double dMo[3][3]);
void multmatrix(double A[3][3],double B[3][3],double C[3][3]);
void Calc_FTC(double *u1,double *v1,int nf,double a,double b,double c,double d,double g,double h,Complex *F);
void Calc_FTC_deriv(double *u1,double *v1,int nf,double a,double b,double c,double d,double g,double h,Complex *F,Complex *Fda,Complex *Fdb,Complex *Fdc,Complex *Fdd,Complex *Fdg,Complex *Fdh);
Complex FTC(double u,double v,double a,double b,double c,double d);
Complex FTC3(double u,double v,double a,double b,double c,double d);
Complex FTCda(double u,double v,double a,double b,double c,double d);
Complex FTCdb(double u,double v,double a,double b,double c,double d);
Complex FTCdc(double u,double v,double a,double b,double c,double d);
Complex FTCdd(double u,double v,double a,double b,double c,double d);
Complex FTC3da(double u,double v,double a,double b,double c,double d);
Complex FTC3db(double u,double v,double a,double b,double c,double d);
Complex FTC3dc(double u,double v,double a,double b,double c,double d);
Complex FTC3dd(double u,double v,double a,double b,double c,double d);
void FacetsOverHorizon(int *tlist,double *vlist,int nfac,int nvert,double *normal,double *centroid,int *NumofBlocks,int *IndexofBlocks);
void FindActualBlockers(int *tlist,double *vlist,int nfac,int nvert,double *E,double *E0,int nE,int *visible);
int is_in_triangle(double *point,double *dir,double *vert1,double *vert2,double *vert3);
double Calculate_AO(int *tlist,double *vlist,int nfac,int nvert,double *angles,double *Eo,double *E0o,double *up,double TIME,double dist,double *freqx,double *freqy,int nfreq,double *offset,double *Fr,double *Fi,double *A,double *Alimit);
void Calculate_AO_deriv(int *tlist,double *vlist,int nfac,int nvert,double *angles,double *Eo,double *E0o,double *up,double TIME,double dist,double *freqx,double *freqy,int nfreq,double *offset,double *Fr,double *Fi,double *dFdxr,double *dFdxi,double *dFdyr,double *dFdyi,double *dFdzr,double *dFdzi,double *dFdAr,double *dFdAi,double *dFdoffr,double *dFdoffi,double *A,double *Alimit,double *dAr,double *dAi);
void calculate_lcurve(int *tlist,double *vlist,int numfac,int numvert,double *angles,double *Eo,double *E0o,int nE,double *TIME,double *bright,double *dbrightx,double *dbrighty,double *dbrightz,double *dbrightb,double *dbrightl,double *dbrighto,double *dbrightp,double *A,double *Alimit,double *dA,int rel,double *params);
double phase_function(double *E,double *E0,double *params,double *dpdp);
void Fit_Occ(int *tlist,double *vlist,int nfac,int nvert,double *angles,double *up,double *E,double *V0,double *TIME,double *offset,double *chords,int *type,int nchords,double *W,double *Chordoffset,double *dist,double *dx,double *dy,double *dz,double *dangles,double *dtox,double *dtoy,double* dCOdoff);
void find_chord_dist(int *tlist,double *vlist,int nfac,int nvert,double *offset,double *p1,double *p2,double* dist2,int* vert2,double *dddx2,double *dddy2);
void find_chord_dist_simple(int *tlist,double *vlist,int nfac,int nvert,double *offset,double *p1,double *p2,double* dist2,int* vert2,double *dddx2,double *dddy2);
void find_chord(int *tlist,double *vlist2,int nfac,int nvert,double *offset,double *a,double *b,int *Adj,int *cledge,int *faedge,double *clpoint,double *fapoint,int *inters,double *dclpx,double *dclpy,double *dfapx,double *dfapy);
int line_intersect(double *a,double *b,double *x,double *y,double *p,double *dpx,double *dpy);
void convex_reg(int* tlist,double* vlist,int nfac,int nvert,double *D,int dm,int dn,double *res,double *drdv);
void dihedral_angle_reg(int *tlist,double *vlist,int nfac,int nvert,double *D,int dm,int dn,double *result,double *drsdv);
void dihedral_angle(int* tlist,double* vlist,int nfac,int nvert,double *res,int *EV,double *dresdx,double *dresdy,double *dresdz);
double center_of_mass(int *tlist,double *vlist,int nfac,int nvert,double *D,int dm,int dn,double *dv,int deriv);

void AdjFacet(int *tlist,double *vlist,int nfac,int nvert,int *A)
{
    
    int i1,i2,i3;
    for(int j=0;j<nfac;j++)
    {
        i1=tlist[3*j]-1;
        i2=tlist[3*j+1]-1;
        i3=tlist[3*j+2]-1;
        set_elI(A,nvert,nvert,j+1,i1,i2);
        set_elI(A,nvert,nvert,j+1,i2,i3);
        set_elI(A,nvert,nvert,j+1,i3,i1);
    }
}

void mult_with_cons(double *A,int m,int n,double C)
{
    for(int j=0;j<m*n;j++)
        A[j]=C*A[j];
}

double sum_vector(double *vec1,int n)
{
  double res=0;
  for(int j=0;j<n;j++)
    res=res+vec1[j];
  return res;
}

void set_submatrix(double *A,int m0,int n0,double *B,int m1,int n1,int k,int l)
{
    
    if(k+m1-1>=m0)
    {
        printf("Matrix size %d %d, submatrix %d %d, at %d %d\n",m0,n0,m1,n1,k,l);
        puts("Error: index too large in set_submatrix\n");
        exit(1);
    }
    if(l+n1-1>=n0)
    {
        printf("Matrix size %d %d, submatrix %d %d, at %d %d\n",m0,n0,m1,n1,k,l);
        puts("Error: index too large in set_submatrix\n");
        exit(1);
    }
    for(int i=0;i<m1;i++)
        for(int j=0;j<n1;j++)
            A[k*n0+l+j+i*n0]=B[j+i*n1];
}

void  set_el(double *A,int m,int n,double a,int i,int j)
{
    
    if(i>m-1 || j>n-1)
    {
        puts("Error: error in set_element, index too large. Bailing out!");
        exit(1);
    }
    A[i*n+j]=a;
}

double  get_el(double *A,int m,int n,int i,int j)
{
    
    if(i>m-1 || j>n-1)
    {
        puts("Error: error in get_element, index too large. Bailing out!");
        exit(1);
    }
    return A[i*n+j];
}

void  set_elI(int *A,int m,int n,int a,int i,int j)
{
    
    if(i>m-1 || j>n-1)
    {
        puts("Error: error in set_element, index too large. Bailing out!");
        exit(1);
    }
    A[i*n+j]=a;
}

int  get_elI(int *A,int m,int n,int i,int j)
{
    
    if(i>m-1 || j>n-1)
    {
        puts("Error: error in get_element, index too large. Bailing out!");
        exit(1);
    }
    return A[i*n+j];
}

void Calculate_Area_and_Normal_Derivative(double *w1,double *w2,double *w3,double *n,double *dndx1,double *dndx2,double *dndx3,double *dndy1,double *dndy2,double *dndy3,double *dndz1,double *dndz2,double *dndz3,double *area,double *dAdx,double *dAdy,double *dAdz)
{
  
  double x1,x2,x3,y1,y2,y3,z1,z2,z3;
  double v1[3],v2[3],cv[3];
  double normc;
  double dcx1[3],dcx2[3],dcx3[3],dcy1[3],dcy2[3],dcy3[3],dcz1[3],dcz2[3],dcz3[3];
  double *dareax,*dareay,*dareaz;
  dareax=dAdx;
  dareay=dAdy;
  dareaz=dAdz;
  for(int i=0;i<3;i++)
  {
    v1[i]=w2[i]-w1[i];
    v2[i]=w3[i]-w1[i];
  }
  cross(v1,v2,cv);
  normc=NORM(cv);
  area[0]=0.5*normc;
  n[0]=cv[0]/normc;
  n[1]=cv[1]/normc;
  n[2]=cv[2]/normc;
  x1=w1[0];
  y1=w1[1];
  z1=w1[2];
  x2=w2[0];
  y2=w2[1];
  z2=w2[2];
  x3=w3[0];
  y3=w3[1];
  z3=w3[2];
  dcx1[0]=0;
  dcx1[1]=z3-z2;
  dcx1[2]=y2-y3;
  
  dcx2[0]=0;
  dcx2[1]=z1-z3;
  dcx2[2]=y3-y1;
  
  dcx3[0]=0;
  dcx3[1]=z2-z1;
  dcx3[2]=y1-y2;
  
  dcy1[0]=z2-z3;
  dcy1[1]=0;
  dcy1[2]=x3-x2;
  
  dcy2[0]=z3-z1;
  dcy2[1]=0;
  dcy2[2]=x1-x3;
  
  dcy3[0]=z1-z2;
  dcy3[1]=0;
  dcy3[2]=x2-x1;
  
  dcz1[0]=y3-y2;
  dcz1[1]=x2-x3;
  dcz1[2]=0;
  
  dcz2[0]=y1-y3;
  dcz2[1]=x3-x1;
  dcz2[2]=0;
  
  dcz3[0]=y2-y1;
  dcz3[1]=x1-x2;
  dcz3[2]=0;
  
  dareax[0]=1/(2*normc)*(cv[0]*dcx1[0]+cv[1]*dcx1[1]+cv[2]*dcx1[2]);
  dareax[1]=1/(2*normc)*(cv[0]*dcx2[0]+cv[1]*dcx2[1]+cv[2]*dcx2[2]);
  dareax[2]=1/(2*normc)*(cv[0]*dcx3[0]+cv[1]*dcx3[1]+cv[2]*dcx3[2]);
  
  dareay[0]=1/(2*normc)*(cv[0]*dcy1[0]+cv[1]*dcy1[1]+cv[2]*dcy1[2]);
  dareay[1]=1/(2*normc)*(cv[0]*dcy2[0]+cv[1]*dcy2[1]+cv[2]*dcy2[2]);
  dareay[2]=1/(2*normc)*(cv[0]*dcy3[0]+cv[1]*dcy3[1]+cv[2]*dcy3[2]);
  
  dareaz[0]=1/(2*normc)*(cv[0]*dcz1[0]+cv[1]*dcz1[1]+cv[2]*dcz1[2]);
  dareaz[1]=1/(2*normc)*(cv[0]*dcz2[0]+cv[1]*dcz2[1]+cv[2]*dcz2[2]);
  dareaz[2]=1/(2*normc)*(cv[0]*dcz3[0]+cv[1]*dcz3[1]+cv[2]*dcz3[2]);
  
  for(int jk=0;jk<3;jk++)
  {
    dndx1[jk]=(normc*dcx1[jk]-cv[jk]*2*dareax[0])/(normc*normc);
    dndx2[jk]=(normc*dcx2[jk]-cv[jk]*2*dareax[1])/(normc*normc);
    dndx3[jk]=(normc*dcx3[jk]-cv[jk]*2*dareax[2])/(normc*normc);
    
    dndy1[jk]=(normc*dcy1[jk]-cv[jk]*2*dareay[0])/(normc*normc);
    dndy2[jk]=(normc*dcy2[jk]-cv[jk]*2*dareay[1])/(normc*normc);
    dndy3[jk]=(normc*dcy3[jk]-cv[jk]*2*dareay[2])/(normc*normc);
    
    dndz1[jk]=(normc*dcz1[jk]-cv[jk]*2*dareaz[0])/(normc*normc);
    dndz2[jk]=(normc*dcz2[jk]-cv[jk]*2*dareaz[1])/(normc*normc);
    dndz3[jk]=(normc*dcz3[jk]-cv[jk]*2*dareaz[2])/(normc*normc);
  }
}

void Convert_to_Matrix(double *xr,double *yr,double *zr,double R[3][3])
{
  R[0][0]=xr[0];
  R[0][1]=xr[1];
  R[0][2]=xr[2];
  R[1][0]=yr[0];
  R[1][1]=yr[1];
  R[1][2]=yr[2];
  R[2][0]=zr[0];
  R[2][1]=zr[1];
  R[2][2]=zr[2];
}

void mult_mat(double A[3][3],double B[3][3],double C[3][3])
{
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<3;j++)
    {
  C[i][j]=A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j];
    }
  }
}

void transpose(double M[3][3],double Mt[3][3])
{

for(int i=0;i<3;i++)
  for(int j=0;j<3;j++)
    Mt[i][j]=M[j][i];
}

void transpose2(double M[3][3],double *Mt)
{

for(int i=0;i<3;i++)
  for(int j=0;j<3;j++)
    Mt[i*3+j]=M[j][i];
}

void Calculate_Frame_Matrix(double *E,double *up,double R[3][3])
{
  
  
  
  
 double x[3],y[3],z[3];
 double nx,ny,nz;
 nz=NORM(E);
 z[0]=E[0]/nz;
 z[1]=E[1]/nz;
 z[2]=E[2]/nz;
 cross(up,z,x);
 nx=NORM(x);
 x[0]=x[0]/nx;
 x[1]=x[1]/nx;
 x[2]=x[2]/nx;
 cross(z,x,y);
 ny=NORM(y);
 y[0]=y[0]/ny;
 y[1]=y[1]/ny;
 y[2]=y[2]/ny;
 Convert_to_Matrix(x,y,z,R);
}

void find_neighborhood(int *tlist,double *vlist,int nfac,int nvert,int *E,int *N,int *E2,int *A)
{
    
    
    int i1,i2,i3;
    
    for(int j=0;j<nfac;j++)
    {
        i1=tlist[3*j]-1;
        i2=tlist[3*j+1]-1;
        i3=tlist[3*j+2]-1;
        E[i1*nvert+i2]=1;
        E[i2*nvert+i1]=1;
        
        E[i2*nvert+i3]=1;
        E[i3*nvert+i2]=1;
        
        E[i1*nvert+i3]=1;
        E[i3*nvert+i1]=1;
        
        N[i1*nfac+j]=1;
        N[i2*nfac+j]=1;
        N[i3*nfac+j]=1;
        set_elI(E2,nvert,nvert,j,i1,i2);
        set_elI(E2,nvert,nvert,j,i2,i3);
        set_elI(E2,nvert,nvert,j,i3,i1);
        
    }
    for(int j=0;j<nfac;j++)
    {
        i1=tlist[3*j]-1;
        i2=tlist[3*j+1]-1;
        i3=tlist[3*j+2]-1;
        A[j*nfac+E2[i2*nvert+i1]]=1;
        A[j*nfac+E2[i3*nvert+i2]]=1;
        A[j*nfac+E2[i1*nvert+i3]]=1;
    }
}

void cross(double *A,double *B,double *C)
{
    
  C[0]=A[1]*B[2]-A[2]*B[1];
  C[1]=A[2]*B[0]-A[0]*B[2];
  C[2]=A[0]*B[1]-A[1]*B[0];
}

void mult_vector(double A[3][3],double *y,double *x)
{
  x[0]=A[0][0]*y[0]+A[0][1]*y[1]+A[0][2]*y[2];
  x[1]=A[1][0]*y[0]+A[1][1]*y[1]+A[1][2]*y[2];
  x[2]=A[2][0]*y[0]+A[2][1]*y[1]+A[2][2]*y[2];
}

void zero_array(double *array,int count)
{
    memset(array,0,sizeof(double)*count);
}

double minv(double *vlist,int nvert,int index)
{
    double min=1E15;
    for(int j=0;j<nvert;j++)
    {
        if(vlist[j*3+index]<min)
            min=vlist[j*3+index];
    }
    return min;
}

double maxv(double *vlist,int nvert,int index)
{
    double max=-1E15;
    for(int j=0;j<nvert;j++)
    {
        if(vlist[j*3+index]>max)
            max=vlist[j*3+index];
    }
    return max;
}

void rotate(double beta,double lambda,double omega,double omega0,double t,double M[3][3],double dMb[3][3],double dMl[3][3],double dMo[3][3])
{
  double f,cf,sf,cb,sb,cl,sl;
  double fmat[3][3],dfm[3][3],blmat[3][3],Dblmat_bet[3][3],Dblmat_lam[3][3];
  f=omega*t+omega0;
  cf=cos(f);
  sf=sin(f);
  cb=cos(beta);
  sb=sin(beta);
  cl=cos(lambda);
  sl=sin(lambda);
  fmat[0][0]=cf;
  fmat[0][1]=sf;
  fmat[0][2]=0;
  fmat[1][0]=-sf;
  fmat[1][1]=cf;
  fmat[1][2]=0;
  fmat[2][0]=0;
  fmat[2][1]=0;
  fmat[2][2]=1;
  
  dfm[0][0]=-t*sf;
  dfm[0][1]=t*cf;
  dfm[0][2]=0;
  dfm[1][0]=-t*cf;
  dfm[1][1]=-t*sf;
  dfm[1][2]=0;
  dfm[2][0]=0;
  dfm[2][1]=0;
  dfm[2][2]=0;
  
  blmat[0][0]=cb*cl;
  blmat[0][1]=cb*sl;
  blmat[0][2]=-sb;
  blmat[1][0]=-sl;
  blmat[1][1]=cl;
  blmat[1][2]=0;
  blmat[2][0]=sb*cl;
  blmat[2][1]=sb*sl;
  blmat[2][2]=cb;
  
  Dblmat_bet[0][0]=-sb*cl;
  Dblmat_bet[0][1]=-sb*sl;
  Dblmat_bet[0][2]=-cb;
  Dblmat_bet[1][0]=0;
  Dblmat_bet[1][1]=0;
  Dblmat_bet[1][2]=0;
  Dblmat_bet[2][0]=cb*cl;
  Dblmat_bet[2][1]=cb*sl;
  Dblmat_bet[2][2]=-sb;
  
  Dblmat_lam[0][0]=-cb*sl;
  Dblmat_lam[0][1]=cb*cl;
  Dblmat_lam[0][2]=0;
  Dblmat_lam[1][0]=-cl;
  Dblmat_lam[1][1]=-sl;
  Dblmat_lam[1][2]=0;
  Dblmat_lam[2][0]=-sb*sl;
  Dblmat_lam[2][1]=sb*cl;
  Dblmat_lam[2][2]=0;
  
  multmatrix(fmat,blmat,M);
  multmatrix(fmat,Dblmat_bet,dMb);
  multmatrix(fmat,Dblmat_lam,dMl);
  multmatrix(dfm,blmat,dMo);
}

void multmatrix(double A[3][3],double B[3][3],double C[3][3])
{
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<3;j++)
    {
  C[i][j]=A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j];
    }
  }
}

void Calc_FTC(double *u1,double *v1,int nf,double a,double b,double c,double d,double g,double h,Complex *F)
{
  
  
 
  double u,v;
  double eps=1e-14;
 
  
  
    
  
  
 
  for(int j=0;j<nf;j++)
  {
   u=u1[j];
   v=v1[j];
  
   
   if(fabs((c-a)*u+(d-b)*v)>eps)
     F[j]=FTC(u,v,a,b,c,d);
  
   else
     F[j]=FTC3(u,v,a,b,c,d);
   if(fabs((g-c)*u+(h-d)*v)>eps)
   
     F[j]=F[j]+FTC(u,v,c,d,g,h);
   
   else
     F[j]=F[j]+FTC3(u,v,c,d,g,h);
   if(fabs((a-g)*u+(b-h)*v)>eps)
   
     F[j]=F[j]+FTC(u,v,g,h,a,b);
   
   else
     F[j]=F[j]+FTC3(u,v,g,h,a,b);
  }


  
}

void Calc_FTC_deriv(double *u1,double *v1,int nf,double a,double b,double c,double d,double g,double h,Complex *F,Complex *Fda,Complex *Fdb,Complex *Fdc,Complex *Fdd,Complex *Fdg,Complex *Fdh)
{
  
  
  
  double eps=1e-14;
  double u,v;
  
  
  
  
  
  
  
  
  
 
   
    
   
    
    
     
     for(int j=0;j<nf;j++)
     {
       u=u1[j];
       v=v1[j];
      

if(fabs((c-a)*u+(d-b)*v)>eps)
{
F[j]=FTC(u,v,a,b,c,d);
Fda[j]=FTCda(u,v,a,b,c,d);
Fdb[j]=FTCdb(u,v,a,b,c,d);
Fdc[j]=FTCdc(u,v,a,b,c,d);
Fdd[j]=FTCdd(u,v,a,b,c,d);
Fdg[j]=0;
Fdh[j]=0;

}

else
{
F[j]=FTC3(u,v,a,b,c,d);
Fda[j]=FTC3da(u,v,a,b,c,d);
Fdb[j]=FTC3db(u,v,a,b,c,d);
Fdc[j]=FTC3dc(u,v,a,b,c,d);
Fdd[j]=FTC3dd(u,v,a,b,c,d);
Fdg[j]=0;
Fdh[j]=0;
}
if(fabs((g-c)*u+(h-d)*v)>eps)
{
F[j]=F[j]+FTC(u,v,c,d,g,h);
Fdc[j]=Fdc[j]+FTCda(u,v,c,d,g,h);
Fdd[j]=Fdd[j]+FTCdb(u,v,c,d,g,h);
Fdg[j]=FTCdc(u,v,c,d,g,h);
Fdh[j]=FTCdd(u,v,c,d,g,h);

}

else
{
F[j]=F[j]+FTC3(u,v,c,d,g,h);
Fdc[j]=Fdc[j]+FTC3da(u,v,c,d,g,h);
Fdd[j]=Fdd[j]+FTC3db(u,v,c,d,g,h);
Fdg[j]=FTC3dc(u,v,c,d,g,h);
Fdh[j]=FTC3dd(u,v,c,d,g,h);
}
if(fabs((a-g)*u+(b-h)*v)>eps)
{
F[j]=F[j]+FTC(u,v,g,h,a,b);
Fda[j]=Fda[j]+FTCdc(u,v,g,h,a,b);
Fdb[j]=Fdb[j]+FTCdd(u,v,g,h,a,b);
Fdg[j]=Fdg[j]+FTCda(u,v,g,h,a,b);
Fdh[j]=Fdh[j]+FTCdb(u,v,g,h,a,b);
}

else
{
F[j]=F[j]+FTC3(u,v,g,h,a,b);
Fda[j]=Fda[j]+FTC3dc(u,v,g,h,a,b);
Fdb[j]=Fdb[j]+FTC3dd(u,v,g,h,a,b);
Fdg[j]=Fdg[j]+FTC3da(u,v,g,h,a,b);
Fdh[j]=Fdh[j]+FTC3db(u,v,g,h,a,b);
}

}

  
}

Complex FTC(double u,double v,double a,double b,double c,double d)
 {
  Complex Eab=std::exp(-2.0*PI*Complex(0.0,1.0)*(a*u+b*v));
  Complex Ecd=std::exp(-2.0*PI*Complex(0.0,1.0)*(c*u+d*v));
  double u2v2=1.0/(pow(u,2)+pow(v,2));
  
  return  1/(PI2*4.0)*((b-d)*u-(a-c)*v)/((a-c)*u+(b-d)*v)*u2v2*(Eab-Ecd);
 }

Complex FTC3(double u,double v,double a,double b,double c,double d)
 {
   double u2v2=1.0/(pow(u,2)+pow(v,2));
   return -Complex(0.0,1.0)/(2*PI)*((b-d)*u-(a-c)*v)*u2v2*std::exp(-2.0*PI*Complex(0.0,1.0)*(a*u+b*v));
 }

Complex FTCda(double u,double v,double a,double b,double c,double d)
 {
   Complex Eab=std::exp(-2.0*PI*Complex(0.0,1.0)*(a*u+b*v));
   Complex Ecd=std::exp(-2.0*PI*Complex(0.0,1.0)*(c*u+d*v));
   
   double  cadb=(c-a)*u+(d-b)*v;
   double u2v2=1.0/(pow(u,2)+pow(v,2));
   double  H=-((b-d)*u-(a-c)*v)/cadb*u2v2;
   
   return 1/(4*PI2)*((d-b)/pow(cadb,2)*(Eab-Ecd)+H*(-2.0*PI*Complex(0.0,1.0)*u*Eab));
 }

Complex FTCdb(double u,double v,double a,double b,double c,double d)
 {
   Complex Eab=std::exp(-2.0*PI*Complex(0.0,1.0)*(a*u+b*v));
   Complex Ecd=std::exp(-2.0*PI*Complex(0.0,1.0)*(c*u+d*v)); 
   double cadb=(c-a)*u+(d-b)*v;
   double u2v2=1.0/(pow(u,2)+pow(v,2));
   Complex H=-((b-d)*u-(a-c)*v)/cadb*u2v2;
   return 1/(4*PI2)*((a-c)/pow(cadb,2)*(Eab-Ecd)+H*(-2.0*PI*Complex(0.0,1.0)*v*Eab));
 }

Complex FTCdc(double u,double v,double a,double b,double c,double d)
 {
   Complex Eab=std::exp(-2.0*PI*Complex(0.0,1.0)*(a*u+b*v));
   Complex Ecd=std::exp(-2.0*PI*Complex(0.0,1.0)*(c*u+d*v)); 
   double cadb=(c-a)*u+(d-b)*v;
   double u2v2=1.0/(pow(u,2)+pow(v,2));
   Complex H=-((b-d)*u-(a-c)*v)/cadb*u2v2;
   return 1/(4*PI2)*((b-d)/pow(cadb,2)*(Eab-Ecd)+H*(2.0*PI*Complex(0.0,1.0)*u*Ecd));
 }

Complex FTCdd(double u,double v,double a,double b,double c,double d)
 {
   Complex Eab=std::exp(-2.0*PI*Complex(0.0,1.0)*(a*u+b*v));
   Complex Ecd=std::exp(-2.0*PI*Complex(0.0,1.0)*(c*u+d*v));
   double cadb=(c-a)*u+(d-b)*v;
   double u2v2=1.0/(pow(u,2)+pow(v,2));
   Complex H=-((b-d)*u-(a-c)*v)/cadb*u2v2;
    return 1/(4*PI2)*((c-a)/pow(cadb,2)*(Eab-Ecd)+H*(2.0*PI*Complex(0.0,1.0)*v*Ecd));
 }

Complex FTC3da(double u,double v,double a,double b,double c,double d)
 {
   return Complex(0.0,1.0)/(2*PI)*v*1.0/(pow(u,2)+pow(v,2))*std::exp(-2*PI*Complex(0.0,1.0)*(c*u+d*v));
 }

Complex FTC3db(double u,double v,double a,double b,double c,double d)
 {

   return -Complex(0.0,1.0)/(2*PI)*u*1.0/(pow(u,2)+pow(v,2))*std::exp(-2*PI*Complex(0.0,1.0)*(c*u+d*v));
 }

Complex FTC3dc(double u,double v,double a,double b,double c,double d)
 {
   double u2v2=1.0/(pow(u,2)+pow(v,2));
   return ((d-b)*pow(u,2)*u2v2+(-Complex(0.0,1.0)+2*(a-c)*PI*u)*v*1.0/(2*PI)*u2v2)*std::exp(-2*PI*Complex(0.0,1.0)*(c*u+d*v));
 }

Complex FTC3dd(double u,double v,double a,double b,double c,double d)
 { double u2v2=1.0/(pow(u,2)+pow(v,2));
   return ((a-c)*pow(v,2)*u2v2+(Complex(0.0,1.0)+2*(d-b)*PI*v)*u*1.0/(2*PI)*u2v2)*std::exp(-2*PI*Complex(0.0,1.0)*(c*u+d*v));
 }

void FacetsOverHorizon(int *tlist,double *vlist,int nfac,int nvert,double *normal,double *centroid,int *NumofBlocks,int *IndexofBlocks)
{
  
  
  
 
 
  int j1,j2,j3;
  int vindex;
  double *v1,*v2,*v3;
  double*w;
  double side1[3],side2[3];
  double cvvec[3];
  double norm;
  double cnormal[3];
  double *jcent,*jnormal,*knormal;
  double cent[3];
 
  
  memset(NumofBlocks,0,sizeof(int)*nfac);
  memset(IndexofBlocks,0,sizeof(int)*nfac*nfac);
  
  for(int j=0;j<nfac;j++)
  {
    
    
    j1=tlist[j*3]-1;
    j2=tlist[j*3+1]-1;
    j3=tlist[j*3+2]-1;
    
    v1=vlist+j1*3;
    v2=vlist+j2*3;
    v3=vlist+j3*3;
    
    for(int i=0;i<3;i++)
    {
      side1[i]=v2[i]-v1[i];
      side2[i]=v3[i]-v1[i];
      cent[i]=(v1[i]+v2[i]+v3[i])/3.0;
    }
    cross(side1,side2,cnormal);
    norm=NORM(cnormal);
    
    
    for(int i=0;i<3;i++)
    {
      cnormal[i]=cnormal[i]/norm;
      normal[3*j+i]=cnormal[i];
      centroid[3*j+i]=cent[i];
    }
  }
  for(int j=0;j<nfac;j++)
  {
    
    
    j1=tlist[j*3]-1;
    j2=tlist[j*3+1]-1;
    j3=tlist[j*3+2]-1;
    
    v1=vlist+j1*3;
    v2=vlist+j2*3;
    v3=vlist+j3*3;
    jcent=centroid+3*j;
    jnormal=normal+3*j;
    
    for(int k=0;k<nfac;k++)
    {
      if(k==j)
	continue;
      
     
      
      
      
      
      knormal=normal+3*k;
      
      for(int i=0;i<3;i++)
      {
	vindex=tlist[k*3+i]-1;
	
	if((vindex==j1) || (vindex==j2) ||( vindex==j3))
	  continue;
	w=vlist+vindex*3; 
	cvvec[0]=w[0]-jcent[0];
	cvvec[1]=w[1]-jcent[1];
	cvvec[2]=w[2]-jcent[2];
	
	if((DOT(cvvec,jnormal))>0 && (DOT(cvvec,knormal)<0)) 
	{
	  IndexofBlocks[nfac*j+(NumofBlocks[j]++)]=k+1;
	  
	  break;
	}
      }
    }
  }
}

void FindActualBlockers(int *tlist,double *vlist,int nfac,int nvert,double *E,double *E0,int nE,int *visible)
{
  
  double *normal,*centroid;
  double *cnormal,*ccent;
  double *bnormal;
  double mu,mu0,bmu,bmu0;
  double *cE,*cE0;
  int i1,i2,i3;
  double *v1,*v2,*v3;
  int blockindex;
  int *IndexofBlocks;
  int *NumofBlocks;
  int blocked=0;
  normal=allocate_zero(nfac*3,sizeof(double));
  centroid=allocate_zero(nfac*3,sizeof(double));
  NumofBlocks=allocate_zero(nfac,sizeof(int));
  IndexofBlocks=allocate_zero(nfac*nfac,sizeof(int));
  
  FacetsOverHorizon(tlist,vlist,nfac,nvert,normal,centroid,NumofBlocks,IndexofBlocks);
  
  for(int j=0;j<nE;j++)
  {
    cE=E+j*3;
    cE0=E0+j*3;
    
    for(int k=0;k<nfac;k++)
    {
      blocked=0;
      
      cnormal=normal+k*3;
      ccent=centroid+k*3;
      
      mu=DOT(cE,cnormal);
      mu0=DOT(cE0,cnormal);
      
	
      if(mu<0 || mu0<0)
	continue;
      
      if(NumofBlocks[k]==0)
      {
	visible[nfac*j+k]=1;
	continue;
      }
      
      
      for(int l=0;l<NumofBlocks[k];l++)
      {
	blockindex=IndexofBlocks[k*nfac+l]-1;
	
	  
	  
	  bnormal=normal+3*blockindex;
	  bmu=DOT(cE,bnormal);
	  bmu0=DOT(cE0,bnormal);
	  
	 if(bmu>0 && bmu0>0)
	    continue;
	  
	  i1=tlist[3*blockindex]-1;
	  i2=tlist[3*blockindex+1]-1;
	  i3=tlist[3*blockindex+2]-1;
	  v1=vlist+3*i1;
	  v2=vlist+3*i2;
	  v3=vlist+3*i3;
	  
	  
	 
	  
	  if(bmu<0 && is_in_triangle(ccent,cE,v1,v2,v3))
	  {
	   
	    
	    blocked=1;
	    break;
	  }
	  
	  if(bmu0<0 && is_in_triangle(ccent,cE0,v1,v2,v3))
	  {
	    blocked=1;
	    break;
	  }
	
      }
      
      
      if(blocked==0)
      {
	
	visible[nfac*j+k]=1;
      }
    }
  }
  free(normal);
  free(centroid);
  free(NumofBlocks);
  free(IndexofBlocks);
}

int is_in_triangle(double *point,double *dir,double *vert1,double *vert2,double *vert3)
{
  double s1[3];
  double s2[3];
  double b[3];
  double A[3][3];
  double det,gamma,beta;
  
  for(int i=0;i<3;i++) {
    s1[i]=vert1[i]-vert2[i];
    s2[i]=vert1[i]-vert3[i];
    b[i]=vert1[i]-point[i];
  }
  det=s1[0]*s2[1]*dir[2]-s1[0]*dir[1]*s2[2]+s2[0]*dir[1]*s1[2]-s2[0]*s1[1]*dir[2]+dir[0]*s1[1]*s2[2]-dir[0]*s2[1]*s1[2];
  if(fabs(det)<0.000001)
    return 0;
  gamma=(dir[2]*(s1[0]*b[1]-b[0]*s1[1])+dir[1]*(b[0]*s1[2]-s1[0]*b[2])+dir[0]*(s1[1]*b[2]-b[1]*s1[2]))/det;
  if(gamma<0||gamma>1)
    return 0;
  
  beta=(b[0]*(s2[1]*dir[2]-dir[1]*s2[2])+b[1]*(dir[0]*s2[2]-s2[0]*dir[2])+b[2]*(s2[0]*dir[1]-s2[1]*dir[0]))/det;
  if(beta<0||beta>1-gamma)
    return 0;
  return 1;
}

double Calculate_AO(int *tlist,double *vlist,int nfac,int nvert,double *angles,double *Eo,double *E0o,double *up,double TIME,double dist,double *freqx,double *freqy,int nfreq,double *offset,double *Fr,double *Fi,double *A,double *Alimit)
{

 Complex *F0;

 F0=allocate_zero(nfreq,sizeof(Complex));
 double AOalb=1;
 double dt1,dt2;
 Complex *F=allocate_zero(nfreq,sizeof(Complex));
 double M[3][3],dMb[3][3],dMo[3][3],dMl[3][3],Mt[3][3];
 double R[3][3],Rdb[3][3],Rdl[3][3],Rdo[3][3],RT[3][3];
 double E[3],E0[3];
 double normalr[3],side1[3],side2[3];
 double dechdx[3],dechdy[3],dechdz[3],dechdA[3];
 double dAlb3[3];
 double n[3],*nb,*cent;
 double *vb1,*vb2,*vb3;
 double vr1[3],vr2[3],vr3[3];
 double *v1,*v2,*v3;
 double scale;
 Complex tscale,FTC;
 double dp;
 double B,TB=0;
 double norm;
 int t1,t2,t3,blocker,sign;
int j1,j2,j3;
 double mu,mu0,area,mub,ech,rexp;
 double *normal,*centroid;
 int *visible;
 int tb1,tb2,tb3; 
 int blocked=0;
 
  double la=0;
 double ha=1; 
 double alb_term=0;
 double alb=0;
 int albedo=0;
 if(false)
 {
     albedo=1;
     la=Alimit[0];
     ha=Alimit[1];
 }
 
 dp=1/(dist*149597871.0)*180.0/PI*3600.0;
 visible=allocate_zero(nfac,sizeof(int));
  

 
 

  
  Calculate_Frame_Matrix(Eo,up,R);
 
 
 
 rotate(angles[0],angles[1],angles[2],angles[3],TIME,M,dMb,dMl,dMo);
 
 
 transpose(M,Mt); 
 mult_mat(R,Mt,RT); 
 mult_vector(M,Eo,E);
 mult_vector(M,E0o,E0);

 

FindActualBlockers(tlist,vlist,nfac,nvert,E,E0,1,visible);
 


 for(int j=0;j<nfac;j++)
 {
   if(visible[j]==0)
     continue;
  
   
   
   if(INI_SET_AO_ALBEDO!=NULL)
       AOalb=INI_SET_AO_ALBEDO[j];
   j1=tlist[j*3]-1;
   j2=tlist[j*3+1]-1;
   j3=tlist[j*3+2]-1;
   
   
   v1=vlist+j1*3;
   v2=vlist+j2*3;
   v3=vlist+j3*3;
   
   
   
   for(int i=0;i<3;i++)
   {
     
     side1[i]=dp*(v2[i]-v1[i]); 
     side2[i]=dp*(v3[i]-v1[i]);
    
   }
   cross(side1,side2,n);
   norm=NORM(n);
   n[0]=n[0]/norm;
   n[1]=n[1]/norm;
   n[2]=n[2]/norm;
   
   mu=DOT(E,n);
   mu0=DOT(E0,n);
  
   mult_vector(RT,v1,vr1);
   mult_vector(RT,v2,vr2);
   mult_vector(RT,v3,vr3);
   for(int i=0;i<3;i++)
   {
     vr1[i]=dp*vr1[i];
     vr2[i]=dp*vr2[i];
     vr3[i]=dp*vr3[i];
   }
    
     
     Calc_FTC(freqx,freqy,nfreq,vr1[0],vr1[1],vr2[0],vr2[1],vr3[0],vr3[1],F0);
     
     
     
     
   area=0.5*norm;

 
    
   if(albedo==0)
        {
            alb=0;
            alb_term=1;
        }
        else
        {
            
            
            alb_term=Albedo_Term(tlist,vlist,nfac,nvert,Alimit,A,j,dAlb3);
        }
    if(INI_HAPKE!=NULL)
        dhapke_bright_fac(E,E0,mu,mu0,INI_HAPKE,INI_HAPKE[4],&B,&dt2,&dt1);
    else
        B=mu0*(1.0/(mu+mu0)+0.1); 
     for(int jf=0;jf<nfreq;jf++)
     {
      
       F[jf]+=alb_term*AOalb*B*F0[jf];
      
       
      }
      if(INI_HAPKE!=NULL)
          TB=TB+alb_term*AOalb*B*mu*area;
      else
        TB=TB+alb_term*AOalb*B*area*mu;
    

}

for(int j=0;j<nfreq;j++)
{
 
  Fr[j]=std::real(std::exp(2.0*PI*Complex(0.0,1.0)*(offset[0]*freqx[j]+offset[1]*freqy[j]))*F[j]/TB);
  Fi[j]=std::imag(std::exp(2.0*PI*Complex(0.0,1.0)*(offset[0]*freqx[j]+offset[1]*freqy[j]))*F[j]/TB);
  
}
free(visible);
free(F0);
free(F);
return TB;
}

void Calculate_AO_deriv(int *tlist,double *vlist,int nfac,int nvert,double *angles,double *Eo,double *E0o,double *up,double TIME,double dist,double *freqx,double *freqy,int nfreq,double *offset,double *Fr,double *Fi,double *dFdxr,double *dFdxi,double *dFdyr,double *dFdyi,double *dFdzr,double *dFdzi,double *dFdAr,double *dFdAi,double *dFdoffr,double *dFdoffi,double *A,double *Alimit,double *dAr,double *dAi)
{
 
  Complex *F=allocate_zero(nfreq,sizeof(Complex));
 Complex *F0,*FTda,*FTdb,*FTdc,*FTdd,*FTdh,*FTdg,*FTdx,*FTdy,*FTdz,*FTdA;
 FTdx=(Complex*)allocate_zero(nfreq*nvert,sizeof(Complex));
 FTdy=(Complex*)allocate_zero(nfreq*nvert,sizeof(Complex));
 FTdz=(Complex*)allocate_zero(nfreq*nvert,sizeof(Complex));
 FTdA=(Complex*)allocate_zero(nfreq*3,sizeof(Complex));
 F0=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 FTda=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 FTdb=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 FTdc=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 FTdd=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 FTdh=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 FTdg=(Complex*)allocate_zero(nfreq,sizeof(Complex));
 
 double M[3][3],dMb[3][3],dMo[3][3],dMl[3][3],Mt[3][3],dMbT[3][3],dMoT[3][3],dMlT[3][3]; 
 double R[3][3],Rdb[3][3],Rdl[3][3],Rdo[3][3],RT[3][3]; 
 double dEdb[3],dEdl[3],dEdo[3],dE0db[3],dE0dl[3],dE0do[3];
 double dndx1[3],dndx2[3],dndx3[3],dndy1[3],dndy2[3],dndy3[3],dndz1[3],dndz2[3],dndz3[3]; 
 double dBdx1,dBdy1,dBdz1,dBdx2,dBdy2,dBdz2,dBdx3,dBdy3,dBdz3,dBdb,dBdl,dBdo; 
 double *dTBdx,*dTBdy,*dTBdz; 
 double dTBdA[3]={0.0,0.0,0.0};
 double dAlb3[3];
 dTBdx=allocate_zero(nvert,sizeof(double));
 dTBdy=allocate_zero(nvert,sizeof(double));
 dTBdz=allocate_zero(nvert,sizeof(double));
 double dmudx1,dmudy1,dmudz1,dmudx2,dmudy2,dmudz2,dmudx3,dmudy3,dmudz3;
 double dmu0dx1,dmu0dy1,dmu0dz1,dmu0dx2,dmu0dy2,dmu0dz2,dmu0dx3,dmu0dy3,dmu0dz3;
 double dmudl,dmudb,dmudo,dmu0dl,dmu0db,dmu0do; 
 double dAdx[3],dAdy[3],dAdz[3]; 
 double dadx,dady,dadz,dbdx,dbdy,dbdz; 
 double E[3],E0[3]; 
 double side1[3],side2[3];
 double n[3];
 double v1db[3],v2db[3],v3db[3],v1dl[3],v2dl[3],v3dl[3],v1do[3],v2do[3],v3do[3]; 
 double vr1[3],vr2[3],vr3[3];
 double v1[3],v2[3],v3[3];
 Complex scale;

 double alb=0;
  double *dSdalb=NULL;
  double *dAlb=NULL;
  double *dAlbTB=NULL;
  int albedo=0;
  double la,ha,alb_term=0;
  double albexp=0;
 if(false)
 {
     albedo=1;
     
     la=Alimit[0];
     ha=Alimit[1];
     if(INI_FIT_AO_ALBEDO==1)
     {
     dAlb=allocate_zero(nfreq*nvert,sizeof(Complex));
     dAlbTB=allocate_zero(nvert,sizeof(double));
     }
 }
 
 
 double dp;
 double B,TB=0.0;
 double norm;
 double mut,mu0t;
 int t1,t2,t3;
int j1,j2,j3;
 double mu,mu0,area;
 double *normal;
 int *visible;
 int tb1,tb2,tb3; 
 int blocked=0;
 
 dp=1/(dist*149597871.0)*180.0/PI*3600.0;
 visible=(int*)allocate_zero(nfac,sizeof(int));
 
  Calculate_Frame_Matrix(Eo,up,R);
  
  
  
 
 
 
 
 rotate(angles[0],angles[1],angles[2],angles[3],TIME,M,dMb,dMl,dMo);
 
 
 transpose(M,Mt); 
 transpose(dMb,dMbT);
 transpose(dMl,dMlT);
 transpose(dMo,dMoT);
 mult_mat(R,Mt,RT); 
 mult_vector(M,Eo,E);
 mult_vector(M,E0o,E0);
 
 mult_mat(R,dMbT,Rdb);
 mult_mat(R,dMlT,Rdl);
 mult_mat(R,dMoT,Rdo);

 mult_vector(dMb,Eo,dEdb);
 mult_vector(dMl,Eo,dEdl);
 mult_vector(dMo,Eo,dEdo);
 mult_vector(dMb,E0o,dE0db);
 mult_vector(dMl,E0o,dE0dl);
 mult_vector(dMo,E0o,dE0do);
 dadx=RT[0][0];
  dady=RT[0][1];
  dadz=RT[0][2];
  dbdx=RT[1][0];
  dbdy=RT[1][1];
  dbdz=RT[1][2];
 

FindActualBlockers(tlist,vlist,nfac,nvert,E,E0,1,visible);
 



 
for(int j=0;j<nfac;j++) 
{
   if(visible[j]==0)
     continue;
  
   
   
   j1=tlist[j*3]-1;
   j2=tlist[j*3+1]-1;
   j3=tlist[j*3+2]-1;
   
  
   
   for(int i=0;i<3;i++)
   {
   v1[i]=*(vlist+j1*3+i)*dp; 
   v2[i]=*(vlist+j2*3+i)*dp;
   v3[i]=*(vlist+j3*3+i)*dp;
   }
   
   if(albedo==0)
        {
            alb=0;
            alb_term=1;
        }
        else
        {
           
           
            alb_term=Albedo_Term(tlist,vlist,nfac,nvert,Alimit,A,j,dAlb3);
            
        }
    
    
    Calculate_Area_and_Normal_Derivative(v1,v2,v3,n,dndx1,dndx2,dndx3,dndy1,dndy2,dndy3,dndz1,dndz2,dndz3,&area,dAdx,dAdy,dAdz);
    
   
   
   mu=DOT(E,n);
   mu0=DOT(E0,n);
  
   mult_vector(RT,v1,vr1);
   mult_vector(RT,v2,vr2);
   mult_vector(RT,v3,vr3);
  
    
     
  
     Calc_FTC_deriv(freqx,freqy,nfreq,vr1[0],vr1[1],vr2[0],vr2[1],vr3[0],vr3[1],F0,FTda,FTdb,FTdc,FTdd,FTdg,FTdh);
  
    
     mult_vector(Rdb,v1,v1db);
     mult_vector(Rdb,v2,v2db);
     mult_vector(Rdb,v3,v3db);
     mult_vector(Rdl,v1,v1dl);
     mult_vector(Rdl,v2,v2dl);
     mult_vector(Rdl,v3,v3dl);
     mult_vector(Rdo,v1,v1do);
     mult_vector(Rdo,v2,v2do);
     mult_vector(Rdo,v3,v3do);
     
     dmudx1=DOT(E,dndx1);
     dmudx2=DOT(E,dndx2);
     dmudx3=DOT(E,dndx3);
     dmudy1=DOT(E,dndy1);
     dmudy2=DOT(E,dndy2);
     dmudy3=DOT(E,dndy3);
     dmudz1=DOT(E,dndz1);
     dmudz2=DOT(E,dndz2);
     dmudz3=DOT(E,dndz3);
     dmudb=DOT(dEdb,n);
     dmudl=DOT(dEdl,n);
     dmudo=DOT(dEdo,n);
     dmu0dx1=DOT(E0,dndx1);
     dmu0dx2=DOT(E0,dndx2);
     dmu0dx3=DOT(E0,dndx3);
     dmu0dy1=DOT(E0,dndy1);
     dmu0dy2=DOT(E0,dndy2);
     dmu0dy3=DOT(E0,dndy3);
     dmu0dz1=DOT(E0,dndz1);
     dmu0dz2=DOT(E0,dndz2);
     dmu0dz3=DOT(E0,dndz3);
     dmu0db=DOT(dE0db,n);
     dmu0dl=DOT(dE0dl,n);
     dmu0do=DOT(dE0do,n);
   if(INI_HAPKE!=NULL)
        dhapke_bright_fac(E,E0,mu,mu0,INI_HAPKE,INI_HAPKE[4],&B,&mut,&mu0t);
    else
    {
        B=mu0*(1.0/(mu+mu0)+0.1); 
        mu0t=(mu/pow(mu+mu0,2)+0.1);
        mut=-mu0/pow(mu+mu0,2);
    }
   
   
   dBdx1=mu0t*dmu0dx1+mut*dmudx1;
   dBdx2=mu0t*dmu0dx2+mut*dmudx2;
   dBdx3=mu0t*dmu0dx3+mut*dmudx3;
   
   dBdy1=mu0t*dmu0dy1+mut*dmudy1;
   dBdy2=mu0t*dmu0dy2+mut*dmudy2;
   dBdy3=mu0t*dmu0dy3+mut*dmudy3;
   
   dBdz1=mu0t*dmu0dz1+mut*dmudz1;
   dBdz2=mu0t*dmu0dz2+mut*dmudz2;
   dBdz3=mu0t*dmu0dz3+mut*dmudz3;
   dBdb=mu0t*dmu0db+mut*dmudb;
   dBdl=mu0t*dmu0dl+mut*dmudl;
   dBdo=mu0t*dmu0do+mut*dmudo;
   
   
       dTBdx[j1]+=dBdx1*area*mu+B*dAdx[0]*mu+B*area*dmudx1;
       dTBdx[j2]+=dBdx2*area*mu+B*dAdx[1]*mu+B*area*dmudx2;
       dTBdx[j3]+=dBdx3*area*mu+B*dAdx[2]*mu+B*area*dmudx3;
       dTBdy[j1]+=dBdy1*area*mu+B*dAdy[0]*mu+B*area*dmudy1;
       dTBdy[j2]+=dBdy2*area*mu+B*dAdy[1]*mu+B*area*dmudy2;
       dTBdy[j3]+=dBdy3*area*mu+B*dAdy[2]*mu+B*area*dmudy3;
       dTBdz[j1]+=dBdz1*area*mu+B*dAdz[0]*mu+B*area*dmudz1;
       dTBdz[j2]+=dBdz2*area*mu+B*dAdz[1]*mu+B*area*dmudz2;
       dTBdz[j3]+=dBdz3*area*mu+B*dAdz[2]*mu+B*area*dmudz3;
       dTBdA[0]+=dBdb*area*mu+B*area*dmudb;
       dTBdA[1]+=dBdl*area*mu+B*area*dmudl;
       dTBdA[2]+=dBdo*area*mu+B*area*dmudo;
       
   
   for(int jf=0;jf<nfreq;jf++)
     {
     F[jf]+=alb_term*B*F0[jf];
     if(albedo==1 && INI_FIT_AO_ALBEDO==1)
     {



        
     }
       FTdx[jf*nvert+j1]+=dBdx1*F0[jf]+B*(FTda[jf]*dadx+FTdb[jf]*dbdx);
       FTdx[jf*nvert+j2]+=dBdx2*F0[jf]+B*(FTdc[jf]*dadx+FTdd[jf]*dbdx);
       FTdx[jf*nvert+j3]+=dBdx3*F0[jf]+B*(FTdg[jf]*dadx+FTdh[jf]*dbdx);
       
       FTdy[jf*nvert+j1]+=dBdy1*F0[jf]+B*(FTda[jf]*dady+FTdb[jf]*dbdy);
       FTdy[jf*nvert+j2]+=dBdy2*F0[jf]+B*(FTdc[jf]*dady+FTdd[jf]*dbdy);
       FTdy[jf*nvert+j3]+=dBdy3*F0[jf]+B*(FTdg[jf]*dady+FTdh[jf]*dbdy);
       
       FTdz[jf*nvert+j1]+=dBdz1*F0[jf]+B*(FTda[jf]*dadz+FTdb[jf]*dbdz);
       FTdz[jf*nvert+j2]+=dBdz2*F0[jf]+B*(FTdc[jf]*dadz+FTdd[jf]*dbdz);
       FTdz[jf*nvert+j3]+=dBdz3*F0[jf]+B*(FTdg[jf]*dadz+FTdh[jf]*dbdz);
        
       
       FTdA[jf*3+0]+=dBdb*F0[jf]+B*(FTda[jf]*v1db[0]+FTdb[jf]*v1db[1]+FTdc[jf]*v2db[0]+FTdd[jf]*v2db[1]+FTdg[jf]*v3db[0]+FTdh[jf]*v3db[1]);
       FTdA[jf*3+1]+=dBdl*F0[jf]+B*(FTda[jf]*v1dl[0]+FTdb[jf]*v1dl[1]+FTdc[jf]*v2dl[0]+FTdd[jf]*v2dl[1]+FTdg[jf]*v3dl[0]+FTdh[jf]*v3dl[1]);
       FTdA[jf*3+2]+=dBdo*F0[jf]+B*(FTda[jf]*v1do[0]+FTdb[jf]*v1do[1]+FTdc[jf]*v2do[0]+FTdd[jf]*v2do[1]+FTdg[jf]*v3do[0]+FTdh[jf]*v3do[1]);
     }
    if(INI_HAPKE!=NULL)
    {
        TB=TB+alb_term*B*area*mu;
        if(albedo==1 && INI_FIT_AO_ALBEDO==1)
        {
            
            dAlbTB[j1]+=dAlb3[0]*B*area*mu;
            dAlbTB[j2]+=dAlb3[1]*B*area*mu;
            dAlbTB[j3]+=dAlb3[2]*B*area*mu;
        }
    }
    else
    {
      TB=TB+alb_term*B*area*mu;
        if(albedo==1 && INI_FIT_AO_ALBEDO==1)
        {
            
           dAlbTB[j1]+=dAlb3[0]*B*area*mu;
            dAlbTB[j2]+=dAlb3[1]*B*area*mu;
            dAlbTB[j3]+=dAlb3[2]*B*area*mu;
        }
    }       
}


double TB2=pow(TB,2);
Complex temp;
for(int j=0;j<nfreq;j++)
{
  scale=std::exp(2.0*PI*Complex(0.0,1.0)*(offset[0]*freqx[j]+offset[1]*freqy[j]));
  for(int k=0;k<nvert;k++)
  {
    dFdxr[j*nvert+k]=std::real(dp*scale*(FTdx[j*nvert+k]*TB-F[j]*dTBdx[k])/TB2);
    dFdxi[j*nvert+k]=std::imag(dp*scale*(FTdx[j*nvert+k]*TB-F[j]*dTBdx[k])/TB2);
    dFdyr[j*nvert+k]=std::real(dp*scale*(FTdy[j*nvert+k]*TB-F[j]*dTBdy[k])/TB2);
    dFdyi[j*nvert+k]=std::imag(dp*scale*(FTdy[j*nvert+k]*TB-F[j]*dTBdy[k])/TB2);
    dFdzr[j*nvert+k]=std::real(dp*scale*(FTdz[j*nvert+k]*TB-F[j]*dTBdz[k])/TB2);
    dFdzi[j*nvert+k]=std::imag(dp*scale*(FTdz[j*nvert+k]*TB-F[j]*dTBdz[k])/TB2);
  }
  dFdAr[j*3+0]=std::real(scale*(FTdA[j*3+0]*TB-F[j]*dTBdA[0])/TB2);
  dFdAi[j*3+0]=std::imag(scale*(FTdA[j*3+0]*TB-F[j]*dTBdA[0])/TB2);
  dFdAr[j*3+1]=std::real(scale*(FTdA[j*3+1]*TB-F[j]*dTBdA[1])/TB2);
  dFdAi[j*3+1]=std::imag(scale*(FTdA[j*3+1]*TB-F[j]*dTBdA[1])/TB2);
  dFdAr[j*3+2]=std::real(scale*(FTdA[j*3+2]*TB-F[j]*dTBdA[2])/TB2);
  dFdAi[j*3+2]=std::imag(scale*(FTdA[j*3+2]*TB-F[j]*dTBdA[2])/TB2);
  temp=scale*F[j]/TB;
  Fr[j]=std::real(temp);
  Fi[j]=std::imag(temp);
  dFdoffr[j*2+0]=std::real(2.0*Complex(0.0,1.0)*PI*freqx[j]*temp);
  dFdoffi[j*2+0]=std::imag(2.0*Complex(0.0,1.0)*PI*freqx[j]*temp);
  dFdoffr[j*2+1]=std::real(2.0*Complex(0.0,1.0)*PI*freqy[j]*temp);
  dFdoffi[j*2+1]=std::imag(2.0*Complex(0.0,1.0)*PI*freqy[j]*temp);
}
if(albedo==1 && INI_FIT_AO_ALBEDO==1)
{
    for(int k=0;k<nvert;k++)
        for(int j=0;j<nfreq;j++)
        {
            scale=std::exp(2.0*PI*Complex(0.0,1.0)*(offset[0]*freqx[j]+offset[1]*freqy[j]));
            
            temp=scale*F[j]/TB;
            Fr[j]=std::real(temp);
            Fi[j]=std::imag(temp);
            temp=scale*(dAlb[j*nvert+k]*TB-F[j]*dAlbTB[k])/TB2;
            dAr[j*nvert+k]=std::real(temp);
            dAi[j*nvert+k]=std::imag(temp);
        }
        free(dAlb);
        free(dAlbTB);
}
free(FTdx);
free(FTdy);
free(FTdz);
free(FTdA);
free(dTBdx);
free(dTBdy);
free(dTBdz);
free(F);
free(FTda);
free(FTdb);
free(FTdc);
free(FTdd);
free(FTdg);
free(FTdh);
free(F0);
free(visible);



}

void calculate_lcurve(int *tlist,double *vlist,int numfac,int numvert,double *angles,double *Eo,double *E0o,int nE,double *TIME,double *bright,double *dbrightx,double *dbrighty,double *dbrightz,double *dbrightb,double *dbrightl,double *dbrighto,double *dbrightp,double *A,double *Alimit,double *dA,int rel,double *params)
{
   
  double *normal,*centroid,*E,*E0;
  double phase,dphase[3];
 double la=0;
 double ha=1; 
 double alb_term=0;
  double *dSx,*dSy,*dSz;
  int albedo=0;
  normal=allocate_raw(numfac*3*sizeof(double));
  double dAlb3[3];
  E=allocate_raw(3*nE*sizeof(double));
  E0=allocate_raw(3*nE*sizeof(double));
  
  double M[3][3],dMb[3][3],dMo[3][3],dMl[3][3];
  double vect[3];
  double alb=0;
  double *dSdalb=NULL;
    double albexp;
 if(false)
 {
     albedo=1;
     dSdalb=allocate_zero(nE*numvert,sizeof(double));
     la=Alimit[0];
     ha=Alimit[1];
 }
 
     
  int *visiblel;
  
  double beta,lambda,omega;
  double mu,mu0;
  double *tb;
  
  double c=0.1;
 

  double In,dIx1,dIx2,dIx3,dIy1,dIy2,dIy3,dIz1,dIz2,dIz3;
  double x1,y1,z1,x2,y2,z2,x3,y3,z3;
  double v1[3],v2[3],v3[3],cv[3],normc,Scatt;
  double dir[3],dir0[3],n[3];
  double dcx1[3],dcx2[3],dcx3[3],dcy1[3],dcy2[3],dcy3[3],dcz1[3],dcz2[3],dcz3[3];
  double dmux1,dmux2,dmux3,dmuy1,dmuy2,dmuy3,dmuz1,dmuz2,dmuz3;
  double dmu0x1,dmu0x2,dmu0x3,dmu0y1,dmu0y2,dmu0y3,dmu0z1,dmu0z2,dmu0z3;
  double *area;

   double *dareax,*dareay,*dareaz;
  dareax=allocate_zero(numfac*3,sizeof(double));
  dareay=allocate_zero(numfac*3,sizeof(double));
  dareaz=allocate_zero(numfac*3,sizeof(double));
 
  double dnx1[3],dnx2[3],dnx3[3],dny1[3],dny2[3],dny3[3],dnz1[3],dnz2[3],dnz3[3];
  double dAx1,dAx2,dAx3,dAy1,dAy2,dAy3,dAz1,dAz2,dAz3;
  double dt1,dt2;
 

  int vert1,vert2,vert3;

  
  dSx=allocate_zero(nE*numvert,sizeof(double));
  dSy=allocate_zero(nE*numvert,sizeof(double));
  dSz=allocate_zero(nE*numvert,sizeof(double));
  
  double dub,dul,duo,du0b,du0l,du0o;
  double *dBb,*dBl,*dBo;
  dBb=allocate_zero(nE,sizeof(double));
  dBl=allocate_zero(nE,sizeof(double));
  dBo=allocate_zero(nE,sizeof(double));
  
  double sumbright=0;
  double suma,sumb,sumc;
  
  
 
   double *dnormalx1,*dnormalx2,*dnormalx3;
  double *dnormaly1,*dnormaly2,*dnormaly3;
  double *dnormalz1,*dnormalz2,*dnormalz3;
  dnormalx1=allocate_zero(numfac*3,sizeof(double));
  dnormalx2=allocate_zero(numfac*3,sizeof(double));
  dnormalx3=allocate_zero(numfac*3,sizeof(double));
  dnormaly1=allocate_zero(numfac*3,sizeof(double));
  dnormaly2=allocate_zero(numfac*3,sizeof(double));
  dnormaly3=allocate_zero(numfac*3,sizeof(double));
  dnormalz1=allocate_zero(numfac*3,sizeof(double));
  dnormalz2=allocate_zero(numfac*3,sizeof(double));
  dnormalz3=allocate_zero(numfac*3,sizeof(double));
  

  area=allocate_zero(numfac,sizeof(double));
  tb=allocate_zero(nE,sizeof(double));
  visiblel=allocate_zero(nE*numfac,sizeof(int));
 
  double omega0;
  beta=angles[0];
  lambda=angles[1];
  omega=angles[2];
  omega0=angles[3];
 
  
  for(int j=0;j<nE;j++)
  {
    rotate(beta,lambda,omega,omega0,TIME[j],M,dMb,dMl,dMo);
    mult_vector(M,&(Eo[3*j]),&(E[3*j]));
    mult_vector(M,&(E0o[3*j]),&(E0[3*j]));
  }
  
  
  FindActualBlockers(tlist,vlist,numfac,numvert,E,E0,nE,visiblel);
  int vr1,vr2,vr3;
  

    
    
    for(int j=0;j<numfac;j++)
    {
        vr1=tlist[3*j]-1; 
        vr2=tlist[3*j+1]-1;
        vr3=tlist[3*j+2]-1;
      x1=vlist[3*vr1];
      y1=vlist[3*vr1+1];
      z1=vlist[3*vr1+2];
      
      x2=vlist[3*vr2];
      y2=vlist[3*vr2+1];
      z2=vlist[3*vr2+2];
      
      x3=vlist[3*vr3];
      y3=vlist[3*vr3+1];
      z3=vlist[3*vr3+2];
      
      v1[0]=x2-x1;
      v1[1]=y2-y1;
      v1[2]=z2-z1;
      
      v2[0]=x3-x1;
      v2[1]=y3-y1;
      v2[2]=z3-z1;
      
      cross(v1,v2,cv);
      normc=NORM(cv);
      normal[3*j+0]=cv[0]/normc;
      normal[3*j+1]=cv[1]/normc;
      normal[3*j+2]=cv[2]/normc;
      
      dcx1[0]=0;
      dcx1[1]=z3-z2;
      dcx1[2]=y2-y3;
      
      dcx2[0]=0;
      dcx2[1]=z1-z3;
      dcx2[2]=y3-y1;
      
      dcx3[0]=0;
      dcx3[1]=z2-z1;
      dcx3[2]=y1-y2;
      
      dcy1[0]=z2-z3;
      dcy1[1]=0;
      dcy1[2]=x3-x2;
      
      dcy2[0]=z3-z1;
      dcy2[1]=0;
      dcy2[2]=x1-x3;
      
      dcy3[0]=z1-z2;
      dcy3[1]=0;
      dcy3[2]=x2-x1;
      
      dcz1[0]=y3-y2;
      dcz1[1]=x2-x3;
      dcz1[2]=0;
      
      dcz2[0]=y1-y3;
      dcz2[1]=x3-x1;
      dcz2[2]=0;
      
      dcz3[0]=y2-y1;
      dcz3[1]=x1-x2;
      dcz3[2]=0;
      
      dareax[3*j+0]=1/(2*normc)*(cv[0]*dcx1[0]+cv[1]*dcx1[1]+cv[2]*dcx1[2]);
      dareax[3*j+1]=1/(2*normc)*(cv[0]*dcx2[0]+cv[1]*dcx2[1]+cv[2]*dcx2[2]);
      dareax[3*j+2]=1/(2*normc)*(cv[0]*dcx3[0]+cv[1]*dcx3[1]+cv[2]*dcx3[2]);
      
      dareay[3*j+0]=1/(2*normc)*(cv[0]*dcy1[0]+cv[1]*dcy1[1]+cv[2]*dcy1[2]);
      dareay[3*j+1]=1/(2*normc)*(cv[0]*dcy2[0]+cv[1]*dcy2[1]+cv[2]*dcy2[2]);
      dareay[3*j+2]=1/(2*normc)*(cv[0]*dcy3[0]+cv[1]*dcy3[1]+cv[2]*dcy3[2]);
      
      dareaz[3*j+0]=1/(2*normc)*(cv[0]*dcz1[0]+cv[1]*dcz1[1]+cv[2]*dcz1[2]);
      dareaz[3*j+1]=1/(2*normc)*(cv[0]*dcz2[0]+cv[1]*dcz2[1]+cv[2]*dcz2[2]);
      dareaz[3*j+2]=1/(2*normc)*(cv[0]*dcz3[0]+cv[1]*dcz3[1]+cv[2]*dcz3[2]);
      
      area[j]=0.5*normc; 
      
      for(int jk=0;jk<3;jk++)
      {
	dnormalx1[3*j+jk]=(normc*dcx1[jk]-cv[jk]*2*dareax[3*j+0])/(normc*normc);
	dnormalx2[3*j+jk]=(normc*dcx2[jk]-cv[jk]*2*dareax[3*j+1])/(normc*normc);
	dnormalx3[3*j+jk]=(normc*dcx3[jk]-cv[jk]*2*dareax[3*j+2])/(normc*normc);
	
	dnormaly1[3*j+jk]=(normc*dcy1[jk]-cv[jk]*2*dareay[3*j+0])/(normc*normc);
	dnormaly2[3*j+jk]=(normc*dcy2[jk]-cv[jk]*2*dareay[3*j+1])/(normc*normc);
	dnormaly3[3*j+jk]=(normc*dcy3[jk]-cv[jk]*2*dareay[3*j+2])/(normc*normc);
	
	dnormalz1[3*j+jk]=(normc*dcz1[jk]-cv[jk]*2*dareaz[3*j+0])/(normc*normc);
	dnormalz2[3*j+jk]=(normc*dcz2[jk]-cv[jk]*2*dareaz[3*j+1])/(normc*normc);
	dnormalz3[3*j+jk]=(normc*dcz3[jk]-cv[jk]*2*dareaz[3*j+2])/(normc*normc);
      }
      
    }
    
    
    for(int e=0;e<nE;e++)
    {
        for(int i=0;i<3;i++)
        {
        dir[i]=E[3*e+i];
	  dir0[i]=E0[3*e+i];
        }
       
      rotate(beta,lambda,omega,omega0,TIME[e],M,dMb,dMl,dMo);
      for(int j=0;j<numfac;j++)
      {
	if(visiblel[e*numfac+j]==0) 
	  continue;
	
	for(int i=0;i<3;i++)
	{
	  
	  n[i]=normal[3*j+i];
	  dnx1[i]=dnormalx1[3*j+i];
	  dnx2[i]=dnormalx2[3*j+i];
	  dnx3[i]=dnormalx3[3*j+i];
	  dny1[i]=dnormaly1[3*j+i];
	  dny2[i]=dnormaly2[3*j+i];
	  dny3[i]=dnormaly3[3*j+i];
	  dnz1[i]=dnormalz1[3*j+i];
	  dnz2[i]=dnormalz2[3*j+i];
	  dnz3[i]=dnormalz3[3*j+i];
	}
	vr1=tlist[3*j]-1; 
        vr2=tlist[3*j+1]-1;
        vr3=tlist[3*j+2]-1;
	x1=vlist[3*vr1+0];
	y1=vlist[3*vr1+1];
	z1=vlist[3*vr1+2];
	
	x2=vlist[3*vr2+0];
	y2=vlist[3*vr2+1];
	z2=vlist[3*vr2+2];
	
	x3=vlist[3*vr3+0];
	y3=vlist[3*vr3+1];
	z3=vlist[3*vr3+2];
	
	dAx1=dareax[3*j+0];
	dAx2=dareax[3*j+1];
	dAx3=dareax[3*j+2];
	
	dAy1=dareay[3*j+0];
	dAy2=dareay[3*j+1];
	dAy3=dareay[3*j+2];
	
	dAz1=dareaz[3*j+0];
	dAz2=dareaz[3*j+1];
	dAz3=dareaz[3*j+2];
	
	dmux1=DOT(dnx1,dir);
	dmux2=DOT(dnx2,dir);
	dmux3=DOT(dnx3,dir);
	
	dmuy1=DOT(dny1,dir);
	dmuy2=DOT(dny2,dir);
	dmuy3=DOT(dny3,dir);
	
	dmuz1=DOT(dnz1,dir);
	dmuz2=DOT(dnz2,dir);
	dmuz3=DOT(dnz3,dir);
	
	dmu0x1=DOT(dnx1,dir0);
	dmu0x2=DOT(dnx2,dir0);
	dmu0x3=DOT(dnx3,dir0);
	
	dmu0y1=DOT(dny1,dir0);
	dmu0y2=DOT(dny2,dir0);
	dmu0y3=DOT(dny3,dir0);
	
	dmu0z1=DOT(dnz1,dir0);
	dmu0z2=DOT(dnz2,dir0);
	dmu0z3=DOT(dnz3,dir0);
	vert1=tlist[3*j]-1;
	vert2=tlist[3*j+1]-1;
	vert3=tlist[3*j+2]-1;
	
	
	mu=DOT(dir,n);
	mu0=DOT(dir0,n);
	
        
       if(params!=NULL)
        {
            phase=phase_function(dir,dir0,params,dphase);
            
            
        }
        else
            phase=1;
        
        if(INI_HAPKE!=NULL)
            dhapke_bright(dir,dir0,mu,mu0,INI_HAPKE,INI_HAPKE[4],&Scatt,&dt2,&dt1); 
        else 
            Scatt=mu*mu0*(1.0/(mu+mu0)+c); 
      
        
        if(albedo==0)
        {
            alb=0;
            alb_term=1;
        }
        else
        {
            
            
            alb_term=Albedo_Term(tlist,vlist,numfac,numvert,Alimit,A,j,dAlb3);
            dSdalb[e*numvert+vert1]=phase*dAlb3[0]*Scatt*area[j];
           dSdalb[e*numvert+vert2]=phase*dAlb3[1]*Scatt*area[j];
           dSdalb[e*numvert+vert3]=phase*dAlb3[2]*Scatt*area[j];
            
            
        }
	In=alb_term*Scatt;
	tb[e]=tb[e]+phase*In*area[j];
       
	
        if(params!=NULL)
        {
            dbrightp[e*3]=dbrightp[e*3]+dphase[0]*In*area[j];
            dbrightp[e*3+1]=dbrightp[e*3+1]+dphase[1]*In*area[j];
            dbrightp[e*3+2]=dbrightp[e*3+2]+dphase[2]*In*area[j];
            
        }
        
        if(INI_HAPKE==NULL)
        {
            dt1=phase*alb_term*(pow(mu/(mu+mu0),2)+c*mu);
            dt2=phase*alb_term*(pow(mu0/(mu+mu0),2)+c*mu0);
        }
	
	dIx1=dt1*dmu0x1+dt2*dmux1;
	dIx2=dt1*dmu0x2+dt2*dmux2;
	dIx3=dt1*dmu0x3+dt2*dmux3;
	
	dIy1=dt1*dmu0y1+dt2*dmuy1;
	dIy2=dt1*dmu0y2+dt2*dmuy2;
	dIy3=dt1*dmu0y3+dt2*dmuy3;
	
	dIz1=dt1*dmu0z1+dt2*dmuz1;
	dIz2=dt1*dmu0z2+dt2*dmuz2;
	dIz3=dt1*dmu0z3+dt2*dmuz3;
	
	
	dSx[e*numvert+vert1]=dSx[e*numvert+vert1]+phase*In*dAx1+dIx1*area[j];
	dSx[e*numvert+vert2]=dSx[e*numvert+vert2]+phase*In*dAx2+dIx2*area[j];
	dSx[e*numvert+vert3]=dSx[e*numvert+vert3]+phase*In*dAx3+dIx3*area[j];
	
	dSy[e*numvert+vert1]=dSy[e*numvert+vert1]+phase*In*dAy1+dIy1*area[j];
	dSy[e*numvert+vert2]=dSy[e*numvert+vert2]+phase*In*dAy2+dIy2*area[j];
	dSy[e*numvert+vert3]=dSy[e*numvert+vert3]+phase*In*dAy3+dIy3*area[j];
	
	dSz[e*numvert+vert1]=dSz[e*numvert+vert1]+phase*In*dAz1+dIz1*area[j];
	dSz[e*numvert+vert2]=dSz[e*numvert+vert2]+phase*In*dAz2+dIz2*area[j];
	dSz[e*numvert+vert3]=dSz[e*numvert+vert3]+phase*In*dAz3+dIz3*area[j];
	
	
	mult_vector(dMb,&(Eo[3*e]),vect);
	dub=DOT(vect,n);
	mult_vector(dMl,&(Eo[3*e]),vect);
	dul=DOT(vect,n);
	mult_vector(dMo,&(Eo[3*e]),vect);
	duo=DOT(vect,n);
	mult_vector(dMb,&(E0o[3*e]),vect);
	du0b=DOT(vect,n);
	mult_vector(dMl,&(E0o[3*e]),vect);
	du0l=DOT(vect,n);
	mult_vector(dMo,&(E0o[3*e]),vect);
	du0o=DOT(vect,n);
	
	dBb[e]=dBb[e]+area[j]*(dt1*du0b+dt2*dub);
	dBl[e]=dBl[e]+area[j]*(dt1*du0l+dt2*dul);
	dBo[e]=dBo[e]+area[j]*(dt1*du0o+dt2*duo);
	
       } 
       
      } 
   double sum_paramc=0;
   double sum_param0=0,sum_param2=0,sum_param1=0;
     double sumbright2=0;
      sumbright=sum_vector(tb,nE);
      sumbright2=pow(sumbright,2);
    if(rel!=0)
    {
      double sbeta=0,slambda=0,somega=0;
      for(int j=0;j<nE;j++)
      {
	bright[j]=nE*tb[j]/sumbright;
	sbeta=sbeta+dBb[j];
	slambda=slambda+dBl[j];
	somega=somega+dBo[j];
        sum_param0+=dbrightp[j*3];
        sum_param1+=dbrightp[j*3+1];
        sum_param2+=dbrightp[j*3+2];
        
      }
      
      
      for(int sh_index=0;sh_index<numvert;sh_index++)
      {
	suma=0;
	sumb=0;
	sumc=0;
	
	for(int k=0;k<nE;k++)
	{
	  suma=suma+dSx[k*numvert+sh_index];
	  sumb=sumb+dSy[k*numvert+sh_index];
	  sumc=sumc+dSz[k*numvert+sh_index];
	}
	for(int i=0;i<nE;i++)
	{
	  
	  dbrightx[i*(numvert)+sh_index]=nE*(sumbright*dSx[i*numvert+sh_index]-tb[i]*suma)/sumbright2;
	  dbrighty[i*(numvert)+sh_index]=nE*(sumbright*dSy[i*numvert+sh_index]-tb[i]*sumb)/sumbright2;
	  dbrightz[i*(numvert)+sh_index]=nE*(sumbright*dSz[i*numvert+sh_index]-tb[i]*sumc)/sumbright2;
	}
      }
      
      
      for(int i=0;i<nE;i++)
      {
	dbrightb[i]=nE*(sumbright*dBb[i]-tb[i]*sbeta)/sumbright2;
	dbrightl[i]=nE*(sumbright*dBl[i]-tb[i]*slambda)/sumbright2;
	dbrighto[i]=nE*(sumbright*dBo[i]-tb[i]*somega)/sumbright2;
        
      }
      if(params!=NULL)
          for(int j=0;j<nE;j++)
        {
            dbrightp[j*3]=nE*(dbrightp[j*3]*sumbright-tb[j]*sum_param0)/sumbright2;
            dbrightp[j*3+1]=nE*(dbrightp[j*3+1]*sumbright-tb[j]*sum_param1)/sumbright2;
            dbrightp[j*3+2]=nE*(dbrightp[j*3+2]*sumbright-tb[j]*sum_param2)/sumbright2;
            
      }
      if(albedo==1)
      {
        for(int i=0;i<numvert;i++)
        {
            suma=0;
            for(int k=0;k<nE;k++)
                suma=suma+dSdalb[k*numvert+i];
            
            for(int h=0;h<nE;h++)
                dA[h*numvert+i]=nE*(sumbright*dSdalb[h*numvert+i]-tb[h]*suma)/sumbright2;
        }
      }
      
    }
    else
    {
        memcpy(bright,tb,sizeof(double)*nE);
        memcpy(dbrightx,dSx,sizeof(double)*nE*numvert);
        memcpy(dbrighty,dSy,sizeof(double)*nE*numvert);
        memcpy(dbrightz,dSz,sizeof(double)*nE*numvert);
        memcpy(dbrightb,dBb,sizeof(double)*nE);
        memcpy(dbrightl,dBl,sizeof(double)*nE);
        memcpy(dbrighto,dBo,sizeof(double)*nE);
        if(albedo==1)
            memcpy(dA,dSdalb,sizeof(double)*numvert*nE);
    }
    
    
      free(dSx);
      free(dSy);
      free(dSz);
      free(dBb);
      free(dBl);
      free(dBo);
     free(normal);
     free(E);
     free(E0);
     free(dareax);
     free(dareay);
     free(dareaz);
     free(area);
     free(tb);
     free(dnormalx1);
     free(dnormalx2);
     free(dnormalx3);
     free(dnormaly1);
     free(dnormaly2);
     free(dnormaly3);
     free(dnormalz1);
     free(dnormalz2);
     free(dnormalz3);
     free(visiblel);
     free(dSdalb);
}

double phase_function(double *E,double *E0,double *params,double *dpdp)
{
    
    double a=acos(DOT(E,E0));
    double e=exp(-a/params[1]);
    double br=1+params[0]*e+params[2]*a;
    dpdp[0]=e;
    dpdp[1]=params[0]*e*a/pow(params[1],2);
    dpdp[2]=a;
    return br;
}

void Fit_Occ(int *tlist,double *vlist,int nfac,int nvert,double *angles,double *up,double *E,double *V0,double *TIME,double *offset,double *chords,int *type,int nchords,double *W,double *Chordoffset,double *dist,double *dx,double *dy,double *dz,double *dangles,double *dtox,double *dtoy,double* dCOdoff)
{
    
    double time=0;
    double M[3][3],MRT[3][3],MRdb[3][3],MRdl[3][3],MRdo[3][3],MRTT[9];
    double MRdbT[9],MRdlT[9],MRdoT[9];
    double R[3][3],Rb[3][3],Rl[3][3],Ro[3][3],Rt[3][3],RbT[3][3],RlT[3][3],RoT[3][3];
    int faedge[2],cledge[2],inters;
    int edget[2];
    int ic1,ic2,fa1,fa2;
    double clpoint[2],fapoint[2],dclpx[4],dclpy[4],dfapx[4],dfapy[4];
    double temp4[4];
    double V[3];
    double dLx=0,dLy=0; 
     double nd_dist,nd_dx,nd_dy;
     int nd_vert;
    double ND_CHORD_WEIGHT=INI_NDCHORD_WEIGHT;
    for(int j=0;j<2*nchords;j++)
        time+=TIME[j];
    time=time/(2*nchords);
    int *Adj=allocate_zero(nvert*nvert,sizeof(int));
    AdjFacet(tlist,vlist,nfac,nvert,Adj);
    Calculate_Frame_Matrix(E,up,M);
    rotate(angles[0],angles[1],angles[2],angles[3],time,R,Rb,Rl,Ro);
    
    V[0]=M[0][0]*V0[0]+M[0][1]*V0[1]+M[0][2]*V0[2];
    V[1]=M[1][0]*V0[0]+M[1][1]*V0[1]+M[1][2]*V0[2];
    transpose(R,Rt); 
 transpose(Rb,RbT);
 transpose(Rl,RlT);
 transpose(Ro,RoT);
 mult_mat(M,Rt,MRT); 
double mr11,mr12,mr13,mr21,mr22,mr23;
mr11=MRT[0][0];
mr12=MRT[0][1];
mr13=MRT[0][2];
mr21=MRT[1][0];
mr22=MRT[1][1];
mr23=MRT[1][2];
 mult_mat(M,RbT,MRdb);
 mult_mat(M,RlT,MRdl);
 mult_mat(M,RoT,MRdo);
 
zero_array(dx,4*nchords*nvert);
zero_array(dy,4*nchords*nvert);
zero_array(dz,4*nchords*nvert);
zero_array(dangles,4*nchords*3);
zero_array(dtox,4*nchords);
zero_array(dtox,4*nchords);
zero_array(dCOdoff,4*nchords*nchords);
transpose2(MRT,MRTT);
transpose2(MRdb,MRdbT);
transpose2(MRdl,MRdlT);
transpose2(MRdo,MRdoT);
double a[2],b[2];
 double *vlist2=allocate_zero(nvert*3,sizeof(double));
 double *vlist2b=allocate_zero(nvert*3,sizeof(double));
 double *vlist2l=allocate_zero(nvert*3,sizeof(double));
 double *vlist2o=allocate_zero(nvert*3,sizeof(double));
 matrix_prod(vlist,nvert,3,MRTT,3,vlist2); 
  matrix_prod(vlist,nvert,3,MRdbT,3,vlist2b);
  matrix_prod(vlist,nvert,3,MRdlT,3,vlist2l);
  matrix_prod(vlist,nvert,3,MRdoT,3,vlist2o);
  double w;
 double el=0;
 for(int j=0;j<nchords;j++)
 {
     if(Chordoffset!=NULL)
     {
     dLx=V[0]*Chordoffset[j];
    dLy=V[1]*Chordoffset[j];
     }
     else
     {
         dLx=0;
         dLy=0;
     }
     if(W!=NULL)
         w=W[j];
     else
         w=1;
     a[0]=chords[4*j]+dLx;
     a[1]=chords[4*j+1]+dLy;
     b[0]=chords[4*j+2]+dLx;
     b[1]=chords[4*j+3]+dLy;
    
     find_chord(tlist,vlist2,nfac,nvert,offset,a,b,Adj,cledge,faedge,clpoint,fapoint,&inters,dclpx,dclpy,dfapx,dfapy);
     if(type[j]==-1)
     {
         find_chord_dist(tlist,vlist2,nfac,nvert,offset,a,b,&nd_dist,&nd_vert,&nd_dx,&nd_dy);
         if(inters==1)
            printf("Intersection with no-detection chord %d (of %d)\n",j,nchords);
           
        
         dist[4*j]=w*nd_dist*ND_CHORD_WEIGHT;
            dist[4*j+1]=0;
            dist[4*j+2]=0;
            dist[4*j+3]=0;
            el=ND_CHORD_WEIGHT*w*(nd_dx*mr11+nd_dy*mr21);
            set_el(dx,4*nchords,nvert,el,4*j,nd_vert);
            el=ND_CHORD_WEIGHT*w*(nd_dx*mr12+nd_dy*mr22);
            set_el(dy,4*nchords,nvert,el,4*j,nd_vert);
            el=ND_CHORD_WEIGHT*w*(nd_dx*mr13+nd_dy*mr23);
            set_el(dz,4*nchords,nvert,el,4*j,nd_vert);
            
            dtox[4*j]=w*ND_CHORD_WEIGHT*(nd_dx);
            dtoy[4*j]=w*ND_CHORD_WEIGHT*(nd_dy);
            
            dangles[4*j*3]=w*ND_CHORD_WEIGHT*(nd_dx*get_el(vlist2b,nvert,3,nd_vert,0)+nd_dy*get_el(vlist2b,nvert,3,nd_vert,1));
         dangles[4*j*3+1]=w*ND_CHORD_WEIGHT*(nd_dx*get_el(vlist2l,nvert,3,nd_vert,0)+nd_dy*get_el(vlist2l,nvert,3,nd_vert,1));
         dangles[4*j*3+2]=w*ND_CHORD_WEIGHT*(nd_dx*get_el(vlist2o,nvert,3,nd_vert,0)+nd_dy*get_el(vlist2o,nvert,3,nd_vert,1));
         
         continue;
     }   
     if(inters==1)
     {
         
         if((fapoint[0]-clpoint[0])*(b[0]-a[0])+(fapoint[1]-clpoint[1])*(b[1]-a[1])<0)
         {
             memcpy(edget,cledge,sizeof(int)*2);
             memcpy(cledge,faedge,sizeof(int)*2);
             memcpy(faedge,edget,sizeof(int)*2);
             
             memcpy(temp4,clpoint,sizeof(double)*2);
             memcpy(clpoint,fapoint,sizeof(double)*2);
             memcpy(fapoint,temp4,sizeof(double)*2);
             
             memcpy(temp4,dclpx,sizeof(double)*4);
             memcpy(dclpx,dfapx,sizeof(double)*4);
             memcpy(dfapx,temp4,sizeof(double)*4);
             
             memcpy(temp4,dclpy,sizeof(double)*4);
             memcpy(dclpy,dfapy,sizeof(double)*4);
             memcpy(dfapy,temp4,sizeof(double)*4);
         }
             
             
         
         dist[4*j]=w*(clpoint[0]-chords[4*j]-dLx); 
         dist[4*j+1]=w*(clpoint[1]-chords[4*j+1]-dLy); 
         dist[4*j+2]=w*(fapoint[0]-chords[4*j+2]-dLx);
         dist[4*j+3]=w*(fapoint[1]-chords[4*j+3]-dLy);
         ic1=cledge[0];
         ic2=cledge[1];
         fa1=faedge[0];
         fa2=faedge[1];
         
         el=dclpx[0]*mr11+dclpx[1]*mr21;
         el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j,ic1);
          el=dclpx[0]*mr12+dclpx[1]*mr22;
          el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j,ic1);
          el=dclpx[0]*mr13+dclpx[1]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j,ic1);
         
          el=dclpx[2]*mr11+dclpx[3]*mr21;
          el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j,ic2);
        el=dclpx[2]*mr12+dclpx[3]*mr22;
        el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j,ic2);
          el=dclpx[2]*mr13+dclpx[3]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j,ic2);
         
         el=dclpy[0]*mr11+dclpy[1]*mr21;
         el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j+1,ic1);
          el=dclpy[0]*mr12+dclpy[1]*mr22;
          el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j+1,ic1);
          el=dclpy[0]*mr13+dclpy[1]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j+1,ic1);
         
         el=dclpy[2]*mr11+dclpy[3]*mr21;
         el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j+1,ic2);
        el=dclpy[2]*mr12+dclpy[3]*mr22;
        el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j+1,ic2);
          el=dclpy[2]*mr13+dclpy[3]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j+1,ic2);
         
          el=dfapx[0]*mr11+dfapx[1]*mr21;
          el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j+2,fa1);
          el=dfapx[0]*mr12+dfapx[1]*mr22;
          el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j+2,fa1);
          el=dfapx[0]*mr13+dfapx[1]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j+2,fa1);
         
         el=dfapx[2]*mr11+dfapx[3]*mr21;
         el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j+2,fa2);
          el=dfapx[2]*mr12+dfapx[3]*mr22;
          el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j+2,fa2);
          el=dfapx[2]*mr13+dfapx[3]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j+2,fa2);
         
           el=dfapy[0]*mr11+dfapy[1]*mr21;
           el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j+3,fa1);
          el=dfapy[0]*mr12+dfapy[1]*mr22;
          el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j+3,fa1);
          el=dfapy[0]*mr13+dfapy[1]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j+3,fa1);
         
          el=dfapy[2]*mr11+dfapy[3]*mr21;
          el=w*el;
         set_el(dx,4*nchords,nvert,el,4*j+3,fa2);
          el=dfapy[2]*mr12+dfapy[3]*mr22;
          el=w*el;
         set_el(dy,4*nchords,nvert,el,4*j+3,fa2);
          el=dfapy[2]*mr13+dfapy[3]*mr23;
          el=w*el;
         set_el(dz,4*nchords,nvert,el,4*j+3,fa2);
         
         set_el(dCOdoff,4*nchords,nchords,-V[0],4*j,j);
         set_el(dCOdoff,4*nchords,nchords,-V[1],4*j+1,j);
         set_el(dCOdoff,4*nchords,nchords,-V[0],4*j+2,j);
         set_el(dCOdoff,4*nchords,nchords,-V[1],4*j+3,j);
         
         dtox[4*j]=w*(dclpx[0]+dclpx[2]);
         dtox[4*j+1]=w*(dclpy[0]+dclpy[2]);
         dtox[4*j+2]=w*(dfapx[0]+dfapx[2]);
         dtox[4*j+3]=w*(dfapy[0]+dfapy[2]);
         
         dtoy[4*j]=w*(dclpx[1]+dclpx[3]);
         dtoy[4*j+1]=w*(dclpy[1]+dclpy[3]);
         dtoy[4*j+2]=w*(dfapx[1]+dfapx[3]);
         dtoy[4*j+3]=w*(dfapy[1]+dfapy[3]);
         
         dangles[4*j*3]=w*(dclpx[0]*get_el(vlist2b,nvert,3,ic1,0)+dclpx[1]*get_el(vlist2b,nvert,3,ic1,1)+dclpx[2]*get_el(vlist2b,nvert,3,ic2,0)+dclpx[3]*get_el(vlist2b,nvert,3,ic2,1));
         dangles[4*j*3+1]=w*(dclpx[0]*get_el(vlist2l,nvert,3,ic1,0)+dclpx[1]*get_el(vlist2l,nvert,3,ic1,1)+dclpx[2]*get_el(vlist2l,nvert,3,ic2,0)+dclpx[3]*get_el(vlist2l,nvert,3,ic2,1));
         dangles[4*j*3+2]=w*(dclpx[0]*get_el(vlist2o,nvert,3,ic1,0)+dclpx[1]*get_el(vlist2o,nvert,3,ic1,1)+dclpx[2]*get_el(vlist2o,nvert,3,ic2,0)+dclpx[3]*get_el(vlist2o,nvert,3,ic2,1));
         
         dangles[4*j*3+3+0]=w*(dclpy[0]*get_el(vlist2b,nvert,3,ic1,0)+dclpy[1]*get_el(vlist2b,nvert,3,ic1,1)+dclpy[2]*get_el(vlist2b,nvert,3,ic2,0)+dclpy[3]*get_el(vlist2b,nvert,3,ic2,1));
         dangles[4*j*3+3+1]=w*(dclpy[0]*get_el(vlist2l,nvert,3,ic1,0)+dclpy[1]*get_el(vlist2l,nvert,3,ic1,1)+dclpy[2]*get_el(vlist2l,nvert,3,ic2,0)+dclpy[3]*get_el(vlist2l,nvert,3,ic2,1));
         dangles[4*j*3+3+2]=w*(dclpy[0]*get_el(vlist2o,nvert,3,ic1,0)+dclpy[1]*get_el(vlist2o,nvert,3,ic1,1)+dclpy[2]*get_el(vlist2o,nvert,3,ic2,0)+dclpy[3]*get_el(vlist2o,nvert,3,ic2,1));
         
         dangles[4*j*3+2*3+0]=w*(dfapx[0]*get_el(vlist2b,nvert,3,fa1,0)+dfapx[1]*get_el(vlist2b,nvert,3,fa1,1)+dfapx[2]*get_el(vlist2b,nvert,3,fa2,0)+dfapx[3]*get_el(vlist2b,nvert,3,fa2,1));
         dangles[4*j*3+2*3+1]=w*(dfapx[0]*get_el(vlist2l,nvert,3,fa1,0)+dfapx[1]*get_el(vlist2l,nvert,3,fa1,1)+dfapx[2]*get_el(vlist2l,nvert,3,fa2,0)+dfapx[3]*get_el(vlist2l,nvert,3,fa2,1));
         dangles[4*j*3+2*3+2]=w*(dfapx[0]*get_el(vlist2o,nvert,3,fa1,0)+dfapx[1]*get_el(vlist2o,nvert,3,fa1,1)+dfapx[2]*get_el(vlist2o,nvert,3,fa2,0)+dfapx[3]*get_el(vlist2o,nvert,3,fa2,1));
         
         dangles[4*j*3+3*3+0]=w*(dfapy[0]*get_el(vlist2b,nvert,3,fa1,0)+dfapy[1]*get_el(vlist2b,nvert,3,fa1,1)+dfapy[2]*get_el(vlist2b,nvert,3,fa2,0)+dfapy[3]*get_el(vlist2b,nvert,3,fa2,1));
         dangles[4*j*3+3*3+1]=w*(dfapy[0]*get_el(vlist2l,nvert,3,fa1,0)+dfapy[1]*get_el(vlist2l,nvert,3,fa1,1)+dfapy[2]*get_el(vlist2l,nvert,3,fa2,0)+dfapy[3]*get_el(vlist2l,nvert,3,fa2,1));
         dangles[4*j*3+3*3+2]=w*(dfapy[0]*get_el(vlist2o,nvert,3,fa1,0)+dfapy[1]*get_el(vlist2o,nvert,3,fa1,1)+dfapy[2]*get_el(vlist2o,nvert,3,fa2,0)+dfapy[3]*get_el(vlist2o,nvert,3,fa2,1));
     }
     else
     {
        find_chord_dist_simple(tlist,vlist2,nfac,nvert,offset,a,b,&nd_dist,&nd_vert,&nd_dx,&nd_dy);
         dist[4*j]=w*chords[j*4]*(1+nd_dist);
         dist[4*j+1]=w*chords[j*4+1];
         dist[4*j+2]=w*chords[j*4+2];
         dist[4*j+3]=w*chords[j*4+3];
         
         el=chords[j*4]*w*(nd_dx*mr11+nd_dy*mr21);
        set_el(dx,4*nchords,nvert,el,4*j,nd_vert);
            el=chords[j*4]*w*(nd_dx*mr12+nd_dy*mr22);
            set_el(dy,4*nchords,nvert,el,4*j,nd_vert);
            el=chords[j*4]*w*(nd_dx*mr13+nd_dy*mr23);
            set_el(dz,4*nchords,nvert,el,4*j,nd_vert);
            
            dtox[4*j]=w*chords[j*4]*(nd_dx);
            dtoy[4*j]=w*chords[j*4]*(nd_dy);
            
            dangles[4*j*3]=w*chords[j*4]*(nd_dx*get_el(vlist2b,nvert,3,nd_vert,0)+nd_dy*get_el(vlist2b,nvert,3,nd_vert,1));
         dangles[4*j*3+1]=w*chords[j*4]*(nd_dx*get_el(vlist2l,nvert,3,nd_vert,0)+nd_dy*get_el(vlist2l,nvert,3,nd_vert,1));
         dangles[4*j*3+2]=w*chords[j*4]*(nd_dx*get_el(vlist2o,nvert,3,nd_vert,0)+nd_dy*get_el(vlist2o,nvert,3,nd_vert,1));
         
     }
 }
 free(vlist2);
 free(vlist2b);
 free(vlist2l);
 free(vlist2o);
 free(Adj);
}

void find_chord_dist(int *tlist,double *vlist,int nfac,int nvert,double *offset,double *p1,double *p2,double* dist2,int* vert2,double *dddx2,double *dddy2)
{
    double w[2];
    double normw;
    double n[2];
    double offx,offy;
    double v[2];
    double d=0;
    double dist=0.0;
    int vert=0;
    double dddx,dddy;
    int k=INI_LOGEXP;
    offx=offset[0];
    offy=offset[1];
    double maxneg,maxpos;
    maxneg=0;
    maxpos=0;
    double closeneg=1E6,closepos=1E6;
    int closenegv=0,closeposv=0,maxnegv=0,maxposv=0;
    
    w[0]=p2[0]-p1[0];
    w[1]=p2[1]-p1[1];
    normw=sqrt(w[0]*w[0]+w[1]*w[1]);
    w[0]=w[0]/normw;
    w[1]=w[1]/normw;
    n[0]=-w[1];
    n[1]=w[0];
    
    for(int j=0;j<nvert;j++)
    {
        v[0]=p1[0]-(vlist[3*j]+offx);
        v[1]=p1[1]-(vlist[3*j+1]+offy);
        d=v[0]*n[0]+v[1]*n[1];
        if(d<0 && d<maxneg)
        {
            maxneg=d;
            maxnegv=j;

        }
        if(d>0 && d>maxpos)
        {
            maxpos=d;
            maxposv=j;
        }
        if(d<0 && -d<closeneg)
        {
            closeneg=-d;
            closenegv=j;
        }
        if(d>0 && d<closepos)
        {
            closepos=d;
            closeposv=j;
        }
    }
    if(maxneg<0 && maxpos>0)
    {
        if(-maxneg<=maxpos)
        {
            dist=maxneg;
            vert=maxnegv;
            dddx=-n[0];
            dddy=-n[1];
        }
        else
        {
            dist=-maxpos;
            vert=maxposv;
            dddx=n[0];
            dddy=n[1];
        }
    }
    else if(closepos<=closeneg)
    {
        dist=closepos;
        vert=closeposv;
        dddx=-n[0];
        dddy=-n[1];
    }
    else
    {
        dist=closeneg;
        vert=closenegv;
        dddx=n[0];
        dddy=n[1];
    }
    double tdist=dist;
   
   dist=1-1/(1+exp(-2*k*tdist));
  
   dddx=-2*k*exp(-2*k*tdist)/pow(1+exp(-2*k*tdist),2)*dddx;
   dddy=-2*k*exp(-2*k*tdist)/pow(1+exp(-2*k*tdist),2)*dddy;
    *dist2=dist;
    *vert2=vert;
    if(isnan(dddx))
        dddx=0;
    if(isnan(dddy))
        dddy=0;
    *dddx2=dddx;
    *dddy2=dddy;
    
}

void find_chord_dist_simple(int *tlist,double *vlist,int nfac,int nvert,double *offset,double *p1,double *p2,double* dist2,int* vert2,double *dddx2,double *dddy2)
{
    double w[2];
    double normw;
    double n[2];
    double offx,offy;
    double v[2];
    double d=0;
    double dist=0.0;
    int vert=0;
    double dddx,dddy;
    int k=INI_LOGEXP;
    offx=offset[0];
    offy=offset[1];
    double maxneg,maxpos;
    maxneg=0;
    maxpos=0;
    double closeneg=1E6,closepos=1E6;
    int closenegv=0,closeposv=0,maxnegv=0,maxposv=0;
    
    w[0]=p2[0]-p1[0];
    w[1]=p2[1]-p1[1];
    normw=sqrt(w[0]*w[0]+w[1]*w[1]);
    w[0]=w[0]/normw;
    w[1]=w[1]/normw;
    n[0]=-w[1];
    n[1]=w[0];
    
    for(int j=0;j<nvert;j++)
    {
        v[0]=p1[0]-(vlist[3*j]+offx);
        v[1]=p1[1]-(vlist[3*j+1]+offy);
        d=v[0]*n[0]+v[1]*n[1];
        if(d<0 && d<maxneg)
        {
            maxneg=d;
            maxnegv=j;

        }
        if(d>0 && d>maxpos)
        {
            maxpos=d;
            maxposv=j;
        }
        if(d<0 && -d<closeneg)
        {
            closeneg=-d;
            closenegv=j;
        }
        if(d>0 && d<closepos)
        {
            closepos=d;
            closeposv=j;
        }
    }
    if(maxneg<0 && maxpos>0)
    {
        if(-maxneg<=maxpos)
        {
            dist=maxneg;
            vert=maxnegv;
            dddx=-n[0];
            dddy=-n[1];
        }
        else
        {
            dist=-maxpos;
            vert=maxposv;
            dddx=n[0];
            dddy=n[1];
        }
    }
    else if(closepos<=closeneg)
    {
        dist=closepos;
        vert=closeposv;
        dddx=-n[0];
        dddy=-n[1];
    }
    else
    {
        dist=closeneg;
        vert=closenegv;
        dddx=n[0];
        dddy=n[1];
    }
    double tdist=dist;
   
   dist=1/(1+exp(-2*k*tdist));
  
   dddx=2*k*exp(-2*k*tdist)/pow(1+exp(-2*k*tdist),2)*dddx;
   dddy=2*k*exp(-2*k*tdist)/pow(1+exp(-2*k*tdist),2)*dddy;
    *dist2=dist;
    *vert2=vert;
    if(isnan(dddx))
        dddx=0;
    if(isnan(dddy))
        dddy=0;
    *dddx2=dddx;
    *dddy2=dddy;
    
}

void find_chord(int *tlist,double *vlist2,int nfac,int nvert,double *offset,double *a,double *b,int *Adj,int *cledge,int *faedge,double *clpoint,double *fapoint,int *inters,double *dclpx,double *dclpy,double *dfapx,double *dfapy)
{
    
    double eps=1E-10;
    int *A=allocate_zero(nvert*nvert,sizeof(int));
    int i1,i2,i3;
    double u1,u2,w1,w2,n3;
    double *v1,*v2,*v3;
    double ip[2];
    double dpx[4],dpy[4];
    double dist,cldist=1E9,fadist=0;
    double *vlist=allocate_zero(3*nvert,sizeof(double));
    memcpy(vlist,vlist2,sizeof(double)*nvert*3);
    memcpy(A,Adj,sizeof(int)*nvert*nvert);
    double bbx[2],bby[2];
    double offx=offset[0];
    double offy=offset[1];
    
    for(int j=0;j<nvert;j++)
    {
        vlist[3*j]+=offx;
        vlist[3*j+1]+=offy;
    }
    bbx[0]=minv(vlist,nvert,0);
    bbx[1]=maxv(vlist,nvert,0);
    bby[0]=minv(vlist,nvert,1);
    bby[1]=maxv(vlist,nvert,1);
    int t=0;
    double p[2];
    p[0]=a[0]+t*(b[0]-a[0]);
    p[1]=a[1]+t*(b[1]-a[1]);
    while(p[0]<bbx[1] && p[0]>bbx[0] && p[1]<bby[1] && p[1]>bby[0])
    {
        t--;
        p[0]=a[0]+t*(b[0]-a[0]);
        p[1]=a[1]+t*(b[1]-a[1]);
    }
    
    
    clpoint[0]=0;
    clpoint[1]=0;
    fapoint[0]=0;
    fapoint[1]=0;
    int sinters=0;
      for(int j=0;j<nfac;j++)
    {
        i1=tlist[3*j]-1; 
        i2=tlist[3*j+1]-1;
        i3=tlist[3*j+2]-1;
        u1=vlist[3*i2]-vlist[3*i1];
        u2=vlist[3*i2+1]-vlist[3*i1+1];
        w1=vlist[3*i3]-vlist[3*i1];
        w2=vlist[3*i3+1]-vlist[3*i1+1];
        n3=u1*w2-u2*w1;
        if(n3<=0)
            continue;
        v1=vlist+3*i1;
        v2=vlist+3*i2;
        v3=vlist+3*i3;
        
        if(get_elI(A,nvert,nvert,i1,i2))
        {
           
            if(line_intersect(a,b,v1,v2,ip,dpx,dpy))
            {
                sinters=1;
                 
                dist=sqrt(pow(ip[0]-p[0],2)+pow(ip[1]-p[1],2));
               
                if(dist<cldist)
                {
                    cldist=dist;
                    cledge[0]=i1;
                    cledge[1]=i2;
                    clpoint[0]=ip[0];
                    clpoint[1]=ip[1];
                    memcpy(dclpx,dpx,sizeof(double)*4);
                    memcpy(dclpy,dpy,sizeof(double)*4);
                }
                if(dist>fadist)
                {
                    fadist=dist;
                    faedge[0]=i1;
                    faedge[1]=i2;
                    fapoint[0]=ip[0];
                    fapoint[1]=ip[1];
                    
                    memcpy(dfapx,dpx,sizeof(double)*4);
                    memcpy(dfapy,dpy,sizeof(double)*4);
                }
            }
            set_elI(A,nvert,nvert,0,i2,i1);
            set_elI(A,nvert,nvert,0,i1,i2);
        }
      
        if(get_elI(A,nvert,nvert,i2,i3))
        {
            if(line_intersect(a,b,v2,v3,ip,dpx,dpy))
            {
                 sinters=1;
                dist=sqrt(pow(ip[0]-p[0],2)+pow(ip[1]-p[1],2));
              
                 if(dist<cldist)
                {
                    cldist=dist;
                    cledge[0]=i2;
                    cledge[1]=i3;
                    clpoint[0]=ip[0];
                    clpoint[1]=ip[1];
                    memcpy(dclpx,dpx,sizeof(double)*4);
                    memcpy(dclpy,dpy,sizeof(double)*4);
                }
                if(dist>fadist)
                {
                    fadist=dist;
                    faedge[0]=i2;
                    faedge[1]=i3;
                    fapoint[0]=ip[0];
                    fapoint[1]=ip[1];
                    
                    memcpy(dfapx,dpx,sizeof(double)*4);
                  
                    memcpy(dfapy,dpy,sizeof(double)*4);
                }
            }
            set_elI(A,nvert,nvert,0,i2,i3);
            set_elI(A,nvert,nvert,0,i3,i2);
        }
       
        if(get_elI(A,nvert,nvert,i3,i1))
        {
            if(line_intersect(a,b,v3,v1,ip,dpx,dpy))
            {
                 sinters=1;
                dist=sqrt(pow(ip[0]-p[0],2)+pow(ip[1]-p[1],2));
                
                 if(dist<cldist)
                {
                    cldist=dist;
                    cledge[0]=i3;
                    cledge[1]=i1;
                    clpoint[0]=ip[0];
                    clpoint[1]=ip[1];
                    memcpy(dclpx,dpx,sizeof(double)*4);
                    memcpy(dclpy,dpy,sizeof(double)*4);
                }
                if(dist>fadist)
                {
                    fadist=dist;
                    faedge[0]=i3;
                    faedge[1]=i1;
                    fapoint[0]=ip[0];
                    fapoint[1]=ip[1];
                    
                    memcpy(dfapx,dpx,sizeof(double)*4);
                  
                    memcpy(dfapy,dpy,sizeof(double)*4);
                }
            }
            set_elI(A,nvert,nvert,0,i3,i1);
            set_elI(A,nvert,nvert,0,i1,i3);
        }
    }
    if(pow(clpoint[0]-fapoint[0],2)+pow(clpoint[1]-fapoint[1],2)>eps && sinters==1)
        *inters=1;
    else
        *inters=0;
    free(A);
    free(vlist);
}

int line_intersect(double *a,double *b,double *x,double *y,double *p,double *dpx,double *dpy)
{
    
    double eps=1E-10;
    int inters=0;
    double t=0;
    double b0=b[0];
    double b1=b[1];
    double a0=a[0];
    double a1=a[1];
    double x0=x[0];
    double x1=x[1];
    double y0=y[0];
    double y1=y[1];
    double dtdx[2],dtdy[2];
    double denom=(b0-a0)*(x1-y1)-(b1-a1)*(x0-y0);
    if(fabs(denom)<eps)
        return 0;
    t=((a0-x0)*(b1-a1)-(a1-x1)*(b0-a0))/denom;
    if(t>=0 && t<=1)
    {
        inters=1;
        p[0]=x[0]+t*(y[0]-x[0]);
        p[1]=x[1]+t*(y[1]-x[1]);
    }
    dtdx[0]=(a1-b1)*(a1*(b0-y0)+b1*y0-b0*y1+a0*(-b1+y1))/pow(denom,2);
    dtdx[1]=(a1 - b1)*(-b1*x0 + a1*(-b0 + x0) + a0*(b1 - x1) + b0*x1)/pow(denom,2);
    dtdy[0]=(a0 - b0)*(-b1*y0 + a1*(-b0 + y0) + a0*(b1 - y1) + b0*y1)/pow(denom,2);
    dtdy[1]=-(a0 - b0)*(-b1*x0 + a1*(-b0 + x0) + a0*(b1 - x1) + b0*x1)/pow(denom,2);
    dpx[0]=1+dtdx[0]*(y0-x0)-t;
    dpx[1]=dtdy[0]*(y0-x0);
    dpx[2]=dtdx[1]*(y0-x0)+t;
    dpx[3]=dtdy[1]*(y0-x0);
    
    dpy[0]=dtdx[0]*(y1-x1);
    dpy[1]=1+dtdy[0]*(y1-x1)-t;
    dpy[2]=dtdx[1]*(y1-x1);
    dpy[3]=dtdy[1]*(y1-x1)+t;
    return inters;
}

void convex_reg(int* tlist,double* vlist,int nfac,int nvert,double *D,int dm,int dn,double *res,double *drdv)
{
    
    
    int nedge;
    double result=0;
    int *E,*N,*E2,*A;
    double *v1,*v2,*v3,n1[3],n2[3];
    double *w1,*w2,*w3;
    double cangle;
    double TA=0;
    double *dTAdx,*dTAdy,*dTAdz;
    dTAdx=allocate_zero(nvert,sizeof(double));
    dTAdy=allocate_zero(nvert,sizeof(double));
    dTAdz=allocate_zero(nvert,sizeof(double));
    
    double dn1dx1[3],dn1dx2[3],dn1dx3[3];
    double dn1dy1[3],dn1dy2[3],dn1dy3[3];
    double dn1dz1[3],dn1dz2[3],dn1dz3[3];
    
    double dn2dx1[3],dn2dx2[3],dn2dx3[3];
    double dn2dy1[3],dn2dy2[3],dn2dy3[3];
    double dn2dz1[3],dn2dz2[3],dn2dz3[3];
    double area1,area2;
    double dA1dx[3],dA1dy[3],dA1dz[3];
    double dA2dx[3],dA2dy[3],dA2dz[3];
    double *dresdx,*dresdy,*dresdz;
    dresdx=allocate_zero(nvert,sizeof(double));
    dresdy=allocate_zero(nvert,sizeof(double));
    dresdz=allocate_zero(nvert,sizeof(double));
    E=allocate_zero(nvert*nvert,sizeof(int));
    N=allocate_zero(nvert*nfac,sizeof(int));
    E2=allocate_zero(nvert*nvert,sizeof(int));
    A=allocate_zero(nfac*nfac,sizeof(int));
    
    if(D!=NULL && dm!=nvert)
    {
        puts("Error: Number of vertex coordinates is not equal to the number of rows in D.");
        exit(1);
    }
    if(D==NULL)
        dn=nvert;
    double *drdx,*drdy,*drdz;
    drdx=allocate_zero(dn,sizeof(double));
    drdy=allocate_zero(dn,sizeof(double));
    drdz=allocate_zero(dn,sizeof(double));
    
    find_neighborhood(tlist,vlist,nfac,nvert,E,N,E2,A);
    
    free(E);
    free(E2);
    free(N);
    nedge=nfac+nvert-2;
    int ind;
    int CB=0;
    int i1,i2,i3;
    int j1,j2,j3;
    int count=0;
    int *NumofBlocks;
    int *IndexofBlocks;
    double *normal,*centroid;
    NumofBlocks=allocate_zero(nfac,sizeof(int));
    IndexofBlocks=allocate_zero(nfac*nfac,sizeof(int));
    normal=allocate_raw(3*nfac*sizeof(double)); 
    centroid=allocate_raw(3*nfac*sizeof(double));
    FacetsOverHorizon(tlist,vlist,nfac,nvert,normal,centroid,NumofBlocks,IndexofBlocks);
    free(normal);
    free(centroid);
    
    
    for(int k=0;k<nfac;k++)
    {
        
        i1=tlist[3*k]-1;
        i2=tlist[3*k+1]-1;
        i3=tlist[3*k+2]-1;
        v1=vlist+3*i1;
        v2=vlist+3*i2;
        v3=vlist+3*i3;
        
        Calculate_Area_and_Normal_Derivative(v1,v2,v3,n1,dn1dx1,dn1dx2,dn1dx3,dn1dy1,dn1dy2,dn1dy3,dn1dz1,dn1dz2,dn1dz3,&area1,dA1dx,dA1dy,dA1dz);
        
        TA+=area1;
        dTAdx[i1]+=dA1dx[0];
        dTAdx[i2]+=dA1dx[1];
        dTAdx[i3]+=dA1dx[2];
        
        dTAdy[i1]+=dA1dy[0];
        dTAdy[i2]+=dA1dy[1];
        dTAdy[i3]+=dA1dy[2];
        
        dTAdz[i1]+=dA1dz[0];
        dTAdz[i2]+=dA1dz[1];
        dTAdz[i3]+=dA1dz[2];
        if(NumofBlocks[k]==0)
            continue; 
            
            for(int j=0;j<NumofBlocks[k];j++)
            {
                CB=IndexofBlocks[nfac*k+j]-1; 
                
                if(A[nfac*k+CB]==0)
                    continue; 
                    
                    j1=tlist[3*CB]-1;
                j2=tlist[3*CB+1]-1;
                j3=tlist[3*CB+2]-1;
                
                w1=vlist+3*j1;
                w2=vlist+3*j2;
                w3=vlist+3*j3;
                Calculate_Area_and_Normal_Derivative(w1,w2,w3,n2,dn2dx1,dn2dx2,dn2dx3,dn2dy1,dn2dy2,dn2dy3,dn2dz1,dn2dz2,dn2dz3,&area2,dA2dx,dA2dy,dA2dz);
                
                cangle=DOT(n1,n2);
                result+=(area2)*(1-cangle);
                dresdx[i1]+=(area2)*(-DOT(dn1dx1,n2));
                dresdx[i2]+=(area2)*(-DOT(dn1dx2,n2));
                dresdx[i3]+=(area2)*(-DOT(dn1dx3,n2));
                
                dresdy[i1]+=(area2)*(-DOT(dn1dy1,n2));
                dresdy[i2]+=(area2)*(-DOT(dn1dy2,n2));
                dresdy[i3]+=(area2)*(-DOT(dn1dy3,n2));
                
                dresdz[i1]+=(area2)*(-DOT(dn1dz1,n2));
                dresdz[i2]+=(area2)*(-DOT(dn1dz2,n2));
                dresdz[i3]+=(area2)*(-DOT(dn1dz3,n2));
                
                dresdx[j1]+=dA2dx[0]*(1-cangle)+(area2)*(-DOT(dn2dx1,n1));
                dresdx[j2]+=dA2dx[1]*(1-cangle)+(area2)*(-DOT(dn2dx2,n1));
                dresdx[j3]+=dA2dx[2]*(1-cangle)+(area2)*(-DOT(dn2dx3,n1));
                
                dresdy[j1]+=dA2dy[0]*(1-cangle)+(area2)*(-DOT(dn2dy1,n1));
                dresdy[j2]+=dA2dy[1]*(1-cangle)+(area2)*(-DOT(dn2dy2,n1));
                dresdy[j3]+=dA2dy[2]*(1-cangle)+(area2)*(-DOT(dn2dy3,n1));
                
                dresdz[j1]+=dA2dz[0]*(1-cangle)+(area2)*(-DOT(dn2dz1,n1));
                dresdz[j2]+=dA2dz[1]*(1-cangle)+(area2)*(-DOT(dn2dz2,n1));
                dresdz[j3]+=dA2dz[2]*(1-cangle)+(area2)*(-DOT(dn2dz3,n1));
                
                
            }
            
            
    }
    
    free(A);
    (*res)=result/TA;
    if(D==NULL)
    {
        for(int j=0;j<nvert;j++)
        {
           
            drdx[j]=(dresdx[j]*TA-result*dTAdx[j])/pow(TA,2);
            drdy[j]=(dresdy[j]*TA-result*dTAdy[j])/pow(TA,2);
            drdz[j]=(dresdz[j]*TA-result*dTAdz[j])/pow(TA,2);
        }
        set_submatrix(drdv,1,3*dn+3,drdx,1,dn,0,0);
        set_submatrix(drdv,1,3*dn+3,drdy,1,dn,0,dn);
        set_submatrix(drdv,1,3*dn+3,drdz,1,dn,0,2*dn);
    }
    else
    {
        for(int j=0;j<nvert;j++)
        {
           
            dresdx[j]=(dresdx[j]*TA-result*dTAdx[j])/pow(TA,2);
            dresdy[j]=(dresdy[j]*TA-result*dTAdy[j])/pow(TA,2);
            dresdz[j]=(dresdz[j]*TA-result*dTAdz[j])/pow(TA,2);
        }
        matrix_prod(dresdx,1,nvert,D,dn,drdx);
        matrix_prod(dresdy,1,nvert,D,dn,drdy);
        matrix_prod(dresdz,1,nvert,D,dn,drdz);
        set_submatrix(drdv,1,3*dn+3,drdx,1,dn,0,0);
        set_submatrix(drdv,1,3*dn+3,drdy,1,dn,0,dn);
        set_submatrix(drdv,1,3*dn+3,drdz,1,dn,0,2*dn);
    } 
    free(drdx);
    free(drdy);
    free(drdz);
    free(dTAdx);
    free(dTAdy);
    free(dTAdz);
    free(dresdx);
    free(dresdy);
    free(dresdz);
    free(NumofBlocks);
    free(IndexofBlocks);
    
    
}

void dihedral_angle_reg(int *tlist,double *vlist,int nfac,int nvert,double *D,int dm,int dn,double *result,double *drsdv)
{
    
    double mul=2;
    double cos_max_angle=0;
    double amul=1;
    double p=1;
    if(INI_DIA_PARAMS)
    {
        mul=INI_DIA_PARAMS[2]; 
        cos_max_angle=cos(INI_DIA_PARAMS[1]*2*PI/360.0); 
        amul=INI_DIA_PARAMS[0];
    }
    *result=0;
   int nedge=nfac+nvert-2;
   double *res,*drdx,*drdy,*drdz;
   int *EV;
   res=allocate_zero(nedge,sizeof(double));
   drdx=allocate_zero(nedge*nvert,sizeof(double));
   drdy=allocate_zero(nedge*nvert,sizeof(double));
   drdz=allocate_zero(nedge*nvert,sizeof(double));

   if(res==NULL || drdx==NULL || drdy==NULL || drdz==NULL)
   {
       fprintf(stderr,"Calloc returned null\n");
       exit(1);
   }
   double *drsdx,*drsdy,*drsdz;
   
   double *dresdx,*dresdy,*dresdz;
   EV=allocate_raw(nvert*nvert*sizeof(int));
   dihedral_angle(tlist,vlist,nfac,nvert,res,EV,drdx,drdy,drdz);
   if(D!=NULL && dm!=nvert)
    {
        puts("Error: Number of vertex coordinates is not equal to the number of rows in D.");
        exit(1);
    }
    if(D==NULL)
        dn=nvert;
    drsdx=allocate_zero(nedge*dn,sizeof(double));
   drsdy=allocate_zero(nedge*dn,sizeof(double));
   drsdz=allocate_zero(nedge*dn,sizeof(double));
    if(D==NULL)
   for(int j=0;j<nedge;j++)
   {
       if(res[j]<cos_max_angle)
           p=amul;
       else
           p=1;
       (*result)+=p*pow(1-res[j],mul);
       for(int k=0;k<nvert;k++)
       {
           
           drsdx[k]+=-p*mul*pow(1-res[j],mul-1)*drdx[j*nvert+k];
           drsdy[k]+=-p*mul*pow(1-res[j],mul-1)*drdy[j*nvert+k];
           drsdz[k]+=-p*mul*pow(1-res[j],mul-1)*drdz[j*nvert+k];
       }
   }
   else
   {
       dresdx=allocate_zero(nvert,sizeof(double));
       dresdy=allocate_zero(nvert,sizeof(double));
       dresdz=allocate_zero(nvert,sizeof(double));
    for(int j=0;j<nedge;j++)
   {
       if(res[j]<cos_max_angle)
           p=amul;
       else
           p=1;
       (*result)+=p*pow(1-res[j],mul);
       for(int k=0;k<nvert;k++)
       {
          
           dresdx[k]+=-p*mul*pow(1-res[j],mul-1)*drdx[j*nvert+k];
           dresdy[k]+=-p*mul*pow(1-res[j],mul-1)*drdy[j*nvert+k];
           dresdz[k]+=-p*mul*pow(1-res[j],mul-1)*drdz[j*nvert+k];
       }
   }
   matrix_prod(dresdx,1,nvert,D,dn,drsdx);
   matrix_prod(dresdy,1,nvert,D,dn,drsdy);
   matrix_prod(dresdz,1,nvert,D,dn,drsdz);
   free(dresdx);
   free(dresdy);
   free(dresdz);
   }
   set_submatrix(drsdv,1,3*dn+3,drsdx,1,dn,0,0);
   set_submatrix(drsdv,1,3*dn+3,drsdy,1,dn,0,dn);
   set_submatrix(drsdv,1,3*dn+3,drsdz,1,dn,0,2*dn);
   free(drsdx);
   free(drsdy);
   free(drsdz);
   free(res);
   free(drdx);
   free(drdy);
   free(drdz);
   free(EV);
}

void dihedral_angle(int* tlist,double* vlist,int nfac,int nvert,double *res,int *EV,double *dresdx,double *dresdy,double *dresdz)
{
    
    
    int nedge;
    int *E,*N,*E2,*A;
    double *v1,*v2,*v3,n1[3],n2[3];
    double *w1,*w2,*w3;
    double dn1dx1[3],dn1dx2[3],dn1dx3[3];
    double dn1dy1[3],dn1dy2[3],dn1dy3[3];
    double dn1dz1[3],dn1dz2[3],dn1dz3[3];
    
    double dn2dx1[3],dn2dx2[3],dn2dx3[3];
    double dn2dy1[3],dn2dy2[3],dn2dy3[3];
    double dn2dz1[3],dn2dz2[3],dn2dz3[3];
    double area;
    double dAdx[3],dAdy[3],dAdz[3];
    E=allocate_zero(nvert*nvert,sizeof(int));
     N=allocate_zero(nvert*nfac,sizeof(int));
     E2=allocate_zero(nvert*nvert,sizeof(int));
     A=allocate_zero(nfac*nfac,sizeof(int));
    find_neighborhood(tlist,vlist,nfac,nvert,E,N,E2,A);
   
    nedge=nfac+nvert-2;
    int ind;
    int f1,f2;
    int i1,i2,i3;
    int j1,j2,j3;
    int count=0;
    
    for(int j=0;j<nvert*nvert;j++)
        EV[j]=-1;
    
    for(int j=0;j<nvert;j++)
        for(int k=j;k<nvert;k++)
        {
            if(get_elI(E,nvert,nvert,j,k)==0)
                continue;
            f1=get_elI(E2,nvert,nvert,j,k);
            f2=get_elI(E2,nvert,nvert,k,j);
           
         
            set_elI(EV,nvert,nvert,count,j,k);
            set_elI(EV,nvert,nvert,count,k,j);
            
            
            i1=tlist[3*f1]-1;
            i2=tlist[3*f1+1]-1;
            i3=tlist[3*f1+2]-1;
            j1=tlist[3*f2]-1;
            j2=tlist[3*f2+1]-1;
            j3=tlist[3*f2+2]-1;
            v1=vlist+3*i1;
            v2=vlist+3*i2;
            v3=vlist+3*i3;
            w1=vlist+3*j1;
            w2=vlist+3*j2;
            w3=vlist+3*j3;
            Calculate_Area_and_Normal_Derivative(v1,v2,v3,n1,dn1dx1,dn1dx2,dn1dx3,dn1dy1,dn1dy2,dn1dy3,dn1dz1,dn1dz2,dn1dz3,&area,dAdx,dAdy,dAdz);
            Calculate_Area_and_Normal_Derivative(w1,w2,w3,n2,dn2dx1,dn2dx2,dn2dx3,dn2dy1,dn2dy2,dn2dy3,dn2dz1,dn2dz2,dn2dz3,&area,dAdx,dAdy,dAdz);
            
            res[count]=DOT(n1,n2);
           
            dresdx[count*nvert+i1]=DOT(dn1dx1,n2);
            dresdy[count*nvert+i1]=DOT(dn1dy1,n2);
            dresdz[count*nvert+i1]=DOT(dn1dz1,n2);
            
            dresdx[count*nvert+i2]=DOT(dn1dx2,n2);
            dresdy[count*nvert+i2]=DOT(dn1dy2,n2);
            dresdz[count*nvert+i2]=DOT(dn1dz2,n2);
            
            dresdx[count*nvert+i3]=DOT(dn1dx3,n2);
            dresdy[count*nvert+i3]=DOT(dn1dy3,n2);
            dresdz[count*nvert+i3]=DOT(dn1dz3,n2);
            
            dresdx[count*nvert+j1]+=DOT(dn2dx1,n1);
            dresdy[count*nvert+j1]+=DOT(dn2dy1,n1);
            dresdz[count*nvert+j1]+=DOT(dn2dz1,n1);
            
            dresdx[count*nvert+j2]+=DOT(dn2dx2,n1);
            dresdy[count*nvert+j2]+=DOT(dn2dy2,n1);
            dresdz[count*nvert+j2]+=DOT(dn2dz2,n1);
            
            dresdx[count*nvert+j3]+=DOT(dn2dx3,n1);
            dresdy[count*nvert+j3]+=DOT(dn2dy3,n1);
            dresdz[count*nvert+j3]+=DOT(dn2dz3,n1);
            count++;
        }
        free(E);
        free(E2);
        free(N);
        free(A);
}

double center_of_mass(int *tlist,double *vlist,int nfac,int nvert,double *D,int dm,int dn,double *dv,int deriv)
{
    
    double dndx1[3],dndx2[3],dndx3[3];
    double dndy1[3],dndy2[3],dndy3[3];
    double dndz1[3],dndz2[3],dndz3[3];
    double com[3];
    double normal[3],A=0,V=0;
    double cx=0.0,cy=0.0,cz=0.0;
    double dAdx[3],dAdy[3],dAdz[3];
    double *dVdx,*dVdy,*dVdz;
    double dpxdx1,dpxdx2,dpxdx3;
    double dpydy1,dpydy2,dpydy3;
    double dpzdz1,dpzdz2,dpzdz3;
    dVdx=allocate_zero(nvert,sizeof(double));
    dVdy=allocate_zero(nvert,sizeof(double));
    dVdz=allocate_zero(nvert,sizeof(double));
    double *dcdx,*dcdy,*dcdz;
    dcdx=allocate_zero(3*nvert,sizeof(double));
    dcdy=allocate_zero(3*nvert,sizeof(double));
    dcdz=allocate_zero(3*nvert,sizeof(double));
    double *v1,*v2,*v3;
    double tpx,tpy,tpz;
    int i1,i2,i3;
    for(int j=0;j<nfac;j++)
    {
        i1=tlist[3*j]-1;
        i2=tlist[3*j+1]-1;
        i3=tlist[3*j+2]-1;
        v1=vlist+3*i1;
        v2=vlist+3*i2;
        v3=vlist+3*i3;
        Calculate_Area_and_Normal_Derivative(v1,v2,v3,normal,dndx1,dndx2,dndx3,dndy1,dndy2,dndy3,dndz1,dndz2,dndz3,&A,dAdx,dAdy,dAdz);
        V=V+1.0/3.0*(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*A;
        tpx=(pow(v1[0]+v2[0],2)+pow(v2[0]+v3[0],2)+pow(v1[0]+v3[0],2));
        tpy=(pow(v1[1]+v2[1],2)+pow(v2[1]+v3[1],2)+pow(v1[1]+v3[1],2));
        tpz=(pow(v1[2]+v2[2],2)+pow(v2[2]+v3[2],2)+pow(v1[2]+v3[2],2));
        cx=cx+2*A*normal[0]*tpx;
        cy=cy+2*A*normal[1]*tpy;
        cz=cz+2*A*normal[2]*tpz;
        if(deriv>0)
        {
        dVdx[i1]+=1.0/3*((normal[0]+v1[0]*dndx1[0]+v1[1]*dndx1[1]+v1[2]*dndx1[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdx[0]);
        dVdx[i2]+=1.0/3*((v1[0]*dndx2[0]+v1[1]*dndx2[1]+v1[2]*dndx2[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdx[1]);
        dVdx[i3]+=1.0/3*((v1[0]*dndx3[0]+v1[1]*dndx3[1]+v1[2]*dndx3[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdx[2]);
        
        dVdy[i1]+=1.0/3*((normal[1]+v1[0]*dndy1[0]+v1[1]*dndy1[1]+v1[2]*dndy1[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdy[0]);
        dVdy[i2]+=1.0/3*((v1[0]*dndy2[0]+v1[1]*dndy2[1]+v1[2]*dndy2[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdy[1]);
        dVdy[i3]+=1.0/3*((v1[0]*dndy3[0]+v1[1]*dndy3[1]+v1[2]*dndy3[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdy[2]);
        
       
        dVdz[i1]+=1.0/3*((normal[2]+v1[0]*dndz1[0]+v1[1]*dndz1[1]+v1[2]*dndz1[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdz[0]);
        dVdz[i2]+=1.0/3*((v1[0]*dndz2[0]+v1[1]*dndz2[1]+v1[2]*dndz2[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdz[1]);
        dVdz[i3]+=1.0/3*((v1[0]*dndz3[0]+v1[1]*dndz3[1]+v1[2]*dndz3[2])*A+(v1[0]*normal[0]+v1[1]*normal[1]+v1[2]*normal[2])*dAdz[2]);
        
        
        
        dpxdx1=2*(v1[0]+v2[0])+2*(v1[0]+v3[0]);
        dpxdx2=2*(v1[0]+v2[0])+2*(v2[0]+v3[0]);
        dpxdx3=2*(v2[0]+v3[0])+2*(v1[0]+v3[0]);
        
        dpydy1=2*(v1[1]+v2[1])+2*(v1[1]+v3[1]);
        dpydy2=2*(v1[1]+v2[1])+2*(v2[1]+v3[1]);
        dpydy3=2*(v2[1]+v3[1])+2*(v1[1]+v3[1]);
        
        dpzdz1=2*(v1[2]+v2[2])+2*(v1[2]+v3[2]);
        dpzdz2=2*(v1[2]+v2[2])+2*(v2[2]+v3[2]);
        dpzdz3=2*(v2[2]+v3[2])+2*(v1[2]+v3[2]);
        
        
        
        dcdx[0*nvert+i1]+=(2*dAdx[0]*normal[0]+2*A*dndx1[0])*tpx+2*A*normal[0]*dpxdx1;
        dcdx[0*nvert+i2]+=(2*dAdx[1]*normal[0]+2*A*dndx2[0])*tpx+2*A*normal[0]*dpxdx2;
        dcdx[0*nvert+i3]+=(2*dAdx[2]*normal[0]+2*A*dndx3[0])*tpx+2*A*normal[0]*dpxdx3;
        
        dcdx[1*nvert+i1]+=(2*dAdx[0]*normal[1]+2*A*dndx1[1])*tpy;
        dcdx[1*nvert+i2]+=(2*dAdx[1]*normal[1]+2*A*dndx2[1])*tpy;
        dcdx[1*nvert+i3]+=(2*dAdx[2]*normal[1]+2*A*dndx3[1])*tpy;

        dcdx[2*nvert+i1]+=(2*dAdx[0]*normal[2]+2*A*dndx1[2])*tpz;
        dcdx[2*nvert+i2]+=(2*dAdx[1]*normal[2]+2*A*dndx2[2])*tpz;
        dcdx[2*nvert+i3]+=(2*dAdx[2]*normal[2]+2*A*dndx3[2])*tpz;
        
        dcdy[0*nvert+i1]+=(2*dAdy[0]*normal[0]+2*A*dndy1[0])*tpx;
        dcdy[0*nvert+i2]+=(2*dAdy[1]*normal[0]+2*A*dndy2[0])*tpx;
        dcdy[0*nvert+i3]+=(2*dAdy[2]*normal[0]+2*A*dndy3[0])*tpx;
        
        dcdy[1*nvert+i1]+=(2*dAdy[0]*normal[1]+2*A*dndy1[1])*tpy+2*A*normal[1]*dpydy1;
        dcdy[1*nvert+i2]+=(2*dAdy[1]*normal[1]+2*A*dndy2[1])*tpy+2*A*normal[1]*dpydy2;
        dcdy[1*nvert+i3]+=(2*dAdy[2]*normal[1]+2*A*dndy3[1])*tpy+2*A*normal[1]*dpydy3;
        
        dcdy[2*nvert+i1]+=(2*dAdy[0]*normal[2]+2*A*dndy1[2])*tpz;
        dcdy[2*nvert+i2]+=(2*dAdy[1]*normal[2]+2*A*dndy2[2])*tpz;
        dcdy[2*nvert+i3]+=(2*dAdy[2]*normal[2]+2*A*dndy3[2])*tpz;
        
       
        dcdz[0*nvert+i1]+=(2*dAdz[0]*normal[0]+2*A*dndz1[0])*tpx;
        dcdz[0*nvert+i2]+=(2*dAdz[1]*normal[0]+2*A*dndz2[0])*tpx;
        dcdz[0*nvert+i3]+=(2*dAdz[2]*normal[0]+2*A*dndz3[0])*tpx;
        
        dcdz[1*nvert+i1]+=(2*dAdz[0]*normal[1]+2*A*dndz1[1])*tpy;
        dcdz[1*nvert+i2]+=(2*dAdz[1]*normal[1]+2*A*dndz2[1])*tpy;
        dcdz[1*nvert+i3]+=(2*dAdz[2]*normal[1]+2*A*dndz3[1])*tpy;
        
        dcdz[2*nvert+i1]+=(2*dAdz[0]*normal[2]+2*A*dndz1[2])*tpz+2*A*normal[2]*dpzdz1;
        dcdz[2*nvert+i2]+=(2*dAdz[1]*normal[2]+2*A*dndz2[2])*tpz+2*A*normal[2]*dpzdz2;
        dcdz[2*nvert+i3]+=(2*dAdz[2]*normal[2]+2*A*dndz3[2])*tpz+2*A*normal[2]*dpzdz3;
        }
    }
    
    com[0]=1.0/48.0*1.0/V*cx;
    com[1]=1.0/48.0*1.0/V*cy;
    com[2]=1.0/48.0*1.0/V*cz;
    double V2=pow(V,2),C;
    C=NORM(com);
    if(deriv>0)
        if(D==NULL)
        {
            for(int j=0;j<nvert;j++)
            {
                dcdx[0*nvert+j]=1.0/48.0*(V*dcdx[0*nvert+j]-dVdx[j]*cx)/V2;
                dcdx[1*nvert+j]=1.0/48.0*(V*dcdx[1*nvert+j]-dVdx[j]*cy)/V2;
                dcdx[2*nvert+j]=1.0/48.0*(V*dcdx[2*nvert+j]-dVdx[j]*cz)/V2;
                
                dcdy[0*nvert+j]=1.0/48.0*(V*dcdy[0*nvert+j]-dVdy[j]*cx)/V2;
                dcdy[1*nvert+j]=1.0/48.0*(V*dcdy[1*nvert+j]-dVdy[j]*cy)/V2;
                dcdy[2*nvert+j]=1.0/48.0*(V*dcdy[2*nvert+j]-dVdy[j]*cz)/V2;
                
                dcdz[0*nvert+j]=1.0/48.0*(V*dcdz[0*nvert+j]-dVdz[j]*cx)/V2;
                dcdz[1*nvert+j]=1.0/48.0*(V*dcdz[1*nvert+j]-dVdz[j]*cy)/V2;
                dcdz[2*nvert+j]=1.0/48.0*(V*dcdz[2*nvert+j]-dVdz[j]*cz)/V2;
                
                dv[j]=1/C*(com[0]*dcdx[j]+com[1]*dcdx[nvert+j]+com[2]*dcdx[2*nvert+j]);
                dv[nvert+j]=1/C*(com[0]*dcdy[j]+com[1]*dcdy[nvert+j]+com[2]*dcdy[2*nvert+j]);
                dv[2*nvert+j]=1/C*(com[0]*dcdz[j]+com[1]*dcdz[nvert+j]+com[2]*dcdz[2*nvert+j]);
            }
            if(C<1e-2)
            {
                C=0;
                zero_array(dv,3*nvert+3);
                
            }
        }
        else
        {
            double *dvdx=allocate_zero(nvert,sizeof(double));
            double *dvdy=allocate_zero(nvert,sizeof(double));
            double *dvdz=allocate_zero(nvert,sizeof(double));
            double *dvdt=allocate_zero(dn,sizeof(double));
            for(int j=0;j<nvert;j++)
            {
                dcdx[0*nvert+j]=1.0/48.0*(V*dcdx[0*nvert+j]-dVdx[j]*cx)/V2;
                dcdx[1*nvert+j]=1.0/48.0*(V*dcdx[1*nvert+j]-dVdx[j]*cy)/V2;
                dcdx[2*nvert+j]=1.0/48.0*(V*dcdx[2*nvert+j]-dVdx[j]*cz)/V2;
                
                dcdy[0*nvert+j]=1.0/48.0*(V*dcdy[0*nvert+j]-dVdy[j]*cx)/V2;
                dcdy[1*nvert+j]=1.0/48.0*(V*dcdy[1*nvert+j]-dVdy[j]*cy)/V2;
                dcdy[2*nvert+j]=1.0/48.0*(V*dcdy[2*nvert+j]-dVdy[j]*cz)/V2;
                
                dcdz[0*nvert+j]=1.0/48.0*(V*dcdz[0*nvert+j]-dVdz[j]*cx)/V2;
                dcdz[1*nvert+j]=1.0/48.0*(V*dcdz[1*nvert+j]-dVdz[j]*cy)/V2;
                dcdz[2*nvert+j]=1.0/48.0*(V*dcdz[2*nvert+j]-dVdz[j]*cz)/V2;
                
                dvdx[j]=1/C*(com[0]*dcdx[j]+com[1]*dcdx[nvert+j]+com[2]*dcdx[2*nvert+j]);
                dvdy[j]=1/C*(com[0]*dcdy[j]+com[1]*dcdy[nvert+j]+com[2]*dcdy[2*nvert+j]);
                dvdz[j]=1/C*(com[0]*dcdz[j]+com[1]*dcdz[nvert+j]+com[2]*dcdz[2*nvert+j]);
            }
            matrix_prod(dvdx,1,nvert,D,dn,dvdt);
            set_submatrix(dv,1,3*dn+3,dvdt,1,dn,0,0);
            matrix_prod(dvdy,1,nvert,D,dn,dvdt);
            set_submatrix(dv,1,3*dn+3,dvdt,1,dn,0,dn);
            matrix_prod(dvdz,1,nvert,D,dn,dvdt);
            set_submatrix(dv,1,3*dn+3,dvdt,1,dn,0,2*dn);
            free(dvdx);
            free(dvdy);
            free(dvdz);
            free(dvdt);
            if(C<1e-2)
            {
                C=0;
                zero_array(dv,3*dn+3);
                
            }
        }
        
        free(dcdx);
        free(dcdy);
        free(dcdz);
        free(dVdy);
        free(dVdx);
        free(dVdz);
        
return C;
}
extern "C" __declspec(dllexport) void triangle_fourier(double*u,double*v,int n,double*a,double*out,double*deriv){
 Complex *f=new Complex[n],*d[6];for(int j=0;j<6;j++)d[j]=new Complex[n];
 Calc_FTC_deriv(u,v,n,a[0],a[1],a[2],a[3],a[4],a[5],f,d[0],d[1],d[2],d[3],d[4],d[5]);
 for(int i=0;i<n;i++){out[2*i]=f[i].real();out[2*i+1]=f[i].imag();for(int j=0;j<6;j++){deriv[12*i+2*j]=d[j][i].real();deriv[12*i+2*j+1]=d[j][i].imag();}}
 delete[]f;for(int j=0;j<6;j++)delete[]d[j];
}
extern "C" __declspec(dllexport) void ao_forward(int*faces,double*mesh,int nf,int nv,double*angles,double*observer,double*sun,double*up,double time,double distance,double*u,double*v,int n,double*offset,int derivative,double*fr,double*fi,double*dxr,double*dxi,double*dyr,double*dyi,double*dzr,double*dzi,double*dr,double*di,double*dor,double*doi){
 if(!derivative){Calculate_AO(faces,mesh,nf,nv,angles,observer,sun,up,time,distance,u,v,n,offset,fr,fi,nullptr,nullptr);return;}
 Calculate_AO_deriv(faces,mesh,nf,nv,angles,observer,sun,up,time,distance,u,v,n,offset,fr,fi,dxr,dxi,dyr,dyi,dzr,dzi,dr,di,dor,doi,nullptr,nullptr,nullptr,nullptr);
}
extern "C" __declspec(dllexport) void lightcurve_forward(int*faces,double*mesh,int nf,int nv,double*angles,double*observer,double*sun,int n,double*time,int relative,double*brightness,double*dx,double*dy,double*dz,double*db,double*dl,double*dw){
 std::vector<double> unused_phase_derivative(3*n,0.0);
 calculate_lcurve(faces,mesh,nf,nv,angles,observer,sun,n,time,brightness,dx,dy,dz,db,dl,dw,unused_phase_derivative.data(),nullptr,nullptr,nullptr,relative,nullptr);
}
extern "C" __declspec(dllexport) void occultation_forward(int*faces,double*mesh,int nf,int nv,double*angles,double*up,double*observer,double*velocity,double*time,double*offset,double*chords,int*type,int n,double*residual,double*dx,double*dy,double*dz,double*da,double*doffx,double*doffy){
 std::vector<double> fixed_chord_offsets(n,0.0),unused_chord_derivatives(4*n*n,0.0);
 Fit_Occ(faces,mesh,nf,nv,angles,up,observer,velocity,time,offset,chords,type,n,nullptr,fixed_chord_offsets.data(),residual,dx,dy,dz,da,doffx,doffy,unused_chord_derivatives.data());
}
extern "C" __declspec(dllexport) void mesh_penalties(int*faces,double*mesh,int nf,int nv,double*convexity,double*dihedral,double*com,double*dc,double*dd,double*dm){
 convex_reg(faces,mesh,nf,nv,nullptr,nv,nv,convexity,dc);
 dihedral_angle_reg(faces,mesh,nf,nv,nullptr,nv,nv,dihedral,dd);
 *com=center_of_mass(faces,mesh,nf,nv,nullptr,nv,nv,dm,1);
}

'''


@lru_cache(maxsize=1)
def _math_library():
    """Compile explicit math in a source-hashed temporary cache with existing MSVC.

    This documented environment exception uses C++17 only; no domain libraries,
    downloads, upstream executables, or source paths are used by this module.
    Static /MT runtime avoids a private compiler-runtime DLL dependency.
    """
    import ctypes,hashlib,json,os,subprocess,tempfile
    from pathlib import Path
    digest=hashlib.sha256(CPP_SOURCE.encode('utf8')).hexdigest()
    cache=Path(os.environ.get('HERTHA_MATH_CACHE',str(Path(tempfile.gettempdir())/'imaging101-hertha-math')))/digest
    cache.mkdir(parents=True,exist_ok=True);dll=cache/'math.dll'
    if not dll.exists():
        vc=Path(os.environ.get('HERTHA_MSVC_ROOT','C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133'))
        sdk=Path(os.environ.get('HERTHA_SDK_ROOT','C:/Program Files (x86)/Windows Kits/10'))
        version=os.environ.get('HERTHA_SDK_VERSION','10.0.19041.0')
        compiler=vc/'bin/Hostx64/x64/cl.exe'
        if not compiler.is_file():raise RuntimeError('The documented existing x64 MSVC C++17 toolchain is unavailable')
        env=dict(os.environ,INCLUDE=';'.join(map(str,[vc/'include',sdk/'Include'/version/'ucrt',sdk/'Include'/version/'shared',sdk/'Include'/version/'um'])),LIB=';'.join(map(str,[vc/'lib/x64',sdk/'Lib'/version/'ucrt/x64',sdk/'Lib'/version/'um/x64'])))
        (cache/'math.cpp').write_text(CPP_SOURCE,encoding='utf8')
        command=[str(compiler),'/nologo','/LD','/O2','/fp:precise','/EHsc','/std:c++17','/MT','/W3','math.cpp','/link','/OUT:math.dll']
        with (cache/'compiler.stdout.log').open('wb') as stdout,(cache/'compiler.stderr.log').open('wb') as stderr:
            result=subprocess.run(command,cwd=cache,env=env,stdout=stdout,stderr=stderr,shell=False)
        (cache/'build.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,source_sha256=digest,include=env['INCLUDE'],library_paths=env['LIB']),indent=2),encoding='utf8')
        if result.returncode:raise RuntimeError(f'Local numerical-kernel compilation failed; logs: {cache}')
    library=ctypes.CDLL(str(dll));dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS');i=ctypes.c_int;d=ctypes.c_double
    library.triangle_fourier.argtypes=[dp,dp,i,dp,dp,dp]
    library.ao_forward.argtypes=[ip,dp,i,i,dp,dp,dp,dp,d,d,dp,dp,i,dp,i]+[dp]*12
    library.lightcurve_forward.argtypes=[ip,dp,i,i,dp,dp,dp,i,dp,i]+[dp]*7
    library.occultation_forward.argtypes=[ip,dp,i,i]+[dp]*7+[ip,i]+[dp]*7
    library.mesh_penalties.argtypes=[ip,dp,i,i]+[dp]*6
    for name in ['triangle_fourier','ao_forward','lightcurve_forward','occultation_forward','mesh_penalties']:getattr(library,name).restype=None
    return library


def triangle_fourier(frequencies,vertices):
    """Complex integral over an oriented 2D triangle and its six vertex derivatives."""
    uv=np.asarray(frequencies,dtype=np.float64).reshape(-1,2);n=len(uv)
    output=np.empty((n,2));derivative=np.empty((n,6,2))
    _math_library().triangle_fourier(uv[:,0].copy(),uv[:,1].copy(),n,np.ascontiguousarray(vertices,dtype=np.float64).reshape(6),output,derivative)
    return output[:,0]+1j*output[:,1],derivative[:,:,0]+1j*derivative[:,:,1]


def _native_mesh(mesh):
    vertices=np.ascontiguousarray(mesh,dtype=np.float64)
    if vertices.shape!=(402,3) or not np.isfinite(vertices).all():raise ValueError('Expected finite native402x3 mesh in km')
    return np.ascontiguousarray(sphere_grid(10)[2]+1,dtype=np.int32),vertices


def ao_operator(mesh,angles,observer,sun,up,time,distance,frequencies,offset,derivative=True):
    """Normalized unblurred AO Fourier field and analytic mesh/spin/shift derivatives.

    Frequencies cycles/arcsec; offset arcsec; distance AU; time days from epoch.
    Derivative blocks are real/imag pairs for x,y,z vertex blocks,3spin,2shift.
    """
    faces,mesh=_native_mesh(mesh);uv=np.asarray(frequencies,dtype=np.float64);n=len(uv)
    size=n if derivative else 1
    arrays=[np.zeros((n,)),np.zeros((n,))]
    arrays.extend(np.zeros((size,402)) for _ in range(6))
    arrays.extend(np.zeros((size,3)) for _ in range(2))
    arrays.extend(np.zeros((size,2)) for _ in range(2))
    vector=lambda x:np.ascontiguousarray(x,dtype=np.float64)
    _math_library().ao_forward(faces,mesh,800,402,vector(angles),vector(observer),vector(sun),vector(up),float(time),float(distance),uv[:,0].copy(),uv[:,1].copy(),n,vector(offset),int(derivative),*arrays)
    return arrays[0]+1j*arrays[1],tuple(arrays[2:]) if derivative else None


def lightcurve_operator(mesh,angles,observer,sun,time,relative=True):
    """LS/Lambert facet brightness, normalized per relative lightcurve, with derivatives."""
    faces,mesh=_native_mesh(mesh);n=len(time);vector=lambda x:np.ascontiguousarray(x,dtype=np.float64)
    output=np.zeros(n);dx=np.zeros((n,402));dy=np.zeros_like(dx);dz=np.zeros_like(dx)
    db=np.zeros(n);dl=np.zeros(n);dw=np.zeros(n)
    _math_library().lightcurve_forward(faces,mesh,800,402,vector(angles),vector(observer),vector(sun),n,vector(time),int(relative),output,dx,dy,dz,db,dl,dw)
    return output,(dx,dy,dz,np.stack([db,dl,dw],axis=1))


def occultation_operator(mesh,angles,up,observer,velocity,time,offset,chords,chord_type):
    """Author four-endpoint residual per chord (km), with fixed native limb convention."""
    faces,mesh=_native_mesh(mesh);n=len(chords);vector=lambda x:np.ascontiguousarray(x,dtype=np.float64)
    output=np.zeros(4*n);dx=np.zeros((4*n,402));dy=np.zeros_like(dx);dz=np.zeros_like(dx);da=np.zeros((4*n,3));dox=np.zeros(4*n);doy=np.zeros(4*n)
    _math_library().occultation_forward(faces,mesh,800,402,vector(angles),vector(up),vector(observer),vector(velocity),vector(time),vector(offset),vector(chords),np.ascontiguousarray(chord_type,dtype=np.int32),n,output,dx,dy,dz,da,dox,doy)
    return output,(dx,dy,dz,da,np.stack([dox,doy],axis=1))


def mesh_penalties(mesh):
    """Convexity, dihedral and COM scalar residuals with Cartesian derivatives."""
    faces,mesh=_native_mesh(mesh);values=[np.zeros(1) for _ in range(3)];derivatives=[np.zeros(1209) for _ in range(3)]
    _math_library().mesh_penalties(faces,mesh,800,402,*values,*derivatives)
    return np.array([x[0] for x in values]),np.stack(derivatives)

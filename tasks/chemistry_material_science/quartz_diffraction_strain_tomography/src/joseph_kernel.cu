/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
struct Geo {float rx,ry,sx,sy,ux,uy;};
namespace astraCUDA3d {struct SDimensions3D {int iVolX,iVolY,iVolZ,iProjU,iProjV,iProjAngles;};}
namespace astraCUDA3d {

static const unsigned int g_anglesPerBlock = 4;

// thickness of the slices we're splitting the volume up into
static const unsigned int g_blockSlices = 32;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

static const unsigned g_MaxAngles = 1024;
__constant__ float gC_RayX[g_MaxAngles];
__constant__ float gC_RayY[g_MaxAngles];
__constant__ float gC_RayZ[g_MaxAngles];
__constant__ float gC_DetSX[g_MaxAngles];
__constant__ float gC_DetSY[g_MaxAngles];
__constant__ float gC_DetSZ[g_MaxAngles];
__constant__ float gC_DetUX[g_MaxAngles];
__constant__ float gC_DetUY[g_MaxAngles];
__constant__ float gC_DetUZ[g_MaxAngles];
__constant__ float gC_DetVX[g_MaxAngles];
__constant__ float gC_DetVY[g_MaxAngles];
__constant__ float gC_DetVZ[g_MaxAngles];


// x=0, y=1, z=2
struct DIR_X {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return x; }
	__device__ float c1(float x, float y, float z) const { return y; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f0, f1, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f0; }
	__device__ float y(float f0, float f1, float f2) const { return f1; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
};

// y=0, x=1, z=2
struct DIR_Y {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return y; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f1, f0, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f0; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
};

// z=0, x=1, y=2
struct DIR_Z {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float c0(float x, float y, float z) const { return z; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return y; }
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f1, f2, f0); }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f2; }
	__device__ float z(float f0, float f1, float f2) const { return f0; }
};

struct SCALE_CUBE {
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1+a2*a2+1.0f) * fOutputScale; }
};

struct SCALE_NONCUBE {
	float fScale1;
	float fScale2;
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1*fScale1+a2*a2*fScale2+1.0f) * fOutputScale; }
};

// threadIdx: x = u detector
//            y = relative angle
// blockIdx:  x = u/v detector
//            y = angle block


template<class COORD, class SCALE>
__global__ void par3D_FP_t(float* D_projData, unsigned int projPitch,
                           cudaTextureObject_t tex,
                           unsigned int startSlice,
                           unsigned int startAngle, unsigned int endAngle,
                           const SDimensions3D dims,
                           SCALE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fRayX = gC_RayX[angle];
	const float fRayY = gC_RayY[angle];
	const float fRayZ = gC_RayZ[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];
	const float fDetUZ = gC_DetUZ[angle];
	const float fDetVX = gC_DetVX[angle];
	const float fDetVY = gC_DetVY[angle];
	const float fDetVZ = gC_DetVZ[angle];
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

	const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float fDistCorr = sc.scale(a1, a2);


	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	if (detectorU >= dims.iProjU)
		return;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* X = startSlice to X = endSlice                            */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   ( 1)       ( 0)    */
		/* ray:   (y) = (ay) * x + (by)    */
		/*        (z)   (az)       (bz)    */

		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += c.tex(tex, f0, f1, f2);
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fVal *= fDistCorr;

		D_projData[(size_t)(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}


}
extern "C" __declspec(dllexport) int project(const float* images,const float* angles,float* output,int m,int n,int channels,int nang,int rays){
 cudaChannelFormatDesc desc=cudaCreateChannelDesc<float>();cudaArray_t array;cudaExtent extent=make_cudaExtent(n,m,channels);
 cudaError_t e=cudaMalloc3DArray(&array,&desc,extent);if(e!=cudaSuccess)return (int)e;
 cudaMemcpy3DParms copy={};copy.srcPtr=make_cudaPitchedPtr((void*)images,n*sizeof(float),n,m);copy.dstArray=array;copy.extent=extent;copy.kind=cudaMemcpyHostToDevice;
 e=cudaMemcpy3D(&copy);if(e!=cudaSuccess)return (int)e;
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
 cudaTextureDesc td={};td.addressMode[0]=td.addressMode[1]=td.addressMode[2]=cudaAddressModeBorder;td.filterMode=cudaFilterModeLinear;td.readMode=cudaReadModeElementType;td.normalizedCoords=0;
 cudaTextureObject_t texture;e=cudaCreateTextureObject(&texture,&resource,&td,NULL);if(e!=cudaSuccess)return(int)e;
 std::vector<Geo> host(nang);
 for(int i=0;i<nang;++i){float c=cos(angles[i]),s=sin(angles[i]);host[i]={-s,c,(float)(-rays*.5*(double)c),(float)(-rays*.5*(double)s),c,s};}
 std::vector<float> temp(nang);
#define CP(NAME,EXPR) for(int i=0;i<nang;++i)temp[i]=(EXPR);cudaMemcpyToSymbol(astraCUDA3d::gC_##NAME,temp.data(),nang*sizeof(float));
 CP(RayX,host[i].rx);CP(RayY,host[i].ry);CP(RayZ,0);CP(DetSX,host[i].sx);CP(DetSY,host[i].sy);CP(DetSZ,-channels*.5f);CP(DetUX,host[i].ux);CP(DetUY,host[i].uy);CP(DetUZ,0);CP(DetVX,0);CP(DetVY,0);CP(DetVZ,1);
#undef CP
 float* out;size_t bytes=(size_t)channels*nang*rays*sizeof(float);cudaMalloc(&out,bytes);cudaMemset(out,0,bytes);
 astraCUDA3d::SDimensions3D dims;dims.iVolX=n;dims.iVolY=m;dims.iVolZ=channels;dims.iProjU=rays;dims.iProjV=channels;dims.iProjAngles=nang;astraCUDA3d::SCALE_CUBE scale;scale.fOutputScale=1;
 for(int first=0;first<nang;){bool alongy=fabs(host[first].ry)>=fabs(host[first].rx);int end=first+1;while(end<nang && (fabs(host[end].ry)>=fabs(host[end].rx))==alongy)++end;
  for(int start=0;start<(alongy?m:n);start+=32){dim3 grid(((rays+31)/32)*((channels+31)/32),(end-first+3)/4);dim3 block(32,4);if(alongy)astraCUDA3d::par3D_FP_t<astraCUDA3d::DIR_Y,astraCUDA3d::SCALE_CUBE><<<grid,block>>>(out,rays,texture,start,first,end,dims,scale);else astraCUDA3d::par3D_FP_t<astraCUDA3d::DIR_X,astraCUDA3d::SCALE_CUBE><<<grid,block>>>(out,rays,texture,start,first,end,dims,scale);}
 first=end;}
 e=cudaDeviceSynchronize();if(e==cudaSuccess)e=cudaMemcpy(output,out,bytes,cudaMemcpyDeviceToHost);
 cudaFree(out);cudaDestroyTextureObject(texture);cudaFreeArray(array);return(int)e;
}

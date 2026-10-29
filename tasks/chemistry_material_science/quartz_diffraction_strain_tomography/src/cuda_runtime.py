"""Compile the visible Joseph kernel with general CUDA runtime primitives.

No ASTRA package, DLL, or compiled domain operator is loaded. The generated DLL
lives in a private temporary directory and is unloaded before directory cleanup.
Current validated host: Windows, NVIDIA Ada GPU, CUDA 12.2, MSVC 14.29.
"""
import atexit,ctypes,os,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
_library=None
_directory=None
_dll_directory=None

def close_runtime():
    global _library,_directory,_dll_directory
    if _library is not None:
        import _ctypes
        _ctypes.FreeLibrary(_library._handle);_library=None
    if _dll_directory is not None:_dll_directory.close();_dll_directory=None
    if _directory is not None:
        path=Path(_directory).resolve();temporary=Path(tempfile.gettempdir()).resolve()
        if path.parent==temporary and path.name.startswith('imaging101-quartz-'):shutil.rmtree(path)
        _directory=None

def _load():
    global _library,_directory,_dll_directory
    if _library is not None:return _library
    nvcc=shutil.which('nvcc');cl=shutil.which('cl')
    if not nvcc or not cl:raise RuntimeError('CUDA nvcc and MSVC cl must be available on PATH')
    cuda=Path(nvcc).parent;msvc=Path(cl).parents[3]
    kit=Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Windows Kits/10'
    version='10.0.19041.0'
    includes=[msvc/'include',kit/'Include'/version/'ucrt',kit/'Include'/version/'shared',kit/'Include'/version/'um']
    libraries=[msvc/'lib/x64',kit/'Lib'/version/'ucrt/x64',kit/'Lib'/version/'um/x64']
    if not all(p.exists() for p in includes+libraries):raise RuntimeError('Validated MSVC and Windows SDK 10.0.19041.0 headers/libraries are required')
    env=os.environ.copy();env['INCLUDE']=';'.join(str(p) for p in includes);env['LIB']=';'.join(str(p) for p in libraries)
    _directory=tempfile.mkdtemp(prefix='imaging101-quartz-');target=Path(_directory)/'joseph.dll'
    source=Path(__file__).with_name('joseph_kernel.cu')
    command=[nvcc,'--shared','-O3','-arch=sm_89','-Xcompiler','/O2,/Oi',str(source),'-o',str(target)]
    completed=subprocess.run(command,cwd=_directory,env=env,capture_output=True,text=True,errors='replace')
    if completed.returncode:raise RuntimeError('Visible CUDA kernel compilation failed: '+completed.stdout+completed.stderr)
    _dll_directory=os.add_dll_directory(str(cuda));_library=ctypes.CDLL(str(target))
    pointer=np.ctypeslib.ndpointer(dtype='float32',flags='C_CONTIGUOUS')
    _library.project.argtypes=[pointer,pointer,pointer]+[ctypes.c_int]*5;_library.project.restype=ctypes.c_int
    return _library

def project(images,angles,rays=192):
    images=np.ascontiguousarray(images,dtype=np.float32);angles=np.ascontiguousarray(np.radians(angles),dtype=np.float32)
    if images.ndim!=3 or angles.ndim!=1 or len(angles)>1024 or not np.isfinite(images).all() or not np.isfinite(angles).all():raise ValueError('invalid projection arrays')
    channels,m,n=images.shape;output=np.zeros((channels,len(angles),rays),dtype=np.float32)
    result=_load().project(images,angles,output,m,n,channels,len(angles),rays)
    if result:raise RuntimeError('CUDA runtime returned error '+str(result))
    return output

atexit.register(close_runtime)

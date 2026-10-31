from pathlib import Path
import numpy as np
from src.visualization import component_metrics
from src import cuda_runtime

def test_visualization_metrics_known_case():
    reference=np.tile(np.array([1.,2.,3.]),(6,1));metrics=component_metrics(reference*2,reference)
    np.testing.assert_allclose(metrics[:,0],1,atol=1e-15);np.testing.assert_allclose(metrics[:,1],np.sqrt(14/3)/2,atol=1e-15)

def test_visible_kernel_temporary_dll_lifecycle():
    result=cuda_runtime.project(np.ones((1,8,10)),np.array([0.]),10)
    directory=Path(cuda_runtime._directory);assert directory.exists() and directory.name.startswith('imaging101-quartz-')
    assert Path(cuda_runtime._library._name).parent==directory
    cuda_runtime.close_runtime();assert not directory.exists() and cuda_runtime._library is None
    repeated=cuda_runtime.project(np.ones((1,8,10)),np.array([0.]),10);np.testing.assert_array_equal(result,repeated)

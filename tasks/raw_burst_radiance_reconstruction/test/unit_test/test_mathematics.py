import numpy as np
from src.physics_model import subPixelMinimum,computeDistance,sample_forward,sample_adjoint
from src.preprocessing import downsample,getTiles
from src.solvers import patchesRMS,mergeWienerDFTPatches,spatialDenoisePatches,cosineWindow2Dpatches,depatchifyOverlap,alignOnALevel
def test_quadratic_optimum_true_two_dimensional_neighborhood():
    yy,xx=np.indices((9,9));d=((yy-4+.3)**2+2*(xx-4-.2)**2).astype('float32')
    offset=subPixelMinimum(d.reshape(1,81),np.array([40],dtype='uint16'))
    np.testing.assert_allclose(offset,[[-.3,.2]],atol=1e-6)
def test_large_integer_rms_and_bayer_average_do_not_overflow():
    x=np.array([[[[300,400],[300,400]]]],dtype='uint16')
    assert np.isclose(patchesRMS(x).item(),np.sqrt(125000))
    np.testing.assert_array_equal(downsample(np.full((4,4),60000,dtype='uint16'),kernel='bayer'),60000)
def test_known_translation_and_sensor_parity():
    rng=np.random.default_rng(27);a=rng.integers(50,900,(64,64),dtype='uint16');b=np.roll(a,(2,-2),(0,1))
    motion=alignOnALevel(a,b,{'verbose':0},tileSize=16,searchRadius=4,distance='L1',subpixel=False)
    np.testing.assert_array_equal(motion[1:-1,1:-1],np.broadcast_to([2,-2],motion[1:-1,1:-1].shape))
def test_discrete_pullback_adjoint_with_clipped_boundaries():
    rng=np.random.default_rng(28);x=rng.normal(size=(12,14));y=rng.normal(size=x.shape)
    assert np.isclose(np.vdot(sample_forward(x,2,-2),y),np.vdot(x,sample_adjoint(y,2,-2)),rtol=1e-13)
def test_windows_partition_unity_and_constant_preservation():
    image=np.full((80,80),123.);tiles=getTiles(image,16,8)
    reconstructed=depatchifyOverlap(cosineWindow2Dpatches(tiles))
    np.testing.assert_allclose(reconstructed[8:-8,8:-8],123,atol=1e-12)
def test_wiener_limits_and_dc_preservation():
    ref=np.ones((2,2,16,16),dtype='complex64')*10;alt=ref*2;var=np.ones_like(ref.real)
    np.testing.assert_allclose(mergeWienerDFTPatches(ref,alt,var,0),ref)
    np.testing.assert_allclose(mergeWienerDFTPatches(ref,ref,var,75),ref)
    out=spatialDenoisePatches(ref,var,.1)
    np.testing.assert_allclose(out[...,0,0],ref[...,0,0]);assert abs(out[...,8,8]).max()<10

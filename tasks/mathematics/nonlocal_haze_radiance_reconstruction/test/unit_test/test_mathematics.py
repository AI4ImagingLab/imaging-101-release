import numpy as np
from src.physics_model import image_formation,recover_radiance,rgb_to_gray
from src.solvers import weighted_system,contrast_stretch,SPHERE_POINTS

def test_physical_inverse_without_display_bias():
    j=np.array([[[.2,.3,.6],[.1,.4,.7]]]);t=np.array([[.3,.8]]);a=np.array([.72,.785,.81])
    measured=image_formation(j,t,a)
    np.testing.assert_allclose(recover_radiance(measured,t,a,leave_haze=1.),j,rtol=0,atol=3e-16)

def test_residual_haze_convention():
    j=np.full((2,3,3),.2);t=np.full((2,3),.5);a=np.array([.72,.785,.81])
    np.testing.assert_allclose(recover_radiance(image_formation(j,t,a),t,a),j+.06*a,atol=3e-16)

def test_explicit_luminance():
    np.testing.assert_allclose(rgb_to_gray(np.eye(3)),[.298936,.587043,.114021],atol=0,rtol=0)

def test_graph_energy_independent_edges():
    h,w=4,5;initial=np.linspace(.3,.8,h*w).reshape(h,w);weight=np.arange(h*w).reshape(h,w)/20
    image=np.arange(h*w*3).reshape(h,w,3)/60;lam=.1
    matrix,rhs,modified,reliability,smooth=weighted_system(initial,weight,image,lam)
    dense=smooth.toarray();np.testing.assert_allclose(dense,dense.T,atol=0)
    np.testing.assert_allclose(dense@np.ones(h*w),0,atol=1e-12)
    x=np.cos(np.arange(h*w)).reshape(h,w);gray=rgb_to_gray(image)
    independent=0.
    for r in range(h):
        for c in range(w):
            for dr,dc in [(1,0),(0,1)]:
                rr,cc=r+dr,c+dc
                if rr<h and cc<w:independent+=lam*(x[r,c]-x[rr,cc])**2/((gray[r,c]-gray[rr,cc])**2+1e-5)
    np.testing.assert_allclose(x.ravel(order='F')@dense@x.ravel(order='F'),independent,rtol=2e-15)
    np.testing.assert_allclose(rhs,reliability.ravel(order='F')*modified.ravel(order='F'),atol=0)
    assert np.linalg.eigvalsh(matrix.toarray()).min()>0

def test_top_boundary_anchor():
    initial=np.array([[.8,.7],[.2,.3]]);weight=np.array([[0.,0.],[1.,1.]])
    _,_,modified,reliability,_=weighted_system(initial,weight,np.ones((2,2,3))*.5)
    np.testing.assert_array_equal(modified[0],[.2,.3]);np.testing.assert_array_equal(reliability[0],[.8,.8])

def test_no_wraparound():
    _,_,_,_,smooth=weighted_system(np.full((3,4),.5),np.ones((3,4)),np.ones((3,4,3))*.5)
    assert smooth[2,3]==0 and smooth[0,11]==0

def test_sphere_constants():
    assert SPHERE_POINTS.shape==(1000,3)
    assert np.max(np.abs(np.linalg.norm(SPHERE_POINTS,axis=1)-1))<.00009
    np.testing.assert_array_equal(SPHERE_POINTS[0],[-.9425,-.2041,-.2648])

def test_contrast_integer_and_degenerate():
    import pytest
    image=np.arange(180,dtype=float).reshape(6,10,3)/179
    out,limits=contrast_stretch(image)
    assert out.dtype==np.uint8 and out.min()==0 and out.max()==255 and limits.shape==(2,3)
    with pytest.raises(ValueError):contrast_stretch(np.ones((2,2,3)))

def test_metrics_known_case():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('independent_judge',Path(__file__).resolve().parents[2]/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
    ref={k:np.array([0.,1.,2.]) for k in j.CONTRACT};out={k:np.array([1.,1.,1.]) for k in j.CONTRACT}
    values=j.metric_values(out,ref)
    assert abs(values['reconstruction_nrmse']-np.sqrt(2/3)/2)<1e-15
    assert abs(values['reconstruction_ncc']-3/np.sqrt(15))<1e-15
    from src.visualization import compare
    metrics=compare(out['reconstruction'],ref['reconstruction'])
    assert abs(metrics['ncc']-values['reconstruction_ncc'])<1e-15
    assert abs(metrics['nrmse']-values['reconstruction_nrmse'])<1e-15

import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.physics_model import design_matrix,local_response,mixed_derivative_density
from src.solvers import vari_T,vari_s,vari_weights,Bayes_factor,OLS2BF
from src.visualization import metrics


def test_quadratic_mixed_derivative_and_rectangle_integral():
    c=np.array([.7,.2,-.3,.4,-.5,1.3])
    b=np.array([.08,.08,.03,.03]);r=np.array([.02,-.04,.02,-.04])
    corner_difference=local_response(c,b,r)@[1.,-1.,-1.,1.]
    expected=-2*mixed_derivative_density(c)*(.08-.03)*(.02+.04)
    np.testing.assert_allclose(corner_difference,expected,rtol=1e-12,atol=1e-15)
    # Adding arbitrary quadratic marginal functions changes the moment but
    # leaves its mixed derivative and physically integrated density unchanged.
    c[:5]+=np.array([2.,3.,-2.,.1,1.])
    assert mixed_derivative_density(c)==-.65


def test_cubic_basis_and_gradient_sign_independently():
    c=np.array([.7,.2,-.3,.4,-.5,1.3,.8,-.4,.5,.9])
    b=.08;r=-.03;d=1e-4
    corners=local_response(c,np.array([b+d,b+d,b-d,b-d]),
                           np.array([r+d,r-d,r+d,r-d]))
    mixed=(corners@[1.,-1.,-1.,1.])/(4*d*d)
    assert mixed==pytest.approx(c[5]+2*c[8]*b+2*c[9]*r,rel=1e-7)


def test_taper_support_and_continuity():
    np.testing.assert_array_equal(vari_T(np.array([0.,1.,1.5,2.,3.]),2),[1.,1.,.5,0.,0.])
    assert vari_s(2,2,0,9,.01)==2
    w,use=vari_weights(2,2,0,2,2,0,np.array([0.,1.,2.]),np.zeros(3),1.,0.,0.)
    np.testing.assert_array_equal(use,[True,True,False]);np.testing.assert_array_equal(w,[1.,1.])


def test_evidence_gamma_formula_independent():
    from scipy.special import gammaln
    from scipy.integrate import quad
    numerator=quad(lambda x:x**1.25*np.exp(-x),0,np.inf)[0]
    denominator=quad(lambda x:x**-.75*np.exp(-x),0,np.inf)[0]
    assert numerator/denominator==pytest.approx(5/16,rel=1e-9)
    for order,p in [(1,4),(2,5),(3,9)]:
        n=35;r2=.78;a=-.75
        expected=gammaln(p/2+a+1)-gammaln(a+1)+gammaln((n-p-1)/2)-gammaln((n-1)/2)+(-(n-p-1)/2+a+1)*np.log(1-r2)
        _,base=Bayes_factor(n,r2)
        value,_=Bayes_factor(n,r2,order,base)
        assert value==pytest.approx(expected,abs=2e-13)


def test_judge_and_plot_metrics_known_cases():
    path=Path(__file__).resolve().parents[2]/'evaluation/judge.py'
    spec=importlib.util.spec_from_file_location('independent_judge',path)
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    reference=np.arange(1.,11311.)[None]
    expected=metrics(reference*2,reference)
    assert expected['ncc']==pytest.approx(1)
    assert expected['nrmse']==pytest.approx(np.sqrt(np.mean(reference**2))/np.ptp(reference))
    assert metrics(np.zeros_like(reference),reference)['ncc']==0
    with pytest.raises(ValueError):metrics(reference,np.ones_like(reference))

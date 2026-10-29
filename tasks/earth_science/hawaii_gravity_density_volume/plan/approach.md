# Implemented prism inversion

## Observations and fixed physical problem

The data are 1019 unchanged stations selected from the USGS 3611-record historical Hawaii table by strict WGS84 latitude/longitude bounds in README. The unknown is a signed effective density contrast on 2048 fixed underground prisms. There is no actual density truth. The reproduced standard inverse is the benchmark reference; no true volcanic structure is claimed. Orthometric elevation is the original station height, not ellipsoidal height. The adopted Cartesian and normal-gravity reductions are explicit approximations, not a reproduction of every reduction in the 2000 geological study.

Preprocessing exactly follows README: compute local east/north coordinates using Earth radius6371000m, latitude reference19.4deg and longitude reference-155.275deg. Somigliana normal gravity, linear free-air .3086mGal/m and infinite-slab density2670kg/m^3 give simple Bouguer b. Let T=[1,x/10000,y/10000]. The orthogonal plane projection P is implemented as value-T*lstsq(T,value), using the same operation on a vector or an operator matrix. Set d=-P*b to switch downward observed gravity into upward gz. Forward predictions are P*K*rho. This consistent projection is essential: removing a trend from observations alone would change the inverse problem.

## Explicit finite-volume forward model

The volume has sixteen3km cells in both x and y, eight2.5km cells in z, bounds[-24,24]km laterally and[-20,0]km vertically. Cell centers are ordered x fastest, then y, then z, so the output is shape(8,16,16) with increasing upward z. Density units are g/cm^3; receivers are at measured orthometric heights. Outside the box density contrast is fixed to zero within this model, while projected regional signals account only for an affine nuisance component.

For node offsets X,Y,Z from a receiver, R=sqrt(X^2+Y^2+Z^2). The gravitational primitive is F=X log(Y+R)+Y log(X+R)-Z atan(XY/(ZR)). Evaluate logarithmic terms only when their multiplier is nonzero and log argument positive; evaluate the last term only for nonzero Z. The original ordinary arctan branch, rather than atan2, is preserved. Sum F at the eight prism corners with sign(-1)^(ix+iy+iz), where0 selects lower and1 upper coordinate bound. Multiply by G*10^8, G=6.67430e-11 SI, to map g/cm^3 density to upward mGal. Positive below-receiver mass gives negative gz. Independent Gauss-Legendre integration of Newton's volume integral verifies branch, sign and scale, separately from the original source fixture.

## Objective and regularizer

The unhalved source objective is f(rho)=norm((A*rho-d)/5)^2+beta*norm(R*rho)^2, where A=P*K, beta=.001 and the constant5mGal is an adopted modeling tolerance, not measured instrument noise. Reference density and initial density are zero. There are no positivity or box constraints on density.

Cell volume V=3000*3000*2500=22.5e9m^3. Smallness coefficient alpha_s=1/3000^2; each smallness residual is sqrt(V)*sqrt(alpha_s)*rho=50*rho. First-difference residuals across interior x and y neighbors are sqrt(V)*(rho_neighbor-rho)/3000=50 times the difference; across z neighbors they are60 times the difference. Exterior boundary derivatives vanish naturally. All three first derivative weights alpha_x,alpha_y,alpha_z are1. Second derivative multipliers are zero in the selected original WeightedLeastSquares; their identically zero rows are omitted locally without changing R-transpose-R. The regularizer is not divided by cell count. Its dimensions depend on the stated density/metre convention; beta has the corresponding reciprocal regularizer units relative to dimensionless weighted misfit. Numeric beta values cannot be transferred to a differently scaled mesh without conversion.

The gradient is2[A^T(A*rho-d)/25+beta*R^T*R*rho]. The Hessian action is2[A^T*A*v/25+beta*R^T*R*v]. These explicit formulas are tested by scalar finite differences and against independently captured original regularizer and final objective. No depth weighting, bounds, iterative reweighting or stochastic initialization is added.

## Original Newton-CG selection

The original unmodified SimPEG InexactGaussNewton is configured with identity CG preconditioning, cg_rtol1e-12,cg_atol0,max2000, maximum10Newton updates, tolF1e-12,tolX1e-12,tolG1e-10. A zero start is used. Identity preconditioning avoids an unspecified domain preconditioner; the source's internally initialized BFGS-H0 factorization is not used by this explicitly selected preconditioner. Local newton_cg forms the same Hessian action and solves H*p=-gradient with general SciPy CG. It then applies an explicit Armijo update: start step1, accept f(rho+step*p)<=f(rho)+1e-4*step*gradient dot p; otherwise shorten by .5, at most10attempts. The source's absolute gradient stopper1e3*eps=.01 (eps=1e-5) is retained along with the selected objective-change tolerance and10-iteration cap. On this strictly quadratic bound problem, one full step is accepted and the next gradient is below1e-8. The source moving-x stopper and maxStep default are inactive on this one-step problem; the local helper stops on the same first active condition. No artificial iteration is used to disguise a direct inverse.

The result is not retuned after comparison: the independently successful original run fixes all constants before local extraction. Predeclared fixture tolerances are recorded privately before local code. Actual local/original density differences are about1.8e-12g/cm^3, much tighter than test and evaluator tolerances.

## Evaluation and sensitivities

Evaluate the whole volume without cropping, rescaling, shifting or alignment. NCC is cosine similarity and NRMSE is RMS error divided by reference dynamic range. A zero estimate has NCC0. The standard-method reference is not actual geology. Reference self-comparison must use a nonzero numerical tolerance: NRMSE<=1e-7 and NCC>=.999999999. The independent Judge also computes RMS projected observed-gravity residual<=2mGal; it uses original target-independent prism kernels and independent QR nuisance projection from public coordinates. It never imports the solver.

Notebook comparisons actively compute original-versus-fresh volume slices, real-data predictions/residuals, all raw correction stages, convergence and beta sensitivities .0001,.001,.01. The chosen beta sensitivity is a method dependence study, not additional observational truth. Residual model errors, approximate terrain handling, plane-projection nullspace, finite-box leakage and depth nonuniqueness limit geological interpretation.

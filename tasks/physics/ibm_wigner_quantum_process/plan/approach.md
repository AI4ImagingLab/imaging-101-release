# Implemented method

The primary method is the published weighted-droplet iterative algorithm, run on all 50 experimental directions, both measured ranks and all four controlled reference channels. It returns a single dimensionless complex unitary process. The published optimized objective is an actively computed comparison, with a separately identified general numerical optimizer. No known target gate is an input, initializer or truth.

## Geometry and forward acquisition

Angles theta and phi use the ordinary polar/azimuth convention. Let n=(sin(theta)cos(phi),sin(theta)sin(phi),cos(theta)). The real normalized quaternion q=(A,B,C,D) defines U=D I+i(A X+B Y+C Z), with computational basis order0,1. Put v=(A,B,C) dot n. Ideal measured complex rank components are e0,I=D^2, e1,I=iDv, e0,j=-i qj D and e1,j=qj v. These follow from the controlled-Pauli auxiliary interferometer. Measurements are dimensionless. Hardware attenuation and control errors remain; the ideal forward residual is diagnostic rather than an exact noise model. Raw counts and shot variances are not supplied, and no noise distribution is invented.

The real spherical basis is b=(sqrt(3/(32pi))*nx, sqrt(3/(32pi))*ny, sqrt(3/(32pi))*nz, 1/sqrt(32pi)). Lebedev weights w integrate the sphere average, summing to1; every basis squared norm is1/(32pi). The four raw spherical functions are (e0+sqrt(3)*e1)/(4sqrt(2pi)). Reorder I,X,Y,Z to X,Y,Z,I and multiply the first three by i. Call these f_j. In the ideal model f_j=qj times the spherical representation of U. Independent synthetic operator identities are tests only; the benchmark remains actual experiment.

## Initial estimate and iteration

Compute C_jk=sum_n w_n conj(f_jn) f_kn and S=C/0.00995. The common scalar0.00995 is the author's rounded analytic pure-Pauli basis norm (1/(32pi)), not a calibration fitted to the hidden gate. It cancels from the normalized iterative output. Preserve it for stage parity. The upstream sign routine always executes its final largest-diagonal branch, overriding preceding ideal-sign branches. The local function implements that effective behavior explicitly. Select largest abs(sqrt(S_jj)); use the signs of its correlations with the other quaternion components and D, setting sign(D)=+1. Initial qj=sign_j sqrt(S_jj). Tiny imaginary roundoff is preserved until the real-quaternion projection.

Combine f=sum_j qj f_j. Project c_k=sum_n w_n conj(b_kn) f_n / sum_n w_n abs(b_kn)^2. Set q=(imag(cX),imag(cY),imag(cZ),real(cI)), then divide by its Euclidean norm. Repeat this operation10 times: the initial projection and nine author updates; the tenth normalized quaternion is the returned matrix. No regularizer, line search or early stopping is added. Upstream uses a fixed iteration count, preserved here. The quaternion order and signs must not be changed to a different convention.

## Active optimized comparison

Project each f_j to its four Pauli coefficients, build M_j, then divide M_j by the Euclidean norm of its coefficient vector. This differs from a Frobenius matrix norm by sqrt(2); preserve the upstream formula. Its initial q is q0/sqrt(sqrt(sum(q0*q0))), where q0 is the signed square-root diagonal estimate above. This is intentionally the author's non-unit initial normalization, not silently repaired. Convert to Euler angles alpha=atan2(C,D)-atan2(A,B), beta=acos(1-2(A^2+B^2)), gamma=atan2(C,D)+atan2(A,B).

Euler angles parameterize A=-sin(beta/2)sin((alpha-gamma)/2), B=sin(beta/2)cos((alpha-gamma)/2), C=cos(beta/2)sin((alpha+gamma)/2), D=cos(beta/2)cos((alpha+gamma)/2). Minimize J=0.5 sum_j ||M_j-qj U(q)||_F^2. The fresh comparison uses SciPy BFGS, maxiter45, gtol1e-8, forward finite-difference step1e-8, recording initial and callback iterates. The independent original MATLAB code was run under Octave fminunc with a documented options/callback compatibility layer. These different general optimizers are not claimed to have identical trajectories. Objective values at every original iterate and the attained minimum are independently compared. Fresh BFGS reports precision loss after reaching the original minimum to1.8e-15; that warning is retained rather than described as optimizer success.

## Evaluation and diagnostics

The primary output is compared with a validated original method result, not actual truth. Remove only one global phase using the argument of the complex matrix inner product. Report RMS magnitude of complex-entry differences and Frobenius unitarity defect. Boundaries1e-6 were frozen before extraction, allow complex64 roundoff, and are much tighter than distinct process differences. No amplitude scaling, crop, basis permutation, or per-entry alignment occurs. Scalar gain fitted to measured data in residual plots is explicitly diagnostic and is never public calibration or a solver input. Rank removal and direction thinning are notebook ablations only; the standard solve always uses all observations. Historical paper values, reproduced original values, and fresh results remain separate.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator independently forms all eight complex expected channels on the 50-direction Lebedev grid, with the public quadrature weights. Weighted mean squared observable error may exceed the attainable fit by at most 1e-6; the original unitarity boundary 1e-6 remains. Global unitary phase is a gauge. An independently optimized quaternion with lower real-data loss passes; a wrong Pauli operation fails. The conditional ideal-unitary fit is not independent process truth. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.

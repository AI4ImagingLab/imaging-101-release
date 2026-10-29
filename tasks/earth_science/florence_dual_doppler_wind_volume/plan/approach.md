# Implemented Florence wind-volume inverse

The fixed measurements are two real NOAA NEXRAD volumes during Hurricane Florence on 14 September 2018, with radar scan reference times 06:24:50 and 06:24:17 UTC. Author files grid_ltx.nc and grid_mhx.nc have source dimensions (31,351,401) in (z,y,x). Sampling z[0:17:2], y[196:237:2], x[155:196:2] without interpolation gives (9,21,21). The window was chosen using observed coverage and original beam-crossing geometry only. No historical wind retrieval is retained. Radar A is at (34.77590942 N,76.87619019 W,43 m); B at (33.989151 N,78.42910767 W,44 m). The source projection origin is (33.989151 N,78.42910767 W). Source filenames are not reliable station identifiers, so A/B refer to this explicit order.

Coordinates x,y,z are metres eastward, northward and upward. Their spacings are 2250,1714.28571428571,1000 m; ranges are x=24375..69375, y=68000..102285.714285714, z=0..8000 m. Known angles alpha (azimuth clockwise from north) and epsilon (elevation) are radians, computed by original known-geometry routines with effective Earth radius (4/3)*6371.1 km. The binary mask m includes jointly valid observations and 30..150 degree beam crossing. All arrays are promoted from original float32 values to float64, retaining source precision. Invalid samples/calibration values are finite zero placeholders with zero radial weights. No noise is added.

At radar r and grid point, predicted outward radial velocity is

```math
v_r=\cos\epsilon_r\sin\alpha_r\,u+\cos\epsilon_r\cos\alpha_r\,v+
\sin\epsilon_r\,(w-|v_{t,r}|).
```

Here u,v,w are eastward, northward and upward air velocities in m/s. Particle terminal velocity is v_t=A*10^(B*Z)*(1.2/rho)^0.4 m/s, with measured reflectivity Z in dBZ, rho=exp(-z/10000) kg/m^3 and freezing height 5000 m. Below freezing, A,B are (-2.6,.0107) for Z<55, (-2.5,.013) for 55<=Z<60, and (-3.95,.0148) for Z>60. At or above freezing, they are (-.817,.0063) for Z<33, (-2.5,.013) for 33<=Z<49, and (-3.95,.0148) for Z>49. Source gaps at exactly Z=60 below freezing and Z=49 above freezing assign zero; none of the selected valid observations lies at those gaps. This disclosed convention is preserved rather than silently repaired.

Let d be observed radial velocity, sigma=sqrt(sum(m*d^2)/sum(m)), and D denote first spatial differences. The exact selected objective is

```math
J=\frac{\sum_{r,z,y,x}m_r(v_r-d_r)^2}{\sigma^2}
+\frac{C_m}{2}\sum_{z,y,x}\delta^2
+C_p\sum_{z,y,x}(u^2+v^2),\qquad
\delta=D_xu+D_yv+D_zw+w\frac{D_z\rho}{\rho}.
```

The radial coefficient is 1, continuity coefficient C_m=256 s^2 and horizontal prior coefficient C_p=1e-4 s^2/m^2. The original configurable model constraint is applied as a zero-mean Gaussian horizontal prior with unit spatial weights. This is a regularizer assumption, not a measured/forecast wind input. The original term excludes w, which we preserve. Smoothing, vorticity, sonde, point and terrain terms are disabled. Weakly observed w remains conditional on continuity and this horizontal prior. Original data have 6576 valid radial samples and sigma=32.63715956937172 m/s. Source gridding, dealiasing and empirical fall speed are accepted approximations; differing scan times limit the frozen-time model.

D uses centered interior differences (f[i+1]-f[i-1])/(2h) and one-sided edge differences (f[1]-f[0])/h and (f[-1]-f[-2])/h, identical to np.gradient(edge_order=1). Its transpose accumulates each row's coefficients: first row contributes (-q[0]/h,+q[0]/h), each interior row contributes -q[i]/(2h) to its left and +q[i]/(2h) to its right, and last row contributes (-q[-1]/h,+q[-1]/h) to the last two entries. It is not generally negative D.

The explicit gradient is 2*A^T[m*(v_r-d)/sigma^2] plus C_m*(D_x^T delta,D_y^T delta,D_z^T delta+delta*(D_z rho)/rho) plus 2*C_p*(u,v,0), where A is the local radial projection. Vertical gradients on first/last z planes are set to zero. Starting w=0 there and preserving zero boundary components in optimizer updates enforces fixed zero vertical velocity at the two analysis planes. Directional tests use admissible perturbations that also vanish there.

The selected unchanged original JAX64 objective and autodiff implementation runs through _get_dd_wind_field_scipy with engine='jax', zero u/v/w initialization, bounds [-100,100] m/s, maximum 500 iterations, tolerance 1e-8, and no low-pass filtering or output masking. Final Florence log records convergence at iteration 99, objective 448.8818667363615 and projected-gradient norm 7.710967420138104e-9. An independent original repeat reproduces the same result. The original NumPy continuity-gradient branch fails the exact discrete-adjoint check and omits a density derivative contribution; selecting the already supplied JAX branch avoids silently copying or correcting that discrepancy.

Local reconstruction uses scipy.optimize.fmin_l_bfgs_b on the explicit objective/gradient, zero initialization, identical bounds, history m=10, factr=1, maxls=40, pgtol=1e-9, maxiter=2000. Nonzero warnflag raises an error. Final local run takes 113 iterations, reaches J=448.8818667363543 and projected-gradient maximum 1.8318539033216306e-9. Original JAXopt and local SciPy line searches differ, so validation compares converged minimizers rather than equal iteration trajectories. No physical model or objective term is changed to obtain agreement.

Evaluation uses the original method's conditional wind volume, not physical truth. Each complete u,v,w array has cosine NCC>=.99999 and dynamic-range NRMSE<=5e-5; no crop, alignment, gain, offset or masking is applied. Actual original/local NRMSE values are 3.7474e-8,5.3364e-8,1.6658e-7. Independent original/local repeats and float32 serialization measurements justify nonzero tolerances. Zero, uniform and spatially reversed submissions fail. Full-field local radial residual RMS is 2.28058715256 m/s versus 32.6295947456 for zero air wind. Tests also enforce gradient/adjoint physics, exact boundary conditions and projected optimality. Active notebook comparisons refit five prior coefficients and three continuity coefficients; changes in w demonstrate model sensitivity rather than independently measured uncertainty.

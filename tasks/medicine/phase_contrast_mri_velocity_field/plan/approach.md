# Implemented phase-contrast inverse

The selected published stage recovers relative tissue velocity from four complex MRI encoding images. It is not a new compressed-sensing MRI solver. Original raw-MRD sorting and the complete108-volume author ESPIRiT/PICS reconstruction were independently executed privately to establish the supplied observation boundary. The cleaned runtime begins at those complex observations and contains no BART, sensitivity fitting, frequency-data solver, domain-library call or upstream import. This boundary is explicit in the public problem.

## Acquisition and identifiable unknown

The complete native grid is100x67x54, with27triggered phases and4encodings(reference,x,y,z), four original receive coils and VENC V=.16m/s. Coordinates are relative image/encoding axes; absent independent gradient-probing calibration, no anatomical gradient polarity is inferred. The deposited metadata's TR7ms is not treated as temporal-phase spacing. No displacement or strain is claimed.

At position r, phase t and directional encoding d, S0=a0 exp(i phi0)+epsilon0 and Sd=ad exp(i[phi0+pi u_d/V])+epsilon_d. Complex amplitudes carry arbitrary signal units, phases radians, velocitym/s. No equality of directional amplitudes is required. Product zd=Sd conj(S0) cancels the common phase, leaving angle theta_d=Arg(zd) in[-pi,pi]. Its source principal velocity w_d=V theta_d/pi is aliased modulo2V. There is no phase-unwrapping algorithm or assumption that all noise voxels are physiologically meaningful.

The required gauge is v_d=M(w_d-mean_t w_d), equal weighting of all27phases, followed by the strict observed support M=1{abs(S0)>.1max(abs(S0))}. The maximum is over all spatial positions and times, reference encoding only. Support can vary with t, so its final masked temporal average need not vanish. Values beyond V after centering are retained. This is a selected periodic-motion convention, not a claim that true steady velocity vanishes. Low-signal/aliasing/reconstruction artifacts remain physical limitations.

## Exact local arithmetic and discretization

Public data are complex64(1,27,4,54,67,100), axes batch,time,encoding,z,y,x. After validating dtype/shape/finite values, transpose and copy into contiguous original x,y,z,time,encoding order. For d=1,2,3 independently evaluate np.angle(S[...,d]*np.conj(S[...,0])), then stack along the last axis. Preserve complex64 products, float32 angles, and NumPy2.4.6weak-scalar operations. Velocity=(phases/pi)*.16 retains float32.

For each direction take the arithmetic mean over the fourth native axis(time), keep that singleton axis, and subtract it in place before masking. The loop order/storage match the source. Compute abs(reference) in float32, threshold its global maximum with strict greater-than, multiply velocity by the boolean support, and transpose the resulting x,y,z,time,3 field to contiguous time,3,z,y,x. Add its singleton batch axis; output float32. All phases, components and voxels remain; there is no crop, interpolation, clipping, spatial regularizer, objective minimization or alignment.

The inverse is pointwise plus a temporal mean. It has no optimizer, initializer, step size, line search, iterative stopping rule or training. The private acquisition solver's20FISTAiterations and wavelet weight.005 describe input construction, not an unimplemented local runtime. Optional recover_velocity parameters VENC, mask_fraction and remove_temporal_mean support identified review comparisons. Standard VENC=.16, mask_fraction=.1 and remove_temporal_mean=True define the bound method. The public output support and gauge are physical interface conventions, not hidden tuned parameters.

## Original checks and meaningful tolerances

Complete original raw-data acquisition and velocity code ran in isolated Linux CPU with original Python source unchanged and published BART1.0.00commit37f3c5f0feaf0a19a7fca85c926ad74101f4a034. It executed216successful numerical calls for108image volumes. Three isolated binary crashes required repeating the identical BART subprocess command; no completed image was replaced by zero or a cached answer. A prior HEAD run stopped at volume101 and is separately failed, not counted complete. Full logs/call hashes and original images/results are archived privately.

Before local extraction, source-AST lines128-149 were independently run on the same frozen complex observations in Windows. Its velocity differs from the original full Linux result by max2.9802322387695312e-8m/s, RMS7.910816474037584e-10m/s, dynamic-range NRMSE2.0920382016755813e-9. The same original block in complex128 gives comparable rounding variation. Frozen phase absolute tolerance2e-6rad, velocity5e-8m/s and exact mask tests precede local implementation. Small contiguous6x6x6spatial stage cases retain all27phases; a separate full-grid test checks the whole independently computed original result.

Independent analytic checks synthesize known signed encoding phases, constant/common background phase, a temporal component and changing magnitude support. They check VENC sign/scale, modulo2V, mean-before-mask order and common-phase cancellation without importing upstream code. These test cases do not replace the real bound observations.

## Evaluation and review comparisons

There is no independently measured velocity truth. Hidden baseline is the original complete acquisition/velocity result. Global and three component cosine NCC and dynamic-range NRMSE must pass .9999995 and1e-6, respectively. The nonzero precision allowance is justified by independent source OS/backend and float32/64 runs; it is not1.1times zero self-error. No fitted normalization or alignment is used.

An independent Judge computes theta with complex128 products and real/imaginary atan2, fixes background beta=mean_t theta using only the public observations, and predicts exp(i[pi v/V+beta]). The RMS complex-phasor discrepancy over the public support and all3directions must be<=2e-6. Outside that support maxabs(v)<=1e-8m/s. These forward and support checks use no hidden velocity to determine a phase or scale. They validate the declared relative inverse, not absolute MRI accuracy. Actual truth takes precedence if supplied later; malformed truth must fail as an evaluator error.

The review notebook reconstructs the complete benchmark and comparison settings from cleaned modules. It compares original/fresh actual maps, all three directions and all phases, orthogonal spatial views, temporal motion/support statistics, independent phasor closure, uncentered-versus-centered velocity, and genuine support-threshold sensitivity. It does not fabricate unprovided fully sampled MRI, clinical ground truth, anatomical orientation or strain curves.

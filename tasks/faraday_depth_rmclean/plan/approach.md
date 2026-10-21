# Implemented mathematics

## Selected physical problem and scope

The full distributed 288-channel synthetic observation is reused, not regenerated or spatially reduced. Unknowns are complex polarized per-bin flux components C on 401 increasing Faraday depth bins phi_j=-600+3j rad/m². P_k=Q_k+iU_k, frequency nu_k is in Hz, sigma_k is the known per-real-component noise std in Jy/beam. c=299792458 m/s and t_k=(c/nu_k)^2 m². Let w_k=1/sigma_k², K=1/sum(w), t0=K sum(w t). Our component convention absorbs bin width; no dphi factor is inserted. The physical acquisition is P=A C+epsilon with A_kj=exp(2i phi_j(t_k-t0)). This is the source's absolute-flux/no-Stokes-I-normalization path. Channel bandwidth averaging, transverse imaging, spectral-index correction, QU model selection, fitted-peak characterization and full-resolution deconvolution are outside this frozen task, not unimplemented parts of its runtime.

The synthetic spectrum has frequency dependent source intensity; uncorrected spectral-index structure is therefore not interpreted as independent Faraday-thick truth. This limitation is visible in channel residuals. No unsaved simulated object is relabeled as actual truth. The evaluation reference is an independently executed selected method.

## Operators and normalization

The forward operator is A C, with positive Fourier sign. Its Hermitian adjoint is (A* P)_j=sum_k P_k exp(-2i phi_j(t_k-t0)), without normalization or weights. They satisfy <A C,P>=<C,A*P> under the complex Euclidean inner product. The inverse variance dirty synthesis is D=K A* diag(w) P. A true single component at phi_j has D_j=C_j at its peak. The transformed real-component noise std is sigmaF=K sqrt(sum(w_k² sigma_k²)); there is no extra factor sqrt(2) because sigma_k is per real component.

The RMSF R at depth offset psi is K sum_k w_k exp(-2i psi(t_k-t0)). For n=401 depth samples, the original complete response extends by ceil(n/2)=201 bins at each end, giving 803 samples from -1203 to1203 rad/m². The response is computed explicitly. Circular rolling is used only to index this extended response; the central401-bin crop supplies the nonwrapped response required for any represented component. It is not a periodic physical boundary condition.

The selected source release uses W=3.8/(max(t)-min(t)) for the Gaussian restoring FWHM. This is the source's conventional analytical-width approximation, not the older 2sqrt(3) approximation often quoted for uniform wavelength-squared weighting. The precise supplied channel sequence and W=58.90951914103005 rad/m² are preserved. The weighted reference t0=0.10327486263730823m² is computed with source operation ordering.

## Explicit inverse updates

Initialize residual r=D, complex components C=0, restored accumulator H=0 and iteration n=0. Choose the first absolute maximum j of r using NumPy argmax; this fixes tie handling. The component update is a=gamma r_j with gamma=0.1. Accumulate C_j <- C_j+a. Form the source-prescribed shift of R so that its peak overlays bin j, and update r <- r-a R_shift. Accumulate H_l <- H_l+a exp[-4ln(2)((phi_l-phi_j)/W)^2]. This Gaussian has unit peak, not unit area. Finally return reconstruction=H+r and components=C. Returned complex arrays include one leading batch axis.

Stopping is max(abs(r)) < 3sigmaF or n=1000, whichever occurs first. The bound observation takes50 updates. There is no line search, gradient descent objective or custom proximal update hidden in a library. CLEAN is a greedy response deconvolution; its regularization is sparse component selection and noise-level early stopping. It is not presented as a generic optimizer minimizing a separately invented loss. The optional second windowed loop is explicitly disabled by selecting the upstream public window=None path (low-level window=NaN), so its conditional is false. That loop is not silently omitted from a purported default-window=0 reproduction.

## Independent extraction and tolerances

The unmodified original Python functions were executed first on the exact bound observations. Original dirty synthesis uses FINUFFT tolerance1e-12, whereas the local code evaluates the finite Fourier sum directly in complex128. The RMSF and CLEAN formulas, gain, complete response padding, selected window condition, Gaussian width, grid and stopping criteria are preserved. Frozen parity thresholds are rtol1e-9 and atol2e-11 in flux units, with exact checks for copied observations, weights, squared wavelengths, depth grid and update count. Independent thin-screen sign/scale/phase, adjoint and residual-operator tests prevent parity alone from certifying a shared mathematical error.

## Evaluation

Both output arrays are scored without alignment or phase/scale fitting: E=norm(estimate-reference)/norm(reference), complex Euclidean norms. Undefined zero-reference norms are evaluator errors. Each E must be<=1e-7, frozen before local implementation, comfortably above deterministic float64/NUFFT differences but far below a changed spectrum. There is no NCC/NRMSE relabeling; these are explicit task-specific complex errors. Original CLEAN reruns are exactly equal. The forward statistic is sum_k abs((A C)_k-P_k)^2/sigma_k² divided by2*288. The independent original gives1.0996341269453398; boundary1.209597539639874 was frozen before the cleaned inverse as1.1 times that value. This is an observational consistency bound, not physical truth accuracy or a chi-square degrees-of-freedom estimate after fitting.

All channels, complex phase, native float64 input precision and full depth support remain visible in notebook plots. The notebook actively reruns the local pipeline; archived original arrays are used only for labeled comparisons. Standard CLI performs reconstruction and NPZ writing, without plotting or changing a notebook backend.

# Shipped interfaces and execution flow

Run `python -m src.main` from this task directory. `main()` loads `data/raw_data.npz`, actively reconstructs the full bound problem, creates `output/`, and writes only the required float64 batch-first reconstruction in `output/reconstruction.npz`. It prints elapsed CPU time, iteration count and chi-square/sample and returns `(result,state)` to Python callers. Importing it does not select a plotting backend or execute a reconstruction. The CLI has no upstream imports, network access or hidden asset dependency.

`load_data(path='data/raw_data.npz')` returns a dictionary of copied float64 arrays with the leading unit batch axis removed: spectra `(16,89)`, dates `(16,)`. It rejects nonfinite values and uses allow_pickle=False. Units and axes are in README.

`prepare(data,mode=MODE)` allocates an explicit namespace: `par` selected flags/constants, `grid` 4584-cell Roche geometry, `bright` dimensionless positive values, `field` degree-15 harmonic coefficients and basis, `line` the one-line calibration, `views` sixteen projected geometries, `wavelength` sixteen 89-point nm arrays, `derivative` Cartesian magnetic basis or scalar zero, `profiles` sixteen disk-integrated Voigt states, `observation` and `variance` flattened 1424-vectors, `constants` selected packing sizes, `response` fixed magnetic matrix or None, and entropy `weights`. It computes line strength from visible I. It never reads files or writes output.

`reconstruct(data,target=1.,max_iterations=20,test_tolerance=1e-4,mode=MODE)` returns `(result,state)`: result is brightness `(4584,)` or magnetic `(3,4584)` in G; state adds `history` `(iterations,3)` of entropy, raw chi-square and Test, `iterate_images` evaluated packed images, `first` captured numerical stages, and `final_profile` `(16,89)` of the final evaluated source-convention model. It mutates its newly allocated state only. Optional target changes in the notebook are sensitivity runs of the same method, not standard CLI defaults. The mode constant is bound independently in each task; neither task imports the other.

`project_spatial(reconstruction,state,mode=MODE)` projects a supplied physical surface field, recomputing normalized disk integrals with the already calibrated geometry/profile state. It mutates state's brightness/profile buffers, returning `(16,89)` normalized measurements. Notebook calls use this result for fresh residuals, distinguishing it from source-convention cached final spectra. It is not used by Judge.

`physics_model.py` contains selected explicit Roche geometry, viewing projection, associated Legendre/vector harmonic formulas and Humlicek local profile mathematics. Numerical class names retain source spelling for equation traceability, not upstream package loading. Units: angles radians, surface area in polar-radius-squared units (canceling in normalized profiles), wavelengths nm, velocities km/s, B G. `starGrid` and `visibleGrid` expose float64 geometry arrays; magnetic basis is complex128 and physical vectors float64. `diskIntProfAndDeriv` contains local profiles `(89,4584)`, disk profiles `(89,)`, brightness derivatives `(4584,89)` or magnetic derivative `(3,135,89)` in the full geometry. Instrument broadening is already inside local Voigt width: `convolveIGnumpy` is an explicitly documented no-op in the selected model.

`solvers.py` implements entropy values/first/second derivatives, explicit ten-direction Skilling--Bryan updates and packing. General-purpose SciPy symmetric eigensolvers provide matrix decompositions only. `mem_iter` returns `(entropy,chi2,test,next_image,positive_entropy,filling_entropy,magnetic_entropy)` and updates its image argument in place; the filling entropy slice is empty for both selected tasks. `calibration.py` performs the scalar observed-equivalent-width calibration with a general-purpose Brent search. Its forward calculations call local physics only. Unselected source file I/O, initialization from fitted maps, exporters and the alternate target-entropy module are absent.

`visualization.py` supplies scatter surface maps and residual plots without backend side effects. The notebook actively calls reconstruction, physical reprojection and target sensitivities; independently labeled archived original arrays are read only for comparisons. It exports scientific groups and numeric sources to the private path configured by the maintainer's export environment variable. Its canonical saved outputs are from actual nbconvert execution.

`evaluation/judge.py` is standalone NumPy/stdlib. It resolves its own hidden assets relative to __file__, never imports source, validates exact shape/dtype/finiteness and physical brightness positivity, scores every required component and the independently formed observation residual, and writes UTF-8 JSON. Exit codes 0 pass, 1 valid poor result, 2 malformed/evaluator/report error. Prediction/report arguments resolve relative to caller cwd. Hidden baseline contains the actual original spatial reconstruction plus target-independent original line kernels, viewing geometry and continuum weights used only by the evaluator. `test/fixtures/upstream.npz` freezes first-update, derivative, geometry and final original measurements; comparing tests consume those numeric fixtures, never the original source. `notebooks/reference_outputs/upstream.npz` is a compact identified copy for review plots; fresh measurements are separately stored by the notebook. Actual truth is unavailable, so no ground_truth file is invented.

## Actual callable inventory

| Module | Shipped signature |
| --- | --- |
| preprocessing.py | `load_data(path='data/raw_data.npz')` |
| preprocessing.py | `prepare(data, mode=MODE)` |
| preprocessing.py | `reconstruct(data, target=1.0, max_iterations=20, test_tolerance=0.0001, mode=MODE)` |
| preprocessing.py | `project_spatial(reconstruction, state, mode=MODE)` |
| physics_model.py | `legendre_values(degree, x)` |
| physics_model.py | `starGrid.__init__(self, numPoints, period=0.0, mass=0.0, radiusEq=0.0, verbose=1)` |
| physics_model.py | `starGrid.GetCartesianCells(self)` |
| physics_model.py | `starGrid.GetSurfaceArea(self)` |
| physics_model.py | `starGrid.GetCartesianRotVel(self)` |
| physics_model.py | `starGrid.getCartesianNormals(self)` |
| physics_model.py | `starGrid.calcRclatOblate(self, period, mass, radius, verbose=1)` |
| physics_model.py | `starGrid.getOblateRadiusStep(self, period, mass, radiusEq)` |
| physics_model.py | `starGrid.gravityDarkening(self, gravDarkCoeff)` |
| physics_model.py | `visibleGrid.__init__(self, starGrid, inclination, cycleAtClat, period, dOmega)` |
| physics_model.py | `calcOmegaClat(period, dOmega, clat)` |
| physics_model.py | `getCyclesClat(period, dOmega, jDates, jDateRef, clat)` |
| physics_model.py | `calcVelDiffrotFactor(period, dOmega, clat)` |
| physics_model.py | `magSphHarmoics.__init__(self, nHarmics)` |
| physics_model.py | `magSphHarmoics.initMagGeom(self, clat, lon)` |
| physics_model.py | `magSphHarmoics.getAllMagVectors(self)` |
| physics_model.py | `magSphHarmoics.getAllMagDerivs(self)` |
| physics_model.py | `magSphHarmoics.getAllMagVectorsCart(self)` |
| physics_model.py | `magSphHarmoics.getAllMagDerivsCart(self)` |
| physics_model.py | `limbDarkening(coeffEta, angle)` |
| physics_model.py | `localProfileAndDeriv.__init__(self, lineData, numPts, wlGrid)` |
| physics_model.py | `localProfileAndDeriv.updateProfDeriv(self, lineData, Blos, dBlos_d, brightMap, numPts, wlGrid, surfaceScale, calcDI, calcDV)` |
| physics_model.py | `diskIntProfAndDeriv.__init__(self, visibleGrid, vMagCart, dMagCart, brightMap, lData, velEq, wlGrid, calcDI, calcDV)` |
| physics_model.py | `diskIntProfAndDeriv.updateIntProfDeriv(self, visibleGrid, vMagCart, dMagCart, brightMap, lData, calcDI, calcDV)` |
| physics_model.py | `diskIntProfAndDeriv.convolveIGnumpy(self, fwhm)` |
| physics_model.py | `getAllProfDiriv(par, listGridView, vecMagCart, dMagCart0, briMap, lineData, wlSynSet)` |
| physics_model.py | `calcSynEW(spec)` |
| solvers.py | `mem_iter(n1, n2, ntot, Img, Data, Fmodel, sig2, Resp, weights, defImg, defIpm, ffIMax, targetAim, fixEntropy)` |
| solvers.py | `get_c_gradc(Data, Fmodel, sig2, Resp)` |
| solvers.py | `get_s_grads(n1, n2, ntot, Img, weights, defImg, defIpm, maxIff)` |
| solvers.py | `searchDir(ntot, maxDir, Resp, sig2, fsupi, gradC, gradS, gradgradS)` |
| solvers.py | `diagDir(edir, nedir, gradgradS, sig2, Resp)` |
| solvers.py | `getCmuSmu(gradC, gradS, edir)` |
| solvers.py | `getL0squared(L_fac, Itot)` |
| solvers.py | `getAlphaMin(gamma)` |
| solvers.py | `getCaimQuad(chiAim, C0, gamma, Cmu)` |
| solvers.py | `chopDown(alpha, alphaLow)` |
| solvers.py | `chopUp(alpha, alphaHigh)` |
| solvers.py | `control(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin)` |
| solvers.py | `updateImg(xq, edir, Img, n1, n2, ntot, maxIff)` |
| solvers.py | `get_test(gradC, gradS)` |
| solvers.py | `packDataVector(obsSet, fitBri, fitMag)` |
| solvers.py | `packResponseMatrix(setSynSpec, nDataTot, npBriMap, magGeom, magGeomType, fitBri, fitMag)` |
| solvers.py | `packModelVector(setSynSpec, fitBri, fitMag)` |
| solvers.py | `packImageVector(briMap, magGeom, magGeomType, fitBri, fitMag)` |
| solvers.py | `unpackImageVector(Image, briMap, magGeom, magGeomType, fitBri, fitMag)` |
| solvers.py | `constantsMEM.__init__(self, par, briMap, magGeom, nDataTot)` |
| solvers.py | `setEntropyWeights(par, magGeom, sGrid)` |
| calibration.py | `equivWidComp2(lineStr, meanEquivWidObs, setSynSpec, lineData)` |
| calibration.py | `fitLineStrength(meanEquivWidObs, par, listGridView, vecMagCart, dMagCart0, briMap, lineData, wlSynSet, verbose=1)` |
| visualization.py | `surface_panel(axis, theta, phi, value, title, cmap='viridis', symmetric=False)` |
| visualization.py | `residual_panel(axis, velocity, residual, title)` |
| main.py | `main()` |


## Current benchmark v2 acceptance

The v2 Judge retains full measured-spectrum chi-square<=1.002 and independently integrates curved Roche cell areas for relative entropy to the unit photosphere. Entropy cost may not exceed attainable baseline cost by more than1e-7 per total area. No ground truth exists. A different higher-entropy data-consistent field passes, while uniform/scaled/shuffled fields fail.25 unit tests passed. See lopeg_pytest.log.

Source reconstruction and historical comparison computations remain unchanged; the standalone Judge is independent and uses the current README/metrics contract.

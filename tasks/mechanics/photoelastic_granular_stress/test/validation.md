# Provenance, explicit correction and validation

## Scientific scope and sources

Real recorded181x181 RGB ROI from author PeGS2 Step09-0001.jpg, original900x840 image. Exact crop rows115:296,columns305:486, unmodified uint8 values. Image SHA256baaaa673b311b9795710fc210e7378c63569f04929f6bfdb300943cb4fbbbbdc; public NPZ SHA25697b83eafe79854dca2b85079e4ac5bc4a1768a7bff3fc9106d25ad2c94332740. Observations are not custom or author simulation. No true force/stress reference exists; inferred method results are labelled baseline references.

Paper1: Daniels,Kollmer,Puckett2017, DOI10.1063/1.4983049, journal method; preprint arXiv1612.03525 and journal text were checked for static force+torque conditions, SectionIVC/Eq21 and Eq19 pixel mismatch. Paper2: Lee et al.2025, arXiv2507.01933, PeGS2 update. Source https://github.com/Jfeatherstone/pepe remotely verified master4cc073750e477fcb491e711dea9bc870af0df276 on2026-10-20. Source data https://github.com/photoelasticity/PeGS2 remotely verified commitceb064470919aa92b63e7eebbde6f4a3535313be. PeGS2 diskSolve defaults maskradius0.96 and sample script metresperpixel=.007/160,fsigma140 are preserved in the selected constrained branch. Original inverse forceOptimize and forward genSyntheticResponse/evaluateStress are independently executed.

**LICENCE RESTRICTION:** measured PeGS2 image data are CC BY-NC-SA4.0. Noncommercial and sharealike terms apply; no unrestricted commercial use or relicensing is asserted. Python mathematical source pepe is separately MIT. NOTICE preserves full terms and contributor attribution. This task is prepared for private scientific evaluation; no external publication was performed.

## Observation-only geometry and calibration

Red-channel radial rim gradients are sampled from radius60–85pixels at0.25pixel spacing, excluding20–25degree sectors around contacts where neighbours obscure the rim. Gaussian radial smoothing2samples precedes negative-gradient peak detection. scipy.optimize.least_squares with soft_l1 loss,f_scale1pixel fits the circle. Centre=[93.26971254206686, 90.04002272424479],radius=72.92296540262036pixels; edge residual RMS1.1238769514385976pixels. geometry.npz freezes measured edge points/fit; it contains no force truth. Raw colour observations suffice to reproduce/replace this geometric annotation. Circle fit derives no target calibration.

The earlier trial centre90,90 exactly coincided with an idealized contact point and triggered one NaN in the unmodified original stress response, which nan_policy='omit' transformed into an empty scalar objective. A half-pixel trial enabled diagnosis but is **not the final geometry**. The final observation-derived centre removes the ambiguity independently. Original singularity failures are archived, not erased. Final radius and nominal spatial calibration remain floating point; the original forceOptimize int truncation is explicitly corrected instead of silently inherited. f_sigma140 is a nominal script assumption, not independent calibration. Notebook126/154N/m experiments reveal its nearly proportional force-scale ambiguity.

## Maintainer archive index

Private archive is `imaging-101-maintainer/research_batch2_20261020/material_mechanics/`, outside this task. It holds pepe.zip/master snapshot+pepe_revision.txt,pegs2.zip/exact source,original JPGs,photoelastic_2017.pdf/photoelastic_2025.pdf when downloaded,capture_photoelastic*.py,photoelastic_oracle.npz/log,photoelastic_forceonly_oracle.npz/log,photoelastic_integer_failure.log,photoelastic_geometry.npz/json,photoelastic_balanced_fullmask_oracle.npz/results.json,photoelastic_balanced_zero_normal_oracle.npz/results.json,photoelastic_balanced_normal_floor_results.json,photoelastic_balanced_oracle.npz/log,photoelastic_balanced_results.json,photoelastic_balanced_stress_oracle.npz/log,photoelastic_corrected_success_criteria.md,isolated oracle-env,temporary executed notebook and final figure exports. Each historical/failed file is retained for diagnosis; no archived solver is imported by shipped source/tests/Judge/notebook.

## Original run and physically necessary correction

Original unmodified forceOptimize exact-observation run preceded local implementation. Command `oracle-env/Scripts/python.exe capture_photoelastic.py`, directional F+alpha fit, four fixed beta,method nelder,maxEvals3000,forceBounds0..1,useToleranceFalse,allowAdd/RemoveFalse,imageScaleFactor1,balanceWeight0. Original runtime11.54940600018017s,888evaluations,bestSSE217.03947017019794. This diagnostic used the earlier half-pixel geometry and integer model sampling; it is not the final benchmark reference. It satisfies the mandatory independent-original execution gate, but its mechanics fail: normalized net force0.08690472731429351 and torque0.09640610853338794.

Independent source inspection found objectiveFunction calls testForceBalance(force,beta,alpha), while signature is(force,alpha,beta); also this penalty does not impose torque. Static equilibrium is a physical requirement of the2017 published derivation, so the unbalanced result was rejected. The final branch retains the original elastic/optical forward and pixel SSE, restores the published three equilibrium constraints and uses the published PeGS2 inner0.96-radius fit mask. General SLSQP enforces these explicit constraints, with85degree angular numerical bounds excluding the collapsed-chord pure-tangential degeneracy. This is an identified correction and permitted optimizer port, **not unchanged forceOptimize reproduction**.

Before local corrected implementation, `photoelastic_corrected_success_criteria.md` fixed: static maxabs<1e-9N,at least25% SSE improvement over zero,three-start objective spread<0.1% and stress NRMSE<0.02,forward atol2e-10/rtol1e-8,objective-gradient relative1e-5,constraint-Jacobian atol1e-8,parameter relative1e-3 and image NRMSE1e-4. Earlier full-rim/degenerate-angle searches failed stability and remain archived. No metric tolerance was relaxed. The final published-mask/nondegenerate solver meets the original criteria.

Final independent oracle command `oracle-env/Scripts/python.exe capture_photoelastic_balanced.py`, followed by capture_photoelastic_balanced_stress.py, completed exit0. Original unmodified genSyntheticResponse is called on the same observed ROI with the observation-fitted geometry and nominal calibration. Static matrix,analytic constraint derivatives and centred1e-6N objective gradient are defined in the independent private oracle before local extraction. Final three starts.02/.04/.07N all successful,objective166.19705636099616/166.19705634802318/166.19705633218433,max equilibrium residual6.94e-18N,zeroSSE517.0640695772449,total runtime6.2216215000953525s. The primary.04N state defines the baseline. The complete original evaluateStress.py_func is observed with sys.setprofile to capture its local principleStressDiff at returns, without changing any source equation; its intensity agrees with original JIT to about1e-12. Frozen independent stress maps are therefore actual original intermediates, not locally generated expected values.

Oracle environment Windows x64,CPython3.11.11,NumPy2.4.6,SciPy1.17.1,Numba0.68.0/llvmlite0.50.0,lmfit1.3.4,private pepe source; original transitive visualization dependencies isolated. CPU threads1,noGPU. Cleaned runtime is separate shared Python with NumPy/SciPy/Matplotlib only. No domain package or oracle import is required in final runtime.

## Equation/function/test mapping

| Published stage | Upstream role | Local function | Validation |
| --- | --- | --- | --- |
| Colour leakage/contrast | PeGS2 contactDetect image convention | preprocessing.prepare | exact frozen input_intensity |
| Geometric disk/calibration | sample metre scale + measured red outline | CENTER,RADIUS,stress_basis | independent geometry fixture and±1pixel experiments |
| Elastic concentrated-contact field | pepe evaluateStress;2017 arbitrary-contact derivation | physics_model.stress_basis | frozen original intensity and observed original tensor intermediate |
| Stress-optic law | genSyntheticResponse/evaluateStress;2017 Eq6 | intensity_from_stress/forward | original forward,homogeneity and range checks |
| Pixel mismatch | forceOptimize objective;2017 Eq19 | solvers.objective | independent corrected original-forward scalar and gradient |
| Inner fit support | PeGS2 diskSolve maskradius0.96 | objective disk_mask | exact fixed convention,full-field scoring retained |
| Force+torque balance |2017 SecIVC/Eq21;source penalty defect investigated | constraint_matrix,decode | analytic Jacobian,independent linear matrix,net force/torque tests |
| Constrained inverse | published constrained nonlinear fit;explicit corrected SLSQP oracle | fit_forces/reconstruct | independent corrected final quantities+three initial states |
| Metrics/physical Judge | independently known cases | Judge+visualization | every key/error/poor/evaluator/report-write contract test |

## Fresh evidence and limitations

Fresh full reconstruction runtime1.1940814999397844s in the actively executed notebook. Against independent constrained oracle: stress NCC0.9999999999896098/NRMSE3.5079598483889804e-8; forces NCC0.9999999999922002/NRMSE3.4668101953935152e-6; angles NCC0.9999999999893701/NRMSE4.772940447602352e-6. All use uncentred cosine/dynamic-range RMS. Independent geometry/initialization perturbations are not hidden alignment. Reference arrays are physically constrained inferred baselines, never actual truth.

Fresh static normalized net force and torque are about1e-17. Unconstrained same-geometry optical fit gives0.12149 net-force fraction and0.23319 torque fraction, demonstrating why optical SSE alone is insufficient. Three fresh balanced initial states give166.19705633–166.19705637 SSE and stress NRMSE<5.5e-8. Centre row±1pixel and radius±1pixel change stress NRMSE0.00754–0.00767; nominal calibration±10% changes loads nearly±10%. These are identifiability limits, not experimental error bars. Point-contact stresses are large near the rim; only figure colour is clipped at99th percentile and explicitly labelled. Saved/Judge arrays remain unchanged.

`evaluation/baseline_reference.npz` keys reconstruction,forces,contact_directions come from the independent corrected original-forward oracle and captured original stress. Notebook upstream.npz is the labelled comparison reference. Fixtures balanced_upstream.npz (original-forward stages,physical matrix/gradient,multistart) and balanced_stress.npz (observed original Python intermediate) serve comparing tests; geometry.npz serves observation-geometry tests. fresh_measurements.npz records only actively computed review quantities and is consumed by final figure exports.

## Mandatory gate ledger

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/source/data/license/equation mapping | passed | immutable revisions,hashes,table above |
| Unmodified original exact-observation run | passed | original run retained,mechanical failure explicitly rejected |
| Explicit corrected published physical inverse | passed | SecIVC/Eq21 constraints,published fit mask,independent original-forward constrained oracle |
| Independent parity and physical/Jacobian checks | passed | exact data,forward/stress/gradient fixtures,equilibrium and unit checks |
| Constants/calibration/geometry fidelity | passed | original nominal values,observation-only circle fit,no hidden target |
| Standard entry point and all-quantity Judge | passed | reconstruction.npz and pass report,physical static metrics included |
| Strict malformed/poor/evaluator/report failures | passed | unit tests for every required quantity |
| Solver-visible self-contained/no leaks | passed | two public array keys+README complete units/calibration,zero GitHub URL leakage |
| Minimal shared runtime/local math | passed | standard library,NumPy,SciPy,Matplotlib;no domain imports |
| Full tests | passed |26passed2.97s on final numerical branch; exact command below |
| Active full notebook and comparisons | passed | successful nbconvert,all ablation/calibration/three-start/geometry computations active |
| Canonical notebook includes outputs | passed | final successfully executed canonical notebook includes all3figures/numeric outputs |
| README/plans semantic review | passed | equations,signs,units,constraints,source correction/defaults/signatures reviewed |
| Inventory/private archive/license | passed | task template;source/data restrictions in NOTICE;private logs/checkouts separate |

Commands: shared Python -m src.main; evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; -m pytest test/ -q --basetemp=<unique private directory>; -m jupyter nbconvert --to notebook --execute notebooks/photoelastic_granular_stress.ipynb --ExecutePreprocessor.kernel_name=research --ExecutePreprocessor.timeout=180 --output photoelastic_executed.ipynb --output-dir=<private archive>. Explicit shared-Python research kernel is selected through private JUPYTER_PATH. Temporary executed copies and QA screenshots are never left in task output.

Exact source identity audit: private pepe_exact_revision_verification.json verifies Git blob SHA1 of SyntheticPE.py,ForceSolve.py,ForceBalance.py and LICENSE.md against remotely queried commit4cc073750e477fcb491e711dea9bc870af0df276; master archive matches these files exactly.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 3 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 26 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. CC BY-NC-SA4.0 image; nominal calibration; original inverse defect corrected explicitly with independent unmodified-forward oracle, not unchanged optimizer reproduction.

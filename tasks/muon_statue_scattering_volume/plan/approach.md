# Implemented mathematics and selected benchmark

The input is the entire author GEANT4/CRY six-plane statue acquisition:522963 events, XYZ in mm, float32. Source event energy is deliberately unused because the selected author POCA branch uses `use_p=False`. No synthetic generator is required. The returned field is angular RMS in rad, not calibrated mass density. No physical voxel truth is available.

## Line fitting and qualification

For each tracker, arrange its3 hits as rows of a3×3 matrix H. Set p=mean(H,axis=rows), center B=H-p, and compute B=U diag(s) V^T using full_matrices=False. The first row of V^T is the orthogonal least-squares line direction u. Float32 arithmetic is preserved. The oracle uses source chunks of200000; local chunks of50000 reduce Windows MKL batched-SVD workspace pressure. Three full incoming-track reruns at50000,100000 and200000 each were bitwise equal to the oracle (private backend audit). This is an execution partition, not a change in fitted observations or SVD mathematics. If u_z>0, negate ALL components. The unchanged source negated only z: this changes a fitted line for certain SVD sign choices. This minimal correction restores the published incoming/outgoing line geometry.

Compute theta=acos(clamp((u dot v)/(norm(u)norm(v)),-1+1e-12,1-1e-12)). Float32 rounds the endpoint tolerance to zero at unity; clipping still prevents invalid acos. The unchanged source used sum(abs(u_i*v_i)), which can suppress physical sign-changing deflections. Directed downward vectors make the geometric dot product unambiguous.

Keep norm(v×u)>=1e-7. The unchanged source required every absolute cross-product component to exceed1e-7 and therefore removed valid lines aligned with a coordinate plane. The corrected criterion preserves the same published nonparallel geometry; it is not a new imaging model. Original and qualified source copies,3-line diff, frozen independent measurements and analytic counterexamples are private. No unchanged-source parity claim is made for the default corrected method.

## Effective interaction and finite volume

Let incoming line be p+t u, outgoing q+s v and normal w=v×u. Solve [u,-v,w](t,s,a)^T=q-p. Let L=p+t u and R=q+s v. The POCA midpoint is M=L+(R-L)/2 in mm. The two endpoints have a connector perpendicular to both lines. Exclude M outside x[-180,180],y[-400,400],z[-860,-460], inclusively. This is the original fixed VOI, not a target-derived mask.

The grid has36×80×40 cells of side10mm. Cell centers are min+(index+.5)*10. For each coordinate, sorted-edge search with right insertion minus1 implements the source AABB predicates. At an internal shared edge the original repeated assignment chooses the larger cell index; the global maximum remains in the final cell. This equivalent local discretization avoids the original large event×all-voxel boolean masks. Full numeric indices and counts are validated against the qualified oracle.

## Angular aggregation

On the RETAINED in-volume event list, let T=quantile(theta,.999), using the original PyTorch linear quantile. Set b_n=min(theta_n,T)^2. At voxel v, c_v counts events and s_v sums b_n. Set r_v=sqrt(s_v/c_v) when c_v>0 and0 otherwise. Then Q=quantile(r,.999) over ALL115200 voxels including zeros, and output q_v=min(r_v,Q). Neither quantile uses physical truth. The output is float32(1,36,80,40), axes batch,x,y,z, rad. Counts are int64 diagnostic measurements. Empty-cell zeros reflect lack of retained events and are not evidence of vacuum.

There is no iterative inverse optimizer, objective normalization, regularizer, step size or early stopping: line SVD,3×3 solves and direct voxel statistics evaluate the selected method. Numerical conditioning threshold1e-7 and both quantiles.999 are explicit fixed method parameters. No smoothing, angle lower cut or momentum reweighting is silently added. The author's example notebook also offers detector smearing/loss and other algorithms; this bound benchmark selects the ideal stored-hit unweighted POCA branch. The paper's count-density display is actively compared with angular RMS but is not mislabeled as the same physical unit.

## Validation and comparisons

Frozen stage tolerances were fixed before extraction: directions rtol2e-6/atol2e-7; centroids rtol1e-7/atol1e-4mm; angles rtol2e-6/atol1e-6rad; POCA rtol2e-5/atol.002mm; voxel indices exact; angular field rtol2e-6/atol2e-7rad. Independent skew-line midpoint/perpendicularity, directed line residual, quadrant-changing angle, nonparallel geometry, shared-edge discretization and count conservation guard against copying known errors.

The notebook actively runs the full corrected reconstruction and three source-error ablations, comparing each with its identified original oracle. It also computes angular-quantile sensitivity, occupancy maps, slice comparisons and full-data source-error statistics. Every fresh method is calculated from raw detector hits. Statistical comparisons use the same output grid; no alignment or intensity fitting is performed.

Evaluation uses every voxel and the benchmark's uncentered cosine NCC and reference-range NRMSE. Hidden reference is the qualified independent author-code run. Fixed boundaries NCC>=.999999 and NRMSE<=5e-6 represent float32 extraction tolerance, supported by repeat/precision checks recorded in validation. They are unrelated to zero self-error and do not claim physical-density accuracy. Invalid input returns Judge error2; valid poor output returns1; pass returns0.

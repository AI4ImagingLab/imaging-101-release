# Shipped interfaces and execution flow

The standard entrypoint is `python -m src.main`; paths resolve relative to that module's task root. `main()` sets PyTorch CPU thread count1, loads only data/raw_data.npz, executes the full qualified local solver, writes output/reconstruction.npz, and returns the unbatched float32 image. It does not change a Matplotlib backend. The independent standalone Judge resolves hidden assets beside its own file and interprets CLI submission/report paths relative to the caller.

## Source modules

| Function/signature | Inputs and defaults | Return / side effects |
|---|---|---|
| preprocessing.load_data(path) | NPZ path; detector_hits float32(1,522963,6,3), mm, axes batch,event,plane,xyz | Validated float32(522963,6,3); finite/key/shape/dtype errors raise |
| physics_model.fit_tracks(hits,corrected=True,chunk_size=50000) | float32(3,3,N), axes xyz,tracker-plane,event; mm | tuple(direction,centroid), each float32(N,3); directions dimensionless; centroids mm |
| scattering_angles(incoming,outgoing,corrected=True) | float32(N,3) downward directions | float32(N,) rad |
| nonparallel_mask(incoming,outgoing,tol=1e-7,corrected=True) | same directions | bool(N,) selection |
| closest_approach(points_in,points_out,tracks_in,tracks_out) | float32(N,3) centroids mm and directions; caller already excludes parallel lines | float32(N,3) midpoint mm |
| in_volume(points) | positions(N,3), mm | bool(N,) inclusive box mask |
| assign_voxels(points) | positions inside box, mm | int32(N,3) xyz indices; out-of-volume raises |
| aggregate_rms(indices,angles,angle_quantile=.999,prediction_quantile=.999) | int32(N,3), float32(N,) rad | tuple(float32(36,80,40) rad,int64(36,80,40) counts); no retained event raises |
| solvers.reconstruct(hits,return_stages=False,correct_direction=True,correct_angle=True,correct_mask=True,angle_quantile=.999,prediction_quantile=.999) | float32(N,6,3), mm | image; if return_stages=True also dict of named tracks/centroids/angles/masks/POCAs/indices/counts |
| visualization.metrics(a,r) | same-shaped finite arrays; no implicit alignment | dict ncc,nrmse; float64 evaluation |
| visualization.slice_panel(ax,image,z_index,title,vmax=None,cmap='viridis',signed=False) | unbatched xyz field in rad; supplied axis | Matplotlib image; displays XY section in mrad with physical mm coordinates |
| main.main() | no arguments | image; writes output/reconstruction.npz |

`return_stages` keys: tracks_in,tracks_out,points_in,points_out each(N,3); angles(N,); parallel_mask(N,) then volume_mask(length selected nonparallel); pocas(M,3),retained_angles(M,),indices(M,3),counts(36,80,40). M is the retained geometric event count, not a change to the supplied acquisition. Original correction switches are only notebook audit helpers. No cross-task imports, upstream imports, caches, domain reconstruction library or trained parameters are used.

## Notebook and hidden assets

The six-chapter notebook loads public input and identified original measurement arrays, actively calls the cleaned solver for fresh full reconstructions, scores them and runs the independent Judge. It computes scientific comparisons and sensitivities; precomputed data is used only as labeled original reference. Its optional environment-controlled export location writes PNG/SVG/PDF and numeric source privately. This is a maintainer export, not a solver input or dependency.

`evaluation/baseline_reference.npz` is the qualified original full field, never truth. `notebooks/reference_outputs/unchanged_original.npz` is the unchanged original field/counts for the explicit correction audit. `test/fixtures/upstream.npz` contains qualified nondegenerate native-hit stages and analytic skew-line cases, consumed by parity/geometry tests. The Judge independently implements strict float32(1,36,80,40) NPZ loading and metrics; it never imports src. Both required metrics must pass. CPU is validated; GPU is neither required nor assumed.

No synthetic observations are generated, so no generate_data.py exists. Private source repositories, paper files, original logs, data archives and capture scripts remain outside the final task.

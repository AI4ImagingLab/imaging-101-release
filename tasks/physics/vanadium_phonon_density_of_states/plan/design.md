# Shipped interfaces and execution

All numerical arrays are NumPy arrays. Data loading strips only the explicit singleton batch dimension; it does not round original observations or replace stored missing measurements. No upstream domain package is imported.

| Module and callable | Inputs and defaults | Return and role |
| --- | --- | --- |
| preprocessing.load_observations(path) | NPZ path; exact five keys | intensity/variance float64(300,149), q(300), energy(149); validates original precision, mask and measured variance |
| preprocessing.preprocess(intensity,variance,q,energy) | Native unbatched arrays | observed/variance float64(300,95), NaN missing boundaries |
| physics_model.gamma0(energy,g,beta=BETA,step=1.) | Nonnegative increasing positive grid and density | Scalar inverse-meV thermal integral with original zero extrapolation |
| physics_model.spectral_orders(energy,g,orders=4,beta=BETA,step=1.) | Positive grid and density | Reflected energy and spectra(orders,2N-1); direct central linear convolutions |
| physics_model.debye_waller(q,energy,g,beta=BETA,step=1.) | Momentum and density | Dimensionless recoil exponent per Q |
| physics_model.kinematic_valid(q,energy) | Q and energy arrays | Bool accessible-neutron-triangle array |
| physics_model.forward(g,q,observed_mask) | g(80),q(300),bool mask(300,95) | Dictionary sp/mp/ms/total, each(300,95); full 240-bin extension and all four orders |
| solvers.initial_density() | None | Normalized quadratic/constant initial g(80) |
| solvers.single_update(corrected,errors,q,prior) | Corrected map/variance(300,95), q, prior(80) | Updated normalized density and propagated variance(80) |
| solvers.reconstruct(intensity,variance,q,energy,max_iterations=20,tolerance=1e-4) | Native unbatched observations | Dictionary reconstruction/variance(80),history(K,80),stages list,processed_intensity/processed_variance(300,95) |
| main.run(task_root=None) | Optional task directory; default derived from file | Fresh dictionary; creates output/reconstruction.npz with batch-first reconstruction(1,80) |

The standard entry point python -m src.main calls run. It loads data, executes all selected inverse updates, and writes only the declared reconstruction. The Notebook calls the same fresh pipeline and a separate tighter reconstruction; it loads source measurements solely for labeled comparisons. Its figures are generated from the live data and fresh reconstruction, not copied plots. The source-reference arrays and fixtures have distinct consumers documented in validation.md.

The standalone evaluation/judge.py uses only its own public-model mathematics and the raw acquisition map, never imports src, and resolves its evaluator assets relative to its own file. CLI prediction and report paths follow the caller's working directory. It validates exact keys, numeric real dtype, batch-first shape and finite prediction values; preserves legitimate NaN observations behind the explicit mask; rejects malformed metrics; and exits0/1/2 for pass/scientific fail/error. The baseline_reference.npz is a labeled standard-method reference for maintainer comparisons, not actual truth and not a forced identity target. The Judge does not need it to compute the physical criterion.

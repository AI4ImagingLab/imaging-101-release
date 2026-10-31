# Public-input benchmark audit

Scope:320 task directories in eight domain folders. The original59 contents remain protected. The prior240 later tasks retain their existing audit; Batch13 adds21 independently validated inverse tasks. Public means available to the evaluated Agent, not authorization to publish or relicense datasets.

Current status: **240 ready**, **21 blocked**, **59 legacy-exempt**.

Ready tasks supply observation/calibration definitions, physical parameters,
units, coordinate conventions, unknowns, and every required NPZ output. They
do not supply the reference solver, hidden truth, fitted answer or source-code
download links. Forward-model definitions and mathematical target constraints
are retained; solver recipes and fixed finite iteration paths are not required.

The machine-readable [task registry](task_registry.json) is authoritative.
The export command refuses the audited blocked tasks. Original tasks remain
legacy-exempt rather than being silently certified by this audit.

| Domain | Total | Original preserved | Ready | Blocked |
| --- | ---: | ---: | ---: | ---: |
| astronomy | 41 | 9 | 29 | 3 |
| physics | 41 | 7 | 33 | 1 |
| biology | 40 | 9 | 27 | 4 |
| mathematics | 30 | 0 | 27 | 3 |
| chemistry_material_science | 40 | 3 | 34 | 3 |
| mechanics | 35 | 1 | 33 | 1 |
| medicine | 55 | 23 | 30 | 2 |
| earth_science | 38 | 7 | 27 | 4 |

Verification evidence:

- All 240 standalone Judges were actually executed after relocation with the
  current computed submissions, from a foreign working directory, using absolute
  trusted Judge paths and caller-relative report paths. All returned exit 0.
  This interface check does not override scientific blocking reasons below.
- Per-task malformed submissions and valid poor-quality controls have separate
  native reports. Error status must have empty metrics; poor-quality finite
  submissions return fail rather than masquerading as input errors.
- Reference-based tasks distinguish method agreement from physical truth.
  Repairs use task-specific forward consistency, objectives or justified gauges;
  actual ground truth was not replaced with a method reconstruction.
- Independent alternative controls include public-only SOLA, Vesta and sediment
  reconstructions. These are maintainer diagnostics, not 240 blind Agent trials.
- The full ADJUST reconstruction and active notebook with its comparisons were
  executed. Their logs and new outputs are distinct from historical oracle runs.
- The relocation inventory verifies exact frozen bytes, original task contents,
  all bound observations and actual ground truth. Versioned method-reference and
  computed-output changes are listed separately with private provenance.
- Changed README/plans and notebook narratives passed actual MathJax and KaTeX
  rendering checks. Directory-depth changes to figure export paths were checked
  separately without claiming another complete numerical notebook execution.
- Public export and trusted evaluation regression tests cover private-file
  exclusion, submission import isolation, native NPZ scoring and error reports.
  Docker boundaries were checked with tests; a real Docker/LLM run is not claimed.

The private evidence directory is
`../../imaging-101-maintainer/benchmark_readiness_20261029/`: per-task reviews,
`TASK_READINESS.csv`, `final_native_evaluation/summary.json`,
`relocated_inventory_verification.json`, `final_public_leak_review.json`,
`math_changed_all/`, group validation logs, and versioned scientific controls.

## Tasks withheld from the local benchmark

These packages remain available for maintainer research. Passing their stored
reference submission through a Judge does not prove that an independent Agent
can infer the target from the public inputs. Missing calibration, an undefined
conditional target, or an unsuitable inverse formulation is not repaired by
relaxing reference tolerances or inventing physical information.

| Task | Reason |
| --- | --- |
| [brillouin_water_spectral_shift](../tasks/physics/brillouin_water_spectral_shift/README.md) | The1e-5 reference checks include calibration arc coordinates/extracted spectra, but the public camera model does not uniquely define the circle/spot selection functional. Need an observation-defined camera calibration objective and arc gauge, or independent geometric calibration; ordinary physical peak fitting does not certify intermediate-array identity. |
| [calcium_cnmf_demixing](../tasks/biology/calcium_cnmf_demixing/README.md) | No physiological component truth exists. Detection/spatial/AR/motion factor identity is compared against one nonconvex registered CNMF result, while no complete rank/regularization/source-selection objective fixes the accepted decomposition. Need a publicly defined joint movie/AR/motion objective, consistent gauges, and observable-quality acceptance. |
| [diffractive_lens_chromatic_radiance](../tasks/mathematics/diffractive_lens_chromatic_radiance/README.md) | Unknown spatially varying color blur and cross-channel reflectance do not uniquely identify a sharp RGB image. Current1e-5 reference image matching defines one local transfer pipeline, with no public unique blind-deconvolution prior/objective. Need an explicit joint kernel/image prior and calibrated observable criterion, or independent PSF/sharp validation. |
| [dipyridyl_relative_electron_density](../tasks/chemistry_material_science/dipyridyl_relative_electron_density/README.md) | Measured amplitudes alone admit distinct phase structures; a validated structural prior/equivalence criterion is unresolved. |
| [hertha_multimodal_shape_spin](../tasks/astronomy/hertha_multimodal_shape_spin/README.md) | Only conditional method shapes exist. Highly precise vertex/spin/nuisance checks bind a finite Gauss-Newton surface and heuristic missed-chord residual; the physical observations support shape/nullspace alternatives. Need independent full AO/photometry/occultation plus shape-prior objective acceptance, including explicit visibility/missing-chord conventions. |
| [in_vivo_ultrasound_microvascular_mapping](../tasks/medicine/in_vivo_ultrasound_microvascular_mapping/README.md) | No independent bubble PSF, metric pixel-pitch or identifiable localization/track target is supplied; legacy strict catalogue identity depends on finite detection/link/tie choices. README marks uncertified and preserves only physical input/output/conceptual hints. A fair calibrated localization benchmark needs additional independent acquisition calibration or true localization evidence. |
| [landsat_two_source_energy_partition](../tasks/earth_science/landsat_two_source_energy_partition/README.md) | The current evaluation requires a finite approximate PT/stability protocol whose result is not a uniquely defined physical closure solution. |
| [magnetic_microscopy_dipole_inversion](../tasks/earth_science/magnetic_microscopy_dipole_inversion/README.md) | The method-selected source catalog is not identified by the public dipole forward model alone. |
| [masaya_uv_so2_column_transects](../tasks/earth_science/masaya_uv_so2_column_transects/README.md) | The public spectrum has unknown instrument shape and broadband/wavelength nuisances, but does not fully define their finite functional family/bounds while enforcing1e-4 reference-column agreement. Need explicit asymmetric line-shape/broadband/stretch parameterization and an observable-fit or fully specified profile-likelihood target. |
| [nascent_transcription_promoter_dynamics](../tasks/biology/nascent_transcription_promoter_dynamics/README.md) | No promoter-state truth exists. Reference posteriors and emission/noise parameters depend on finite beam truncation and nonconvex EM, without a complete public approximate-likelihood/beam-domain target. Need a clearly defined posterior inference/likelihood objective, label gauge and acceptance permitting improved supported fits. |
| [nonlocal_haze_radiance_reconstruction](../tasks/mathematics/nonlocal_haze_radiance_reconstruction/README.md) | One hazy RGB image does not identify both radiance and transmission without a certified prior/independent target; strict legacy haze-line/display identity is not physical truth. README marks uncertified and legacy finite recipe is maintainer-only. A fair scientific criterion needs an independently justified model/prior or measured target. |
| [perovskite_dual_contrast_atomic_columns](../tasks/chemistry_material_science/perovskite_dual_contrast_atomic_columns/README.md) | No independent column-profile truth exists. Covariance and peak tolerances bind local fit apertures, subtraction/background and sublattice assignments that are not uniquely defined by the public Gaussian image model. Need complete image/profile fitting domains/background nuisance objective and amplitude-consistency acceptance. |
| [protein_joint_cd_ir_structure](../tasks/biology/protein_joint_cd_ir_structure/README.md) | A validated public statistical target for the discrete self-consistent solution ensemble is unresolved. |
| [raw_burst_radiance_reconstruction](../tasks/mathematics/raw_burst_radiance_reconstruction/README.md) | Heterogeneous exposure, unknown warps and only an exact legacy integer merge result do not independently identify clean radiance. A finite registration/merge identity cannot certify an arbitrary valid inverse without independent clean target or fully specified conditional calibrated objective. README marks uncertified and legacy recipe is maintainer-only. |
| [scattering_microrheology_modulus](../tasks/mechanics/scattering_microrheology_modulus/README.md) | Derivative/modulus reference identity depends on an unspecified intercept-window selection and Gaussian local-regression bandwidth. The physical Siegert/GSER laws alone do not determine a noisy lag derivative or its uncertainty. Need a complete smoothing functional/resolution convention and observation-consistent acceptance. |
| [sentinel1_volcano_sbas](../tasks/earth_science/sentinel1_volcano_sbas/README.md) | Circular phase coherence does not identify whole-cycle corrections or affine/deformation/elevation splits. Five1e-5 reference-field checks still bind one closure/connected-component integer solution and finite deramping procedure without a method-independent integer/spatial prior objective. Need explicit gauge/regularization and full unwrapped observation objective, or independent displacement calibration. |
| [sofi_live_cell_fluctuations](../tasks/biology/sofi_live_cell_fluctuations/README.md) | Fluctuation-statistic calculation, explicitly outside strict latent-inverse benchmark selection. |
| [solar_aia_dem_mapping](../tasks/astronomy/solar_aia_dem_mapping/README.md) | Six channels underdetermine20 DEM bins; no true DEM exists.1e-6 DEM matching binds data-adaptive response weights, discrepancy parameter search and positivity processing. Need the full public regularization/discrepancy functional or an independently evaluated nonnegative thermal-fit/prior objective accepting improved solutions. |
| [solar_coronal_pfss](../tasks/astronomy/solar_coronal_pfss/README.md) | Boundary-value extrapolation, explicitly outside strict latent-inverse benchmark selection. |
| [ub199_hysteretic_switching_density](../tasks/chemistry_material_science/ub199_hysteretic_switching_density/README.md) | The derivative/conditional uncertainty from finite noisy FORC records is not uniquely defined by the physical mixed derivative.1e-8/1e-6 reference identity binds full lower-branch/background/evidence/taper processing. Need an explicit local observation/model-selection functional with method-independent covariance/derivative validation or a declared reference-measurement calibration. |
| [ucla_bold_hemodynamic_neural_deconvolution](../tasks/medicine/ucla_bold_hemodynamic_neural_deconvolution/README.md) | The blind HRF/input target depends on a finite event/lag-selection protocol whose independent physiological estimand is unresolved. |

Use [the public benchmark guide](PUBLIC_BENCHMARK.md) to export only README,
raw observations and the environment into an isolated solver workspace and to
score `output/reconstruction.npz` on the trusted maintainer host. Source, plans,
notebooks, test fixtures, evaluation references and existing outputs stay private.
Existing task-specific local-research and external redistribution limitations
remain in their notices and provenance; scientific readiness is not a license grant.


## Batch13 addition

All21 new packages passed independent parent source/physics, public-contract, full reconstruction, standalone Judge and active notebook review. Their integrated release exports and trusted CLI evaluations passed separately. The private [batch report](../../imaging-101-maintainer/research_batch13_20261030/report/REPORT.md) and [acceptance ledger](../../imaging-101-maintainer/research_batch13_20261030/release_postflight.json) retain actual commands, numeric evidence and limitations. All prior299tasks/7494protected files remain unchanged. An unbound legacy placeholder remains unverified rather than being assigned an invented scientific definition. This is not21blindAgenttrials or a new certification of old blocked tasks.

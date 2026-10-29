# Shipped interfaces and execution

preprocessing.load_observations(path="data/raw_data.npz") returns nine vectors of length110935: eight original float32 observations and int64 pixel_index. It validates completeness and finite values. prepare_forcing(inputs) returns full LAI, canopy mask, zenith, vapour pressure, roughness, displacement and height, plus canopy SnC/SnS(length110929) and bare Sn(length6). Units follow the README. Inputs are not modified.

physics_model.py contains 44 explicit selected mathematical functions. Prefixes met/rad/res/MO/wnd/CI identify physical roles and source traceability. The fixed PT/OSEB, KN99 resistance and Goudriaan wind mathematics is local. Unselected SW/PM/4SAIL/Massman paths are absent; unsupported selector values raise rather than silently choosing another model. This is not a wrapper or opaque vendored framework.

solvers.reconstruct(path="data/raw_data.npz",nominal_height=.3) returns (result,diagnostics). The five-key result uses the exact batch-first float64 public shapes and units. Diagnostics retain all18 PT states, nine OSEB states and preprocessing quantities. The primary public nominal height is .3m; only labeled supplemental experiments use .27/.33m. Every input-valid pixel remains. Invalid or out-of-range source flags raise; absent-canopy zeros occur only at the six known bare pixels. Reconstruction has no file side effects.

main.run(output_path="output/reconstruction.npz") performs the complete inverse, writes the NPZ, prints actual runtime and flags, and returns the result. python -m src.main is the standard CLI. Imports never change the plotting backend. visualization.image_from_pixels(values,pixel_index,shape=(336,335)) maps values for review: NaN background means missing acquisition, never a dropped fitted pixel. save_review_figure(figure,path) exports a computed figure at150dpi.

The notebook actively runs the full primary inverse/Judge and both complete height perturbations. Frozen original arrays are comparisons only. Corrected, historical uncorrected and historical cross-backend measurements remain separately identified. Source/PDF/GeoTIFF/oracle archives stay outside the package.

The independent Judge imports no solver. Assets resolve relative to itself; actual truth, if supplied, precedes a method reference. Public raw observations supply independent thermal reprojection. It validates exact float64 shapes, finite values, positive Kelvin temperatures and absent-canopy zero components, then evaluates every temperature/flux/G/radiation array and both complete energy budgets. Invalid metrics, boundaries, evaluator assets or finite-input overflow produce empty-metric error JSON. Native CLI exits0/1/2 for pass/fail/error. NPZ loading forbids pickle; prediction/report paths follow caller cwd. All test assets are under test/.

Actual selected signatures (arrays broadcast; source float32 casts are deliberate; units are in approach.md):

| Function | Arguments/defaults |
|---|---|
| CI_calc_omega0_Kustas | LAI, f_C, x_LAD=1, isLAIeff=True |
| CI_calc_omega_Kustas | omega0, theta, w_C=1 |
| MO_calc_L | ustar, T_A_K, rho, c_p, H, LE |
| MO_calc_Psi_H | zoL |
| MO_calc_Psi_M | zoL |
| MO_calc_mo_length_hv | ustar, T_A_K, rho, c_p, H, LE |
| MO_calc_u_star | u, z_u, L, d_0, z_0M |
| MO_psi_h_brutsaert | zol |
| MO_psi_m_brutsaert | zol |
| TSEB_OSEB | Tr_K, T_A_K, u, ea, p, Sn, L_dn, emis, z_0M, d_0, z_u, z_T, calcG_params=[[1], 0.35], const_L=None, T0_K=[], kB=TSEB_KB_1_DEFAULT |
| TSEB_TSEB_PT | Tr_K, vza, T_A_K, u, ea, p, Sn_C, Sn_S, L_dn, LAI, h_C, emis_C, emis_S, z_0M, d_0, z_u, z_T, leaf_width=0.1, z0_soil=0.01, alpha_PT=1.26, x_LAD=1, f_c=1.0, f_g=1.0, w_C=1.0, resistance_form=None, calcG_params=None, const_L=None, kB=TSEB_KB_1_DEFAULT, massman_profile=None |
| TSEB__L_diff | L, L_old |
| TSEB__check_default_parameter_size | parameter, input_array |
| TSEB_calc_F_theta_campbell | theta, F, w_C=1, Omega0=1, x_LAD=1 |
| TSEB_calc_G | calcG_params, Rn_S, i=None |
| TSEB_calc_H_C_PT | delta_R_ni, f_g, T_A_K, P, c_p, alpha |
| TSEB_calc_T_C_series | Tr_K, T_A_K, R_A, R_x, R_S, f_theta, H_C, rho, c_p |
| TSEB_calc_T_S | T_R, T_C, f_theta |
| TSEB_calc_resistances | res_form, res_types |
| TSEB_check_flux_lims | *fluxes, lims=TSEB_FLUX_LIMS |
| TSEB_monin_obukhov_convergence | l_mo, l_queue, l_converged, flag |
| met_calc_c_p | p, ea |
| met_calc_delta_vapor_pressure | T_K |
| met_calc_lambda | T_A_K |
| met_calc_psicr | c_p, p, Lambda |
| met_calc_rho | p, ea, T_A_K |
| met_calc_stephan_boltzmann | T_K |
| met_calc_sun_angles | lat, lon, stdlon, doy, ftime |
| met_calc_vapor_pressure | T_K |
| rad__calc_taud | x_lad, lai |
| rad_calc_K_be_Campbell | theta, x_lad=1, radians=False |
| rad_calc_L_n_Campbell | T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1 |
| rad_calc_Sn_Campbell | lai, sza, S_dn_dir, S_dn_dif, fvis, fnir, rho_leaf_vis, tau_leaf_vis, rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln, x_LAD=1, LAI_eff=None |
| rad_calc_difuse_ratio | S_dn, sza, press=1013.25, SOLAR_CONSTANT=1320 |
| rad_calc_potential_irradiance_weiss | sza, press=1013.25, SOLAR_CONSTANT=1320, fnir_ini=0.5455 |
| rad_calc_spectra_Cambpell | lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=1, lai_eff=None |
| res_calc_R_A | z_T, ustar, L, d_0, z_0H |
| res_calc_R_S_Kustas | u_S, deltaT, params=None |
| res_calc_R_x_Norman | LAI, leaf_width, u_d_zm, params=None |
| res_calc_roughness | LAI, h_C, w_C=1, landcover=res_CROP, f_c=None |
| res_calc_z_0H | z_0M, kB=0 |
| wnd_calc_A_Goudriaan | h_C, LAI, leaf_width |
| wnd_calc_u_C_star | u_friction, h_C, d_0, z_0M, L=float('inf') |
| wnd_calc_u_Goudriaan | u_C, h_C, LAI, leaf_width, z |

# Implemented mathematics: four-state calibrated ADR polarimetry

## Bound real acquisition and calibration

The selected real Kazansky birefringent calibration specimen is deposited in recOrder_test_data.zip, 2022_08_04_recOrder_pytest_20x_04NA. Retain firsttimepoint, all3positions, all16planes, all128x256 native pixels and four analyzer states. This is not a cell or a synthetic object. Each defocus plane is an effective 2D retarder image, not tomographic local material properties. No newly generated observations are involved. Native lateral pitch is unavailable because page PixelSizeUm=0; use detector-pixel lateral coordinates. Actual TIFF page hardware labels20x0.55NA dry,10ms,532nm override inconsistent stale calibration-MicroscopeParameters defaults60x/NA1.3; no invented corrected pitch is introduced.

Public independent empty-field images and the calibration instrument matrix A are included unchanged. A is float64 as deposited calibration JSON; it is cast to float32 for the original numerical inverse. Background TIFFs are float64 and are cast to float32 after loading. Subtract independently measured black level100.58ADU from both sample and background. No target-fitted scale or extra privileged observation is used. Absolute optical power and background exposure are not deposited, so output T is explicitly relative intensity. Bound sample contains no ADC65535 saturation; all finite noisy retention P>1 values are retained.

## Acquisition and Stokes definitions

For circular incident light, the specimen Stokes vector s=[T,TP sin(delta) sin(2theta),-TP sin(delta) cos(2theta),TP cos(delta)]. Here delta is weak-branch retardance0..pi/2rad; theta is slow-axis angle0..pirad, modulo pi; T is relative background-normalized intensity and P is retained polarization. Detector x increases right and detector y increases down; no flip/rotation is applied. This sign convention is independently checked against complex Jones matrices for a lossless linear retarder. In the absence of depolarization, the norm of polarized components equals T, and for general P it equals TP.

Camera model: I=A Mbg s+d*1+eta, where I is four-state ADU, Mbg is spatial background Mueller map, d is dark offset,1 is four-vector of ones and eta is noise. Empty-field Stokes vector g=A^+(Ibg-d1) constrains Mbg under the same attenuating/depolarizing-retarder assumption. All matrices are pixel-local; no spatial kernel, FFT, interpolation, smoothing, boundary extension or wraparound is used.

## Explicit inverse, including measured-background matrix

Compute instrument analysis C=pinv(A) in torch.float32 with original default Moore–Penrose tolerance. Sample Stokes a=C(I-d1); background g=C(Ibg-d1). Let g=(g0,g1,g2,g3), and q=g1^2+g2^2. Construct the original ADR Mueller matrix:

M00=g0; M11=(g0*g2^2+g1^2*g3)/q; M12=M21=g1*g2*(g3-g0)/q; M13=g1; M22=(g0*g1^2+g2^2*g3)/q; M23=g2; M31=-g1; M32=-g2; M33=g3. All other entries are zero. Matrix storage is explicitly float32, matching original waveorder. Invert every4x4 matrix with torch.linalg.inv, then s=Mbg^-1 a for each position/plane. This model's direct formula is singular for a perfectly isotropic background q=0; the independently deposited calibration has nonzero q and is nonsingular throughout. No numerical epsilon is silently added.

For corrected s=(s0,s1,s2,s3), set L=sqrt(s1^2+s2^2+s3^2). Recover delta=asin(sqrt(s1^2+s2^2)/L), theta=(atan2(s1,-s2) modulo2pi)/2, T=s0, P=L/s0. Keep arrays float32. In particular, do not clamp P to1, impose a hidden retardance mask, rotate/flip theta, or use abs for signed Stokes components. Output keys reconstruction,orientation,transmittance,depolarization all have shape(1,3,16,128,256). Relative transmission is not absolute calibrated material attenuation; P outside0..1 is a visible noisy estimate rather than a physically admissible hidden correction.

This is a closed-form analytic inverse and per-pixel linear-system evaluation. There is no optimized loss, regularizer, initialization, gradient update, step-size, line search or stopping rule. Original apply_inverse_transfer_function defaults remove_estimated_background=False,project_stokes_to_2d=False,flip_orientation=False,rotate_orientation=False are retained. The notebook's optional background ablation sets correct_background=False; dark-offset sensitivity reruns at80.58 and120.58ADU hold raw data fixed and do not modify the standard100.58 setting.

## Evaluation and visual conventions

All4quantities are required. Per-array relative-L2 is used for delta,T,P. Slow-axis angles are pi-periodic; compute wrapped difference wdelta=atan2(sin(2(theta_est-theta_ref)),cos(2(theta_est-theta_ref)))/2. The angular RMSE is sqrt(sum[sin(delta_ref)^2*wdelta^2]/sum[sin(delta_ref)^2]) in radians. Continuous retardance weighting reflects loss of angle identifiability at zero birefringence; no pixel is excluded. All arrays must be finite, correct realdtype/shape, and angle must lie0..pi. There is no registration or intensity fitting.

Glyphs are display-only, subsampled every16pixels and shown only when displayed retardance>0.03rad; this is not an evaluation mask or saved-array modification. Main maps share physical color limits across positions. The active notebook reruns all3x16planes and actual no-background/dark-offset experiments, plots all4required quantities, full defocus trends and retention>1 failures. Reference is independently executed waveorder on exactly these data, not real material truth. Phase-from-defocus, deconvolution, and full permittivity-tensor models are distinct and outside this selected ADR inverse.

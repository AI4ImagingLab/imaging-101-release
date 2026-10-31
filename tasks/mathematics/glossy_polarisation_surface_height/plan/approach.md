# Implemented polarimetric height method

This task binds the full author sample: 457 rows, 235 columns, 36 analyser images, 74,331 foreground pixels. There is no downsampling, crop or generated target. Raw intensities and analyser angles remain float64. Masks are boolean encodings of the unchanged author indicators. The selected method is the known-direction, unknown scalar illumination case of the ECCV 2016 height-from-polarisation implementation, including its mixed diffuse/specular treatment.

## Coordinates and forward model

Image rows increase downward and columns rightward. Physical image coordinates x,y increase rightward and upward, respectively. All three spatial coordinates x,y,z use one pixel as the unit. Height z has no independently known metric depth scale. Slopes p=∂z/∂x, q=∂z/∂y define the normal ν=(-p,-q,1)/sqrt(1+p²+q²). The view direction is v=(0,0,1). The refractive index is η=1.5 and the known light direction is s=(2,0,7), without normalisation. The fitted illumination vector ℓ is a positive scalar multiple of s and includes albedo and intensity. This is not per-pixel albedo inference.

For analyser angle α, the intensity is I=U[1+ρ cos(2α−2φ)]. U is mean unpolarised intensity; ρ is degree of linear polarisation; φ is phase modulo π. The design matrix has rows [1,cos(2α),sin(2α)]. An unweighted least-squares fit gives b0,b1,b2. The implementation preserves the upstream expressions Imax=b0+sqrt(b1²+b2²), Imin=b0−sqrt(b1²+b2²), U=(Imax+Imin)/2, ρ=(Imax−Imin)/(Imax+Imin), φ=mod(atan2(b2,b1)/2,π). Background intermediates are zero. No gamma, noise whitening, centering, median filtering or target-derived calibration is introduced.

## Fresnel inversion

Diffuse degree of polarisation is

ρd(θ)=(η−η⁻¹)²sin²θ/[2+2η²−(η+η⁻¹)²sin²θ+4cosθ sqrt(η²−sin²θ)],

where θ is the zenith angle and cosθ=νz. `diffuse_zenith` evaluates the exact algebraic inverse in the source. Its numerator is A=2ρ+2η²ρ−2η²+η⁴+ρ²+4η²ρ²−η⁴ρ²−4η³ρ sqrt(−(ρ−1)(ρ+1))+1. Its denominator is D=η⁴ρ²+2η⁴ρ+η⁴+6η²ρ²+4η²ρ−2η²+ρ²+2ρ+1. The result is θ=acos(min(real(sqrt(A/D)),1)). Complex intermediate square roots and the real-part clipping preserve the upstream numerical convention.

Specular degree is ρs(θ)=2sin²θ cosθ sqrt(η²−sin²θ)/[η²−sin²θ−η²sin²θ+2sin⁴θ]. The upstream table samples θ=0,0.01,… up to the final sample below π/2. Retain only the increasing branch through its maximum and linearly interpolate θ against ρs. No extrapolation is allowed. The bound sample stays inside the table range. Replace θ at the supplied specular pixels with this inverse and replace their phase by mod(φ+π/2,π). Diffuse phase is unchanged.

## Illumination and ambiguity

Illumination estimation uses diffuse foreground only. For each such pixel form the row B=[sinφ sinθ, cosφ sinθ, cosθ] and the ambiguity matrix T=diag(−1,−1,1). Start ℓ=s. Select BT for pixels where (Bℓ−U)²>(BTℓ−U)² and B otherwise, giving P. Fit only the scalar multiplier of ℓ:

ℓnew=ℓ [(Pℓ)ᵀU]/[(Pℓ)ᵀ(Pℓ)].

Alternate assignment and fitting until ||ℓnew−ℓ||₂<10⁻⁹ or the upstream counter exceeds 100; thus at most 101 updates are evaluated. The exact sample converges after 9 updates with ℓ approximately (0.08305138,0,0.29067983). Full precision is retained in fixtures, never this rounded text. No random initialization is used for this known-direction case.

## Sparse height equations and boundary discretization

The unknown vector z contains foreground heights in row-major order. Each diffuse pixel supplies phase cosφ p−sinφ q=0 and ratio −ℓx p−ℓy q=min(1,U)/cosθ−ℓz. The intensity clipping `min(1,U)` occurs only in height-system assembly, after light fitting, exactly as upstream.

Let H=(ℓ+v)/||ℓ+v||₂. Each specular pixel instead supplies p=−Hx/Hz, q=−Hy/Hz and the phase constraint cosφ p−sinφ q=0. H uses the actual fitted ℓ as supplied to the published code, not a newly normalised light vector. This numerical convention is explicitly preserved despite the physical approximation inherent in treating a light/albedo-scaled vector in the halfway relation.

The mask is padded by one zero pixel. At an interior pixel with all eight neighbours, Dx uses weights (−1,−4,−1)/12 on the left column and (1,4,1)/12 on the right column. Dy uses (1,4,1)/12 on the upper row and (−1,−4,−1)/12 on the lower row. These approximate derivatives with unit spacing and y upward. When all axial but not all diagonal neighbours exist, use ordinary centered axial differences. At other boundaries, each axis uses centered differences if both neighbours exist and one-sided differences toward an existing neighbour otherwise. A photometric equation is omitted only if one coordinate axis has no available neighbour. There is no periodic wraparound or implicit rectangular-domain integration.

Pixels with all four axial neighbours also contribute λ(zleft+zright+zup+zdown−4zcenter)=0, with λ=0.1. Thus the squared objective has effective Laplacian weight λ²=0.01. Rows are emitted in pixel order, smoothing first, then each pixel's phase/ratio or specular equations. The final equation fixes the first row-major foreground pixel to zero. The full matrix C has 222,682 rows, 74,331 columns and 1,395,562 nonzero entries.

## Objective, solution and stopping

Minimise ||Cz−d||₂² with no division by the number of equations and no factor needed for this closed-form linear minimiser. No artificial iterations or additional regularizer are present. The upstream sparse QR solves C\d. The cleaned implementation evaluates the equivalent normal equations (CᵀC)z=Cᵀd through sparse SuperLU. On the full sample, independent comparison to upstream QR gives maximum absolute height error below 10⁻⁹ pixel and relative L2 below 5×10⁻¹¹. A normal-equation gradient residual checks stationarity. The anchored system is nonsingular for the bound mask. This direct solver has no initial height, learning rate, line search or stopping tolerance. Normal equations can lose precision on other ill-conditioned datasets; this claim of validated accuracy is limited to the bound benchmark and independently tested analytic cases.

## Output and evaluation

Store float64 height[1,457,235] with zero background, no centering, rescaling or postprocessing. The original source uses NaN outside the mask; changing only that excluded background to zero makes the submission finite. Masked NCC is cosine similarity without mean subtraction. Masked NRMSE is RMS difference divided by the reference dynamic range. The hidden baseline is the upstream method output, explicitly not physical truth. Two upstream executions agree exactly; an independent local algebra/linear-solver extraction agrees within 10⁻⁹ pixel. Boundaries NCC≥0.999999 and NRMSE≤10⁻⁷ allow numerical representation differences, including float32 submission quantisation; zero, sign reversal and 1% scale errors fail. The notebook reports fresh local/upstream metrics, residuals, a λ=0 ablation and controlled intensity-noise sensitivity. No noise experiment is presented as physical ground truth.

# Implemented magnetic microscopy method

## Acquisition and unknowns

The bound real QDM image has shape (600,960), 2.35 µm centre spacing and sensor height 5 µm. Surface z=0, z increases upward. The unknowns are 75 dipole positions p=(x,y,z) in µm and moments m=(mx,my,mz) in A m². Raw Bz is supplied in nT. No physical source truth is available. Euler local backgrounds are nuisance parameters, not required output quantities.

For R=r-p in metres and radius ρ=||R||, the dipole upward field in tesla is c[3Rz(m·R)/ρ^5 - mz/ρ^3]. Sum over sources for global prediction. The coefficient row with respect to m is c(3RxRz/ρ^5,3RyRz/ρ^5,3Rz²/ρ^5-1/ρ³). Here c=1.25663706212e-6/(4π), preserving the upstream CODATA vacuum permeability. Coordinate conversion is 1e-6 m/µm, field conversion 1e9 nT/T, no magnetic-moment normalization. Positive-down inclination is atan2(-mz,hypot(mx,my)); declination is atan2(mx,my).

## Fourier preprocessing and gradients

Continue Bz upward by h=5 µm to sensor z=10 µm. For frequencies kx=2π*fftfreq(960,2.35), ky=2π*fftfreq(600,2.35), multiply each forward FFT coefficient by exp(-h*sqrt(kx²+ky²)); apply NumPy inverse FFT, retain the real part. Use the unpadded periodic image and its zero mode unchanged. NumPy's forward FFT is unnormalized; inverse has factor 1/(600*960).

Differentiate the continued field f along the supplied coordinate arrays using centred finite differences internally and one-sided first-order boundary differences. For the vertical derivative use [U_s(f)-U_-s(f)]/(2s), s=mean(|dx|,|dy|)=2.35 µm. U_h denotes the continuation above, including negative displacement for downward continuation. This central difference preserves the executable upstream convention despite the source comment claiming a forward difference. Gradients are nT/µm. Total-gradient amplitude T=sqrt(fx²+fy²+fz²). Map T to [0,1] by clipping to its 1st/99th percentiles then affine scaling.

## Scale-space detection

Use ten linearly spaced Gaussian standard deviations 10..100 µm, divided by pixel spacing. Compute scale-normalized response -σ² GaussianLaplacian(Tstretch,σ), reflective image boundaries and SciPy default Gaussian support. Find maxima over the 3×3×3 neighbourhood in row/column/scale, strictly greater than 0.025. Exclude a border floor(30/2.35)=12 pixels in image dimensions, no scale-border exclusion. Sort descending by response using stable sorting. Build a cKDTree to enumerate candidate overlapping circle pairs within 2*maxσ*sqrt(2). In the original pair order remove the smaller overlapping blob, ties removing the earlier index; zero overlap is required. For zero overlap the exact disk intersection formula reduces to centre distance < sqrt(2)*(σ1+σ2). Construct each square half-width σ*sqrt(2)*spacing*1.3. This produces 100 windows, matching the upstream execution exactly.

## Euler position and full-vector moment

For all N points in a window define G rows (fx,fy,fz,3) and h=x*fx+y*fy+z*fz+3f. The homogeneity equation (r-p)·∇f=3(b-f) gives G(p_x,p_y,p_z,b)^T=h. Solve the four-parameter unweighted least-squares normal equations (G^T G)v=G^T h. Coordinates stay in µm, gradients nT/µm, background nT. Structural index is exactly 3; there is no regularizer, centring or coordinate scaling. Window selection includes all sensor centres inside its bounds.

Subtract b from f, convert to tesla, form the N×3 dipole coefficient matrix A above at the estimated position and sensor z=10 µm, and solve (A^T A)m=A^T d with a positive-definite solver. This is a closed-form linear solve, with no initialization, learning, line search or iterative stopping criterion. Keep the published normal-equation discretization. Estimate residual e=d-Am and covariance [sum(e²)/(N-3)]*(A^T A)^-1. Define R2=1-sum(e²)/||d-mean(d)||² and SNR=10log10(var(d,ddof=1)/var(e,ddof=1)), in dB. Discard positions z>0; keep all remaining 75 for output. For interpretation only, identify R2>=0.85 and SNR>5 dB as good sources.

## Evaluation and limits

Independent upstream output supplies the hidden standard-method reference, not ground truth. Match sources by minimum summed squared 3-D location differences; apply that same assignment to signed moment vectors. Position RMSE averages all 225 coordinate differences. Moment error is ||M-Mref||/||Mref||, with no mean subtraction or rescaling. Both boundaries are 1e-7 in their respective units, nonzero float64 reproducibility tolerances. They assess reference reproduction, not the validity of the magnetic dipole approximation. The notebook additionally shows measured/continued field, all-source prediction, residual, all/reliably fit magnetic directions, source depths and R2/SNR distributions.

The paper's verbal interpretation of SNR=5 as fivefold signal/noise disagrees with the implemented dB statistic: 5 dB corresponds to variance ratio 10^0.5. Preserve the executable definition and label it in dB. The full-image sum omits local constant backgrounds intentionally, as in the upstream comparison. Local estimates can interfere when summed, so a global noise-weighted chi-square is not justified without calibrated experimental noise.

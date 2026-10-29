# Implemented angular-field reconstruction

## Binding and conventions

The benchmark binds all16 author-supplied real bunny HDR frames at native2048×2448 resolution. No crop, subsampling, tone mapping or reconstructed-path support enters inversion. Intensities retain relative instrumental radiance units; no absolute radiometric/pixel-pitch calibration is invented. Generator g and analyzer c use the numeric recorded radians. Mechanical viewing handedness is not independently stated in metadata.

The paper main-text rotation equations and its appendix/source use different illumination-angle signs. The source's named forward component corresponds to the sum-angle family in the actual stored numeric coordinates, and its reverse component to the difference-angle family. Independent trigonometric/Jones tests verify these angular models and explain how a local propagation basis can reverse handedness. They do not establish unmeasured mechanical hardware directions or certify unique bounce paths. The benchmark therefore defines all five original degrees of freedom by precise harmonic fields and preserves the original source mapping in provenance.

## Forward model and linearization

Let U be a real polarization-independent field and Cs=sr+i si, Cd=dr+i di be complex radiance coefficients at one pixel. The measured intensity is

```math
I(g,c)=\tfrac12(U+|C_s|+|C_d|)
+\tfrac12[s_r\cos2(c+g)+s_i\sin2(c+g)
+d_r\cos2(c-g)+d_i\sin2(c-g)].
```

All five real unknowns have the same instrumental radiance unit. Write the linear coefficient vector x=(x1,x2,x3,x4,x5). A measurement row of W is

```math
W(g,c)=(1,\cos2c\cos2g,\sin2c\sin2g,
\cos2c\sin2g,\sin2c\cos2g).
```

For each pixel, the selected published objective is the unweighted sum of squared residuals ||Wx−y||² across the16 supplied states. Multiplying this objective by1/16 would leave its minimizer unchanged; no such normalization enters the native pseudoinverse. The design has rank5 and condition number2. There is no regularizer, spatial coupling, hidden support, boundary extension, initialization, iteration budget or stopping rule: this is a direct closed-form inverse.

## Exact original solve and five-field parameterization

Input preprocessing retains the original clip(y,0,None); no current raw sample is negative. Values as large as155.8404998779297 remain in inversion. Images become float64 in(row,column,state) layout. Compute Wplus=np.linalg.pinv(W), with the original NumPy default singular cutoff and no overridden rcond, then X=tensordot(Wplus,images,axes=(1,-1)). This is the selected native method, not a general optimizer substituted for a custom iteration.

The source's component formulas are

```math
s_r=x_2-x_3,\quad s_i=x_4+x_5,\qquad
d_r=x_2+x_3,\quad d_i=-x_4+x_5,
```

```math
A_s=\sqrt{s_r^2+s_i^2},\quad A_d=\sqrt{d_r^2+d_i^2},\quad
\phi_s=\tfrac12\operatorname{atan2}(s_i,s_r),\quad
\phi_d=\tfrac12\operatorname{atan2}(d_i,d_r),\quad
U=2x_1-A_s-A_d.
```

The output stores(U,sr,si,dr,di), preserving both amplitudes and phases without arbitrary phase at a zero amplitude. This is an invertible coordinate change, not dropping unknowns: x1=(U+As+Ad)/2, x2=(sr+dr)/2, x3=(dr−sr)/2, x4=(si−di)/2, x5=(si+di)/2. The independent original reference derives the same Cartesian fields from its actual returned amplitudes/half-angle phases; maximum difference from direct source x quadratures is7.1054e−15. Native signed U is retained, including its negative0.1008875% fraction. No positivity projection or gain alignment is introduced.

## Physical validation and approximation error

Analytic Malus-law mixtures independently recover every unknown, test total angular-mean power and positive radiometric scaling. Explicit one/two Jones-mirror calculations distinguish sum/difference responses in declared local bases. Generator-angle finite differences verify both quadrature signs. Native W/pseudoinverse, nonconstant real crop X and all five original amplitude/phase quantities have independently captured numeric fixtures. Full-field original repeated runs are bit-identical, and local output must satisfy the predeclared1e−8 relative/scaled absolute criteria without changing them after extraction.

Five basis functions exclude analyzer-only and generator-only modulation. Real-scene global relative observation residual is0.15243698953044157. Independent nine-term linear Mueller fitting, adding cos2c,sin2c,cos2g,sin2g, reduces it to0.02948077533200727. Matrix condition numbers are both2; the discrepancy is not explained by numerical ill-conditioning. Pixel median relative residual falls from0.0770132 to0.0138323. This restricted-model error remains visible and quantified. The selected output remains the original five-field decomposition, not an asserted exact path separation or a replacement nine-parameter inverse.

## Evaluation and measured comparisons

Each complete output channel is compared using ||estimate−reference||2/||reference||2. A nonzero1e−6 boundary was frozen before local results; measured valid float32 serialization errors lie between1.24689e−8 and2.52227e−8. The independent angular-forward residual uses sqrt(sum(predicted−observed)²/sum(observed²)) over all16 full images. Its frozen boundary is1.01times original residual plus1e−6, or0.153962359425746. These limits neither demand a false zero model residual nor accept inaccurate field substitutions. No NCC/DR-NRMSE is silently mislabeled; all five signed channel norms are nonzero. There is no truth file for this real experiment; an unchanged author execution supplies the method reference.

Notebook comparisons actively reconstruct the same complete data and preserve original amplitude/phase/distribution views. A nine-basis control quantifies omitted transport terms. Fixed angle-only selections5/8/12/16 are frozen before local code: five-state indices[5,7,9,10,14] minimize condition number over all5-row subsets; eight/twelve keep the first2/3 generator settings with all analyzer states. Full-data/native-reference errors and held-out observation behavior are labeled accordingly, never truth error. Additional full-grid Gaussian perturbations use0.5%,1%,2% of each frame's measured spatial RMS, fixed diagnostic RNG31917, and native lower clipping with clipping fraction measured. They are numerical sensitivity experiments, not estimated instrument noise or newly bundled observations.

The sensitivity-only observation draws are explicitly implemented in src/generate_data.py. The helper returns float64 images after one framewise Gaussian draw/add per state, without clipping; reconstruct retains lower clipping afterward. It never creates the real primary acquisition or writes raw_data.npz. Moving the existing generation block into this module preserves exactly the frozen seed, order, standard deviations, subsequent clipping and full-field results.

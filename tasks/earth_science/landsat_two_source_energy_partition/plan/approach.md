# Actual mathematics and correction protocol

This is explicitly corrected reproduction of mature original PT/OSEB, not unchanged source. All110935finite acquired pixels enter;110929 have vegetation and 6 are bare. Native float32 bands are retained. Earlier geeet defects, uncorrected pyTSEB source, exact uncorrected local equality and independent forcing-failure evidence remain private. Complete corrected original+repeat were frozen before local correction; strict numerical boundaries remain unchanged.

NDVI gives LAI=-ln(1-clip(NDVI-.05,0,1))/.5 at originalfloat32precision. Public h=.3m,leafwidth=.1m,greenfraction1,clumpingcover1,spherical leaf parameter1 are assumptions, not target-fitted calibration. PressurePa becomesmb by/100. Vapour pressure=6.112exp[17.67(Tdew-273.15)/(Tdew-29.65)]mb. Specific humidity q=.622ea/[p+(.622-1)ea], cp=(1-q)1003.5+q1865J/kg/K;rho=100p/(287.04Ta)[1-(1-.622)ea/p]. Latentheat lambda=10^6[2.501-.002361(Ta-273.15)]J/kg. The saturation-pressure slope is computed in kPa/K and multiplied by10 to mb/K before comparison with gamma=cp*p/(.622*lambda), also in mb/K. No spatial operator, training, simulated generator or ground truth exists.

Solar geometry uses30.25N,38.25E,day111,localtime10.98,standard meridian45E; declination=.409sin(2pi doy/365-1.39), original equation-of-time/hour-angle conventions. For solar cosine mu, airmass m=1/mu, V=1320(1-.5455),N=1320*.5455W/m², restored Weiss horizontal equations are Bv=Vexp[-.185(p/1013.25)m]mu;Dv=.4(Vmu-Bv);w=1320*10^[-1.195+.4459log10m-.0345(log10m)²];Bn=max(0,[Nexp(-.06(p/1013.25)m)-w]mu);Dn=max(0,.6[(N-w)mu-Bn]).
Original1313.25,logcoszen,andcross-band/incorrectprojection expressions are explicitly corrected. ESSD2023formalSupplementTableS4part4 writes Bn*m before finaldivision bym, resolving the normal/horizontal ambiguity in Peng2023C24/C26. Water absorption increases with optical path; sea-level nadir visible beam fraction isexp(-.185). All other source empiricalcloudindex/fraction expressions remain, including capped clear fractions and power.6667.

Canopy beam extinction K(theta)=sqrt(x²+tan²theta)/(x+1.774(x+1.182)^(-.733)). Diffuse transmittance integrates2exp[-K(theta)LAI]cos(theta)sin(theta)dtheta by original5-degree rectangle quadrature;kd=-ln(tauDiffuse)/LAI. For leaf absorptivity a=1-rhoLeaf-tauLeaf, r=(1-sqrt(a))/(1+sqrt(a)),deep-canopy reflectance b=2kr/(k+1), E=exp(-sqrt(a)kL), transmission=(b²-1)E/[b rhoSoil-1+b(b-rhoSoil)E²],F=(b-rhoSoil)E²/(b rhoSoil-1),albedo=(b+F)/(1+bF). Apply beam/diffuse k and angularly clumped/effective LAI respectively. Canopy shortwave sums(1-transmission)(1-albedo)incident over visible/NIR and beam/diffuse;soil sums transmission*(1-rhoSoil)*incident. Bare soil uses spectral soil absorption only.

Thermal emission LC=epsilonC sigma Tc^4,LS=epsilonS sigma Ts^4. Longwave canopy albedo aL/transmission tL come from diffuse RT with thermal optics:LnS=epsilonS*tL*Ldown+epsilonS*(1-tL)*LC-LS;LnC=(1-aL)(1-tL)(Ldown+LS)-2(1-tL)LC. Primitive NaN limiting guards are retained; no selected final solution containsNaN and no invalid fit is zero-filled. Bare Ln=epsilonS(Ldown-sigmaTr^4).

Nadir view fraction f=1-exp(-K(0)*Omega0*LAI),with Omega0 derived from gap fraction fc exp(-KLAI/fc)+(1-fc);fc=1 here. Source float32 final view fraction and angular clumping remain. Thermal mixing uses fourthpowers. Ts=[(Tr^4-fTc^4)/(1-f)]^.25;a negative/NaN radicand gets the original invalid flag, never a scored-pixel mask. With canopy Hc, the series network gives
TcLinear=[Ta/Ra+Tr/(Rs(1-f))+Hc Rx/(rho cp)(1/Ra+1/Rs+1/Rx)]/[1/Ra+1/Rs+f/(Rs(1-f))].
D=TcLinear(1+Rs/Ra)-Hc Rx/(rho cp)(1+Rs/Rx+Rs/Ra)-Ta Rs/Ra.
dTc=[Tr^4-fTcLinear^4-(1-f)D^4]/[4(1-f)D³(1+Rs/Ra)+4fTcLinear³].
Tc=TcLinear+dTc is the actual single Newton correction, not a substituted exact nonlinear solver.

Crop z0=h/8,d=.65h,z0H=z0(kB0). u*=.41u/[ln((zu-d)/z0)-PsiM((zu-d)/L)+PsiM(z0/L)].
Ra=[ln((zT-d)/z0H)-PsiH((zT-d)/L)+PsiH(z0H/L)]/(.41u*).
Source Brutsaert stable/unstable corrections remain explicit. Goudriaan wind U(z)=Uc exp[-a(1-z/h)],a=.28LAI^(2/3)h^(1/3)leafwidth^(-1/3).
Rx=90/LAI sqrt(leafwidth/U(d+z0));Rs=1/[.0038max(Ts-Tac,0)^(1/3)+.012U(z0soil)].
Minimum wind/u*.01m/s and resistance.1s/m are original physical regularization, not post-hoc output clipping.
Tac=(Ta/Ra+Ts/Rs+Tc/Rx)/(1/Ra+1/Rs+1/Rx);Hs=rho cp(Ts-Tac)/Rs. The correct Tac network is active, unlike the rejected geeet variable update.

Initialize alphaPT1.26;reduce by.1 while daytime soil LE is negative. Hc=RnC[1-alpha fg slope/(slope+gamma)];update Tc,Ts,resistances,Tac,Hs. InitiallyG=.35RnS,LEs=RnS-G-Hs,LEc=RnC-Hc. If alpha reaches zero,set LEc/LEs=0 and Hs=min(Hs,RnS-G),G=max(G,RnS-Hs)for energy closure. This is the selected daytime dry constraint, not suppression of condensation in all scenes. Preserve branches0initial PT,3reduced alpha,5dry. Out-of-range252/invalid flags are rejected,never masked.

Initialize L=infinity. Virtual sensible flux Hv=H+.61Ta cp LE/lambda;L=-rho cp Ta u*³/(.41*9.8*Hv);positive heated daytime flux gives negative L,zero Hv gives neutral infinity. Recompute u*;maximum15 outer iterations,relative L threshold.001,six-value queue checks2/3 cycles. Exact float32 intermediate casts and default broadcasting remain. The initial inf/inf relative-L warning is retained and disclosed; all final outputs are finite. No optimizer or loss is substituted.

ForLAI zero,standard OSEB uses Tr=Ts with no Tc. Rn=soilSn+epsilonS(Ldown-sigmaTr^4),G=.35Rn,H=rho cp(Tr-Ta)/Ra,LE=Rn-G-H,withsource daytime zero-LE closure and stability iteration. All six bare pixels return flag10. Output float64 packages original values without changing the mathematics.

Fixtures precede extraction;corrected full source/repeat precedes local correction. Pure-function tolerances1e-6relative/1e-5 absolute;final temperature.001K/flux.1W/m²;radiometric.001K;per-component closure.005W/m².All arrays and pixels are scored. Notebook height.27/.33mfull refits,source correction,anduncorrected cross-backend comparisons are supplemental and never calibrate primary thresholds.

Brutsaert stability functions use zeta=z/L. For finite zeta>=0 both corrections are -6.1 log[zeta+(1+zeta^2.5)^(1/2.5)]. For zeta<0 set y=-zeta. Heat correction is (.943/.78)log[(.33+y^.78)/.33]. Momentum uses a=.33,b=.41,x=(y/a)^.333333, then caps y at b^-3 (the source computes x before this cap). It evaluates log(a+y)-3b*y^.333333+(b*a^.333333/2)log[(1+x)^2/(1-x+x²)]+sqrt(3)b*a^.333333 atan[(2x-1)/sqrt(3)]+psi0, where psi0=-log(a)+sqrt(3)b*a^.333333*pi/6. Neutral or nonfinite inputs have zero corrections. The rounded exponent and exact cap order are preserved numerical conventions, not silently changed formulas.

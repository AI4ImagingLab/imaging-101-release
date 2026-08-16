# MEG relaxation matrix (Martinsson 1993)

Gamma_ji = alpha * (P/P0) * (295/T)^0.5 * (1-e^-m)/(1-e^{-mT/295})
           * [ (1 + 1.5*(hc/k)*E_i/(T*sigma)) / (1 + 1.5*(hc/k)*E_i/T) ]^2
           * e^{-beta*dE*(hc/k)/T}

with hc/k = 1.44, P0 = 1.01325 (atm), reference T = 295 K, [alpha,beta,sigma,m]=[0.0231,1.67,1.21,0.1487] for N2.

Detailed balance: Gamma_ij = Gamma_ji * (2i+1)/(2j+1) * e^{dE*(hc/k)/T}.
Only even |i-j| couplings are populated. Diagonal: Gamma_ii = -sum_k Gamma_ki (column-sum rule). Matrix size js=30.

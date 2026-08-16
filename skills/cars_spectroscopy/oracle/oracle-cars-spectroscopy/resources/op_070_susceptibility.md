# Resonant susceptibility (G-matrix mixed lines)

Grid: nu_s = arange(nu_mod[0]-10, nu_mod[-1]+10, 0.05). nu_mod = nu*nu_stretch + nu_shift.

Branches = (0, +2, -2) [Q, S, O]; vibrational v in {0,1}; js=30.

Per (branch, v):
  K = diag(line_pos(v, j, branch)) + i*Gamma
  (eigvals, V) = eig(K); Vinv = inv(V)
  del_pop = pop_factor(T,v,j,branch)
  d = sqrt(cross_section)   # branch-dependent, see line-structure
  term = (d @ V) * (Vinv @ diag(del_pop) @ d)
  chi_rs += sum_j 0.5 * term[j] * conj(-nu_s + eigvals[j]) / ((-nu_s+Re eig)^2 + (Im eig)^2)

Final scale: chi_rs /= (2*pi*c), c = 2.9979e10 cm/s.

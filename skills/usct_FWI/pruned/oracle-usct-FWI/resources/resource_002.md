# NCG + More-Thuente line search structure

## Outer NCG (max_iters=3)
- ub = 1/v_bounds[0] (=1/1300), lb = 1/v_bounds[1] (=1/1700), broadcast to x shape.
- fk, gk = fun(x0).
- it==0: pk=-gk; ak=0.01; gkTgk=<gk,gk>; is_init=True.
- it>0: gkTgk=<gk,gk>; gk_diff=gk-gkold; bk = <gk,gk_diff>/gkTgkold if gkTgkold>0 else 0; bk=max(0,bk) if bk<0; pk=-gk+bk*pkold.
- Save gkold=gk, pkold=pk, gkTgkold=gkTgk.
- a0 = 1.0 on first line search else previous ak.
- Call cvsrch(fun, xk, fk, gk, a0, pk, ub, lb, maxfev=5) -> xk,fk,gk,ak,info,nfev.
- g2normnx = norm(gk)/N; stop if < 1e-13; after it>0 stop if |fk-fkold|/max(|fkold|,eps) <= 1e-10.

## cvsrch (More-Thuente) constants
xtol=1e-15, ftol=1e-4, gtol=1e-2, stpmin=1e-15, stpmax=1e15, xtrapf=4.0, p5=0.5, p66=0.66.
- dginit = <conj(g.T.flatten()), conj(s.flatten())>; if dginit>=0 return (no descent).
- Bracketing loop; each trial: stp clamped to [stpmin,stpmax]; x = clip(wa + stp*s, lb, ub); (f,g)=fcn(x); dg=<conj(g),conj(s)>; ftest1 = finit + stp*ftol*dginit.
- Exit codes info: 1 success (f<=ftest1 and |dg|<=gtol*(-dginit)); 3 nfev>=maxfev; others per standard More-Thuente.
- Two-stage modified-function update (stage1) uses dgtest = ftol*dginit.

## cstep
Cubic/quadratic interpolation choosing next trial `stp` among four cases (fp>fx; sign change; |dp|<|dx|; else), with bracket safeguard `p66` shrink toward stx when bracketed. Returns updated interval (stx,fx,dx,sty,fy,dy,stp,...,brackt,info).

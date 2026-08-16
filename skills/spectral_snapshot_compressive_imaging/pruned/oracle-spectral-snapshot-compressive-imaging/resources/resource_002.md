# GAP schedule (standard run)

Config:
- lambda = 1, accelerate = True
- iter_max = 20 (per stage), tv_iter_max = 5
- sigma = [130]*8  (8 stages; TV weight per stage = sigma/255)
- global k counter; stop when k == 123 (124 inner iterations total)

Prior selection predicate use_cnn(k) is True exactly for:
  {83,84,85, 87,88,89, 91,92,93, 95,96,97, 99,100,101,
   103,104,105, 107,108,109, 111,112,113, 115,116,117,
   119,120,121, 122,123}
Otherwise use TV with weight sigma_stage/255 and max_num_iter=5.

Equivalent predicate:
```
def use_cnn(k):
    return ((k>=122 and k<=125) or (k>=119 and k<=121) or (k>=115 and k<=117) or
            (k>=111 and k<=113) or (k>=107 and k<=109) or (k>=103 and k<=105) or
            (k>=99 and k<=101) or (k>=95 and k<=97) or (k>=91 and k<=93) or
            (k>=87 and k<=89) or (k>=83 and k<=85))
```
CNN noise levels (low, mid, high) all = 10 (used as 10/255).
If no learned weights: use_cnn always False (TV-only), same 124-iteration budget.

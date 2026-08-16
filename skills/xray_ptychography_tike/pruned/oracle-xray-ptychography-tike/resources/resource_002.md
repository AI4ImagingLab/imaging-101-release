# Object canvas arithmetic (public case)

shifted max (float): row 351.9177, col 348.8925
ceil: 352, 349
probe footprint: 128 x 128
buffer: 2

    H = 352 + 128 + 2 = 482
    W = 349 + 128 + 2 = 479

=> psi shape (1, 482, 479) complex64, filled with 0.5+0j.

Bounds check with integer-truncated positions (positions = scan_shifted.astype(int32)):
- max(row) + 128 = 479 <= 482  (3-row margin)
- max(col) + 128 = 476 <= 479  (3-col margin)

So buffer=2 is exact and every patch slice `obj[r:r+128, c:c+128]` is in bounds;
no clipping or padding is needed.

If an implementation computes a larger canvas anyway, the alignment rule is:
keep the top-left origin, slice `phase[:482, :479]`. Never center-crop, and never
search over shifts to 'optimize' the frame.

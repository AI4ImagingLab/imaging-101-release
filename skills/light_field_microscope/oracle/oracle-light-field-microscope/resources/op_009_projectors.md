# Forward / backward projectors

img_shape = tex_shape = (n_lenslets*newSpacingPx,)*2
n_lenslets = max(1, round(base_n_lenslets * fieldOfViewScale))   # 30 here -> 450x450

## Lenslet grid (full)
coords = arange(n_lenslets)*newSpacingPx; coords -= coords[n_lenslets//2]
lenslet_rows, lenslet_cols = meshgrid(coords, coords, indexing='ij')

## Pattern ops (only where tex_mask[lr,lc])
for lr,lc with tex_mask true:
  dr = lr - tex_nnum[0]//2; dc = lc - tex_nnum[1]//2
  row_idx = lenslet_rows + tex_shape[0]//2 + dr; col_idx = lenslet_cols + tex_shape[1]//2 + dc
  img_r   = lenslet_rows + img_shape[0]//2 + dr; img_c   = lenslet_cols + img_shape[1]//2 + dc
  valid = in-bounds on all four; tex_indices = ravel_multi_index((row_idx,col_idx)[valid], tex_shape)
  img_indices = ravel_multi_index((img_r,img_c)[valid], img_shape)
  kernels[depth] from bank[lr][lc][depth]; adjoint_kernels[depth] = kernels[depth][::-1,::-1]

## forward_project(volume) -> (img_shape) float64
projection = zeros(img_shape)
for d in range(n_depths):
  plane_flat = volume[:,:,d].ravel()
  for op:
    sampled = plane_flat[op.tex_indices]
    if not any(sampled): continue
    sparse = zeros(img_shape).ravel(); sparse[op.img_indices] = sampled; sparse=sparse.reshape(img_shape)
    projection += fftconvolve(sparse, op.kernels[d], mode='same')
projection[projection<0]=0

## backward_project(projection) -> tex_shape+(n_depths,)
for d:
  back_flat = zeros(prod(tex_shape))
  for op:
    conv = fftconvolve(projection, op.adjoint_kernels[d], mode='same')
    back_flat[op.tex_indices] += conv.ravel()[op.img_indices]
  out[:,:,d] = back_flat.reshape(tex_shape)
out[out<0]=0
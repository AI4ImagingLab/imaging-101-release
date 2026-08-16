# Band-window learned denoiser: exact window slicing and CNN forward

## Per-band 7-channel input (numpy dstack, target band i, nC=31)
```
if i < 3:
    if i == 0: win = dstack(x[:,:,0], x[:,:,0], x[:,:,0], x[:,:,0:4])
    if i == 1: win = dstack(x[:,:,0], x[:,:,0], x[:,:,0], x[:,:,1:5])
    if i == 2: win = dstack(x[:,:,0], x[:,:,0], x[:,:,1], x[:,:,2:6])
    sigma_ch = low
elif i > nC-4:
    if i == nC-3: win = dstack(x[:,:,i-3:i+1], x[:,:,i+1], x[:,:,i+2], x[:,:,i+2])
    if i == nC-2: win = dstack(x[:,:,i-3:i+1], x[:,:,i+1], x[:,:,i+1], x[:,:,i+1])
    if i == nC-1: win = dstack(x[:,:,i-3:i+1], x[:,:,i],   x[:,:,i],   x[:,:,i])
    sigma_ch = mid
else:
    win = x[:,:,i-3:i+4]
    sigma_ch = high
# tensor: from_numpy(ascontiguous(win)).permute(2,0,1).float().unsqueeze(0) -> (1,7,H,W)
# Nsigma = full((1,1,1,1), sigma_ch/255.)
# output = model(tensor, Nsigma).squeeze().cpu().numpy()  -> (H,W)
```
All three of (low, mid, high) noise levels are 10 (i.e. 10/255) in the standard run.

## CNN forward (SDeCNN-style)
```
sf = 2
pad H,W up to even with ReplicationPad2d((0, padRight, 0, padBottom))
x = PixelUnshuffle(sf)(x)                 # channels *= 4
m = Nsigma.repeat(1,1,H',W')
x = concat([x, m], dim=1)                 # 7*4 + 1 input channels
x = head(conv 3x3 -> 128, ReLU)
x = body(13 x (conv 3x3 -> 128, ReLU))    # nb=15 total conv layers
x = tail(conv 3x3 -> 1*sf*sf, no ReLU)
x = PixelShuffle(sf)(x)
x = x[..., :H, :W]
```
Load weights with map_location to CPU/GPU, set eval(), freeze parameters. If no weight file exists, skip this prior entirely and keep TV.

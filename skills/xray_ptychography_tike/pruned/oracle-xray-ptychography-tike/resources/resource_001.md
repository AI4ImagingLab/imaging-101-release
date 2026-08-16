# Scan shift convention details

Reference formula (verified against the public case):

    scan_shifted = scan - np.amin(scan, axis=0) + 20.0

Verified public-case numbers:
- raw min: [-164.338058, -164.390533]
- raw max: [167.579636, 164.502014]
- shift amount = amin = [-164.338058, -164.390533]
- shifted min: [20.0, 20.0]
- shifted max: [351.917664, 348.892548]
- ceil of max: [352, 349]

Why upper-left corners and not centers: the forward model slices the object as
`obj[r:r+ph, c:c+pw]` with `(r, c) = int(scan[i])`. If you instead treated the
coordinates as probe centers (subtracting ph/2, pw/2), the object frame and the
patch slicing start points both shift, and the output will not be in the
benchmark frame. Keep the corners-as-slicing-starts interpretation and the
20.0 offset exactly as specified.

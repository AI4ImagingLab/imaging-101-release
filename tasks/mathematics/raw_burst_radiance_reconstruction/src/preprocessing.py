"""hdrplus burst alignment functions.
Copyright (c) 2021 Antoine Monod

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

This file implements an algorithm possibly linked to the patent US9077913B2.
This file is made available for the exclusive aim of serving as scientific tool
to verify the soundness and completeness of the algorithm description.
Compilation, execution and redistribution of this file may violate patents rights in certain countries.
The situation being different for every country and changing over time,
it is your responsibility to determine which patent rights restrictions apply to you
before you compile, use, modify, or redistribute this file.
A patent lawyer is qualified to make this determination.
If and only if they don't conflict with any patent terms,
you can benefit from the following license terms attached to this file.

Adapted locally: explicit NumPy distances, corrected 3x3 indexing and safe RMS.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import signal
def load_data(path):
    with np.load(path,allow_pickle=False) as f: return {'frames':f['frames'][0].copy()}
def isTypeInt(x):return np.issubdtype(x.dtype,np.integer)
def umean4_(a,b,c,d):return ((a.astype('uint64')+b+c+d+2)>>2).astype(a.dtype)
def fmean4_(a,b,c,d):return (a+b+c+d)*.25
def getTiles(image,window,steps=1):
    if steps is None:steps=1
    return np.lib.stride_tricks.sliding_window_view(image,(window,window))[::steps,::steps]
def downsample(image, kernel='gaussian', factor=2):
	'''Apply a convolution by a kernel if required, then downsample an image.
	Args:
		image: the input image (WARNING: single channel only!)
		kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
		factor: downsampling factor
	'''
	# Special case
	if factor == 1:
		return image

	# Filter the image before downsampling it
	if kernel is None:
		filteredImage = image
	elif kernel == 'gaussian':
		# gaussian kernel std is proportional to downsampling factor
		filteredImage = gaussian_filter(image, sigma=factor * 0.5, order=0, output=None, mode='reflect')
	elif kernel == 'bayer':
		# Bayer means that a simple 2x2 aggregation is required
		if isTypeInt(image):
			return umean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
		else:
			return fmean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
	else:
		# filter by convoluting with the input kernel
		filteredImage = signal.convolve2d(image, kernel, boundary='symm', mode='valid')

	# Shape of the downsampled image
	h2, w2 = np.floor(np.array(filteredImage.shape) / float(factor)).astype(int)

	# Extract the pixels
	if isTypeInt(image):
		return np.rint(filteredImage[:h2 * factor:factor, :w2 * factor:factor]).astype(image.dtype)
	else:
		return filteredImage[:h2 * factor:factor, :w2 * factor:factor]

def getAlignedTiles(image, tileSize, motionVectors):
	'''Replace tiles within an image based on estimated motion between
	said image and a reference image image.'''

	# For convenience
	# total number of image tiles overlapped by half in each spatial dimension: h*w
	h, w = image.shape[0] // (tileSize // 2) - 1, image.shape[1] // (tileSize // 2) - 1
	# total number of tiles that have estimated motion : hm*wm (<= h*w)
	hm, wm, _ = motionVectors.shape
	# Extract all the possible tiles of the image
	imageTiles = getTiles(image, tileSize)
	# Get the indices of the beginning of each tiles when moved
	indIm = np.round((np.repeat((np.arange(hm) * tileSize // 2).reshape(hm, 1), wm, axis=1) + motionVectors[:, :, 0])).astype(int).clip(0, image.shape[0] - tileSize)
	indJm = np.round((np.repeat((np.arange(wm) * tileSize // 2).reshape(1, wm), hm, axis=0) + motionVectors[:, :, 1])).astype(int).clip(0, image.shape[1] - tileSize)
	indI = np.repeat((np.arange(h) * tileSize // 2).reshape(h, 1), w, axis=1)
	indJ = np.repeat((np.arange(w) * tileSize // 2).reshape(1, w), h, axis=0)
	indI[:indIm.shape[0], :indIm.shape[1]] = indIm  # tiles that have no motion vectors attached will remain identical
	indJ[:indJm.shape[0], :indJm.shape[1]] = indJm

	# Only keep the tiles we're interested in (step = tileSize // 2)
	alignedTiles = imageTiles[indI.reshape(h * w), indJ.reshape(h * w)].reshape((h, w, tileSize, tileSize))

	return alignedTiles

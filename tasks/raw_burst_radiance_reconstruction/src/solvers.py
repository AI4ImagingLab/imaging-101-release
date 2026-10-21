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
import time
import numpy as np
import torch
def getTime(previous,label):
    current=time.time();print(label,':',round((current-previous)*1000,2),'milliseconds');return current
from .preprocessing import downsample,getTiles,getAlignedTiles,isTypeInt
from .physics_model import computeDistance,computeTilesDistanceL1_,subPixelMinimum
def alignHdrplus(referenceImage, alternateImages, params, options):
	'''Implements the coarse-to-fine alignment on 4-level gaussian pyramids
	as defined in Algorithm 1 of Section 3 of the IPOL article.
	Args:
		referenceImage / alternateImages: Bayer / grayscale images
		params: dict containing both algorithm parameters and output choices
		options: dict containing options extracted from the script command (input/output path, mode, verbose)
	'''
	# For convenience
	currentTime, verbose = time.time(), options['verbose'] > 2
	# factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
	factors = params['tuning']['factors']
	tileSizes = params['tuning']['tileSizes']
	distances = params['tuning']['distances']
	searchRadia = params['tuning']['searchRadia']
	subpixels = params['tuning']['subpixels']

	upsamplingFactors = factors[1:] + [1]
	previousTileSizes = tileSizes[1:] + [None]

	if params['mode'] == 'bayer':
		# If dealing with raw images, 2x2 bayer pixels block are averaged
		# (motion can then only be multiples of 2 pixels in original image size)
		imRef = downsample(referenceImage, kernel='bayer')
		if verbose:
			currentTime = getTime(currentTime, ' --- Ref Bayer downsampling')
		tileSize = 2 * tileSizes[0]
	else:
		imRef = referenceImage
		tileSize = tileSizes[0]

	refTiles = getTiles(referenceImage, tileSize, tileSize // 2)  # tiles overlap by half in each spatial dimension
	alignedTiles = np.empty(((len(alternateImages) + 1,) + refTiles.shape), dtype=refTiles.dtype)
	alignedTiles[0] = refTiles  # because Ref. image has no motion wrt itself
	allLevels=[]
	motionVectors = np.empty((len(alternateImages), refTiles.shape[0], refTiles.shape[1], 2), dtype=int)

	# construct 4-level coarse-to fine pyramid of the reference
	referencePyramid = hdrplusPyramid(imRef, factors)
	if verbose:
		currentTime = getTime(currentTime, ' --- Create ref pyramid')

	# Align each alternate image to the reference image
	for i, alternateImage in enumerate(alternateImages):

		if params['mode'] == 'bayer':
			# dowsample bayer to grayscale
			imAlt = downsample(alternateImage, kernel='bayer')
			if verbose:
				currentTime = getTime(currentTime, ' --- Alt Bayer downsampling')
		else:
			imAlt = alternateImage

		# 4-level coarse-to fine pyramid of alternate image
		alternatePyramid = hdrplusPyramid(imAlt, factors)
		if verbose:
			currentTime = getTime(currentTime, ' --- Create alt pyramid')

		# succesively align from coarsest to finest level of the pyramid
		alignments = None
		for lv in range(len(referencePyramid)):
			alignments = alignOnALevel(
				referencePyramid[lv],
				alternatePyramid[lv],
				options,
				upsamplingFactors[-lv - 1],
				tileSizes[-lv - 1],
				previousTileSizes[-lv - 1],
				searchRadia[-lv - 1],
				distances[-lv - 1],
				subpixels[-lv - 1],
				alignments
			)
			allLevels.append(alignments.copy())
		if verbose:
			currentTime = getTime(currentTime, ' --- Align pyramid')

		# use alignment vectors to get the tiles of alternateImage matching with those of the reference image
		if params['mode'] == 'bayer':
			# estimated motion must be upsampled by a factor of 2 to go back to original image size
			alignments = upsampleAlignments([], [], alignments, 2, tileSize, tileSizes[0], None)
		alignedAltTiles = getAlignedTiles(alternateImage, tileSize, alignments)
		if verbose:
			currentTime = getTime(currentTime, ' --- Get aligned tiles')
		motionVectors[i] = alignments
		alignedTiles[i + 1] = alignedAltTiles
		if verbose:
			currentTime = getTime(currentTime, ' -- Aligned frame {}/{} to the reference image'.format(i + 1, len(alternateImages)))

	return motionVectors, alignedTiles, allLevels

def hdrplusPyramid(image, factors=[1, 2, 4, 4], kernel='gaussian'):
	'''Construct 4-level coarse-to-fine gaussian pyramid
	as described in the HDR+ paper and its supplement (Section 3.2 of the IPOL article).
	Args:
		image: input image (expected to be a grayscale image downsampled from a Bayer raw image)
		factors: [int], dowsampling factors (fine-to-coarse)
		kernel: convolution kernel to apply before downsampling (default: gaussian kernel)'''
	# Start with the finest level computed from the input
	pyramidLevels = [downsample(image, kernel, factors[0])]

	# Subsequent pyramid levels are successively created
	# with convolution by a kernel followed by downsampling
	for factor in factors[1:]:
		pyramidLevels.append(downsample(pyramidLevels[-1], kernel, factor))

	# Reverse the pyramid to get it coarse-to-fine
	return pyramidLevels[::-1]

def upsampleAlignments(referencePyramidLevel, alternatePyramidLevel, previousAlignments, upsamplingFactor, tileSize, previousTileSize, method='hdrplus'):
	'''Upsample alignements to adapt them to the next pyramid level (Section 3.2 of the IPOL article).'''
	# As resolution is multiplied, so are alignment vector values
	previousAlignments *= upsamplingFactor
	# Different resolution upsampling factors and tile sizes lead to different vector repetitions
	repeatFactor = upsamplingFactor // (tileSize // previousTileSize)
	# UpsampledAlignments.shape can be less than referencePyramidLevel.shape/tileSize
	# eg when previous alignments could not be computed over the whole image
	upsampledAlignments = previousAlignments.repeat(repeatFactor, 0).repeat(repeatFactor, 1)

	# If the method is not defined, no need to go further in the upsampling
	if method is None :
		return upsampledAlignments

	# For convenience
	h, w = upsampledAlignments.shape[0], upsampledAlignments.shape[1]

	#
	# HDR+ method
	#
	# Take as candidates the alignments
	# for the 3 nearest coarse-scale tiles, the nearest tile plus
	# the next-nearest neighbor tiles in each dimension

	# Pad alignments by mirroring to avoid nearest neighbor tile problems on edges
	paddedPreviousAlignments = np.pad(previousAlignments, pad_width=((1,), (1,), (0,)), mode='edge')

	# Create a mask of closest neighbors specifying the offsets to use to get the right indexes
	# Build the elemental tile to be repeated
	tile = np.empty((repeatFactor, repeatFactor, 2, 2), dtype=int)
	# upper and left coarse-scale tiles
	tile[:(repeatFactor // 2), :(repeatFactor // 2)] = [[-1, 0], [0, -1]]
	# upper and right coarse-scale tiles
	tile[:(repeatFactor // 2), (repeatFactor // 2):] = [[-1, 0], [0, 1]]
	# lower and left coarse-scale tiles
	tile[(repeatFactor // 2):, :(repeatFactor // 2)] = [[1, 0], [0, -1]]
	# lower and right coarse-scale tiles
	tile[(repeatFactor // 2):, (repeatFactor // 2):] = [[1, 0], [0, 1]]
	# Repeat the elemental tile into the full mask
	neighborsMask = np.tile(tile, (upsampledAlignments.shape[0] // repeatFactor, upsampledAlignments.shape[1] // repeatFactor, 1, 1))

	# Compute the indices of the neighbors using the offsets mask
	ti1 = np.repeat(np.clip(2 + np.arange(h) // repeatFactor + neighborsMask[:, 0, 0, 0], 0, paddedPreviousAlignments.shape[0] - 1).reshape(h, 1), w, axis=1).reshape(h * w)
	ti2 = np.repeat(np.clip(2 + np.arange(h) // repeatFactor + neighborsMask[:, 0, 1, 0], 0, paddedPreviousAlignments.shape[0] - 1).reshape(h, 1), w, axis=1).reshape(h * w)
	tj1 = np.repeat(np.clip(2 + np.arange(w) // repeatFactor + neighborsMask[0, :, 0, 1], 0, paddedPreviousAlignments.shape[1] - 1).reshape(1, w), h, axis=0).reshape(h * w)
	tj2 = np.repeat(np.clip(2 + np.arange(w) // repeatFactor + neighborsMask[0, :, 1, 1], 0, paddedPreviousAlignments.shape[1] - 1).reshape(1, w), h, axis=0).reshape(h * w)
	# Extract the previously estimated motion vectors associeted with those neighbors
	ppa1 = paddedPreviousAlignments[ti1, tj1].reshape((h, w, 2))
	ppa2 = paddedPreviousAlignments[ti2, tj2].reshape((h, w, 2))

	# Get all possible tiles in the reference and alternate pyramid level
	refTiles = getTiles(referencePyramidLevel, tileSize, 1)
	altTiles = getTiles(alternatePyramidLevel, tileSize, 1)

	# Compute the distance between the reference tile and the tiles that we're actually interested in
	d0 = computeTilesDistanceL1_(refTiles, altTiles, upsampledAlignments).reshape(h * w)
	d1 = computeTilesDistanceL1_(refTiles, altTiles, ppa1).reshape(h * w)
	d2 = computeTilesDistanceL1_(refTiles, altTiles, ppa2).reshape(h * w)

	# Get the candidate motion vectors
	candidateAlignments = np.empty((h * w, 3, 2))
	candidateAlignments[:, 0, :] = upsampledAlignments.reshape(h * w, 2)
	candidateAlignments[:, 1, :] = ppa1.reshape(h * w, 2)
	candidateAlignments[:, 2, :] = ppa2.reshape(h * w, 2)

	# Select the best candidates by keeping those that minimize the L1 distance with the reference tiles
	selectedAlignments = candidateAlignments[np.arange(h * w), np.argmin([d0, d1, d2], axis=0)].reshape((h, w, 2))
	if h < referencePyramidLevel.shape[0] // (tileSize // 2) - 1 or w < referencePyramidLevel.shape[1] // (tileSize // 2) - 1:
		# tiles were no alignment was computed will have an estimated motion of 0 pixels
		newAlignments = np.zeros((referencePyramidLevel.shape[0] // (tileSize // 2) - 1, referencePyramidLevel.shape[1] // (tileSize // 2) - 1, 2), dtype=selectedAlignments.dtype)
		newAlignments[:h, :w] = selectedAlignments
	else:
		newAlignments = selectedAlignments

	return newAlignments

def alignOnALevel(referencePyramidLevel, alternatePyramidLevel, options, upsamplingFactor=1, tileSize=16, previousTileSize=None, searchRadius=4, distance='L2', subpixel=True, previousAlignments=None):
	'''motion estimation performed at a single Gaussian pyramid level.
	Args:
		referencePyramidLevel / alternatePyramidLevel: images at the current pyramid level on which motion is estimated
		options: dict containing the level of verbosity
		upsamplingFactor: int, factor between the previous (coarser) and current pyramid level
		tileSize: current tile size
		previousTileSize: None / int, tile size used in the previous (coarser) level (if any)
		searchRadius: int, pilots the size of the square search area around the initial guess
		distance: str ('L1'/'L2'), type of norm used to compute distances between ref and alternate tiles
		subpixel: bool, whether to compute sub-pixel alignment from the pixel-level result
		previousAlignments: None / 2d array of 2d arrays, can be used to update the initial guess
	'''
	# For convenience
	verbose, currentTime = options['verbose'] > 3, time.time()

	# Distances shall be computed over float32 values.
	# By casting the input images here, this save the casting of much more data afterwards
	if isTypeInt(referencePyramidLevel):
		referencePyramidLevel = referencePyramidLevel.astype(np.float32)
	if isTypeInt(alternatePyramidLevel):
		alternatePyramidLevel = alternatePyramidLevel.astype(np.float32)

	# Extract the tiles of the reference image overlapped by half in each spatial dimension
	refTiles = getTiles(referencePyramidLevel, tileSize, steps=tileSize // 2)
	if verbose:
		currentTime = getTime(currentTime, ' ---- Divide in tiles')

	# Upsample the previous alignements for initialization
	if previousAlignments is None:
		# no initial offset, search windows centered around reference tiles
		upsampledAlignments = np.zeros((refTiles.shape[0], refTiles.shape[1], 2), dtype=np.float32)
	else :
		# use the upsampled previous alignments as initial guesses
		upsampledAlignments = upsampleAlignments(
			referencePyramidLevel,
			alternatePyramidLevel,
			previousAlignments,
			upsamplingFactor,
			tileSize,
			previousTileSize
		)
	if verbose:
		currentTime = getTime(currentTime, ' ---- Upsample alignments')

	# For convenience
	h, w, sR = upsampledAlignments.shape[0], upsampledAlignments.shape[1], 2 * searchRadius + 1

	# the initial offsets / alignment guesses [u0, v0] correspond to the upsampled previous alignments
	u0 = np.round(upsampledAlignments[:, :, 0]).astype(int)
	v0 = np.round(upsampledAlignments[:, :, 1]).astype(int)
	if verbose:
		currentTime = getTime(currentTime, ' ---- Compute indices')

	# Get all possible square search areas in the alternate image. Pad the original image with infinite values
	# each area has a side of length (tileSize + 2 * searchRadius)
	searchAreas = getTiles(np.pad(alternatePyramidLevel, searchRadius, mode='constant', constant_values=2**16 - 1), window=tileSize + 2 * searchRadius)
	# Only keep those corresponding to the area around the reference tile location + [u0, v0]
	indI = np.clip(((np.repeat((np.arange(h) * tileSize // 2).reshape(h, 1), w, axis=1)).reshape(h, w) + u0).reshape(h * w), 0, searchAreas.shape[0] - 1)
	indJ = np.clip(((np.repeat((np.arange(w) * tileSize // 2).reshape(1, w), h, axis=0)).reshape(h, w) + v0).reshape(h * w), 0, searchAreas.shape[1] - 1)
	searchAreas = searchAreas[indI, indJ].reshape((h, w, tileSize + 2 * searchRadius, tileSize + 2 * searchRadius))
	if verbose:
		currentTime = getTime(currentTime, ' ---- Extract searching areas')

	# Compute the distances between the reference tiles and each tile within the corresponding search areas
	assert(h <= refTiles.shape[0] and w <= refTiles.shape[1])
	distances = computeDistance(refTiles[:h, :w], searchAreas, distance)
	if verbose:
		currentTime = getTime(currentTime, ' ---- Compute distances')

	# Get the indexes (within the search area) of alternate tiles of minimum distance wrt the reference
	minIdx = np.argmin(distances, axis=1).astype(np.uint16)  # distance minimum indices are flattened

	levelOffsets = np.zeros((upsampledAlignments.shape[0] * upsampledAlignments.shape[1], 2), dtype=np.float32)
	levelOffsets[..., 0], levelOffsets[..., 1] = np.unravel_index(minIdx, (sR, sR))  # get 2d indexes from flattened indexes
	if subpixel:
		# Initialization
		subpixOffsets = np.zeros_like(levelOffsets)
		# only work on distance minimums that actually feature a 3*3 = 9 pixel neighborhood
		validMinIdx = (minIdx // sR > 0) & (minIdx // sR < sR - 1) & (minIdx % sR > 0) & (minIdx % sR < sR - 1)
		# Find the optimal offset only for valid positions
		subpixOffsets[validMinIdx] = subPixelMinimum(distances[validMinIdx], minIdx[validMinIdx])
		# Update the offsets
		levelOffsets += subpixOffsets

	levelOffsets = levelOffsets.reshape((h, w, 2))
	# final values: initial guess [u0, v0] + current motion estimation that is actually between - and + searchRadius
	levelOffsets[..., 0] = u0 + levelOffsets[..., 0] - searchRadius
	levelOffsets[..., 1] = v0 + levelOffsets[..., 1] - searchRadius

	if verbose:
		currentTime = getTime(currentTime, ' ---- Deduce positions')

	return levelOffsets

def centeredCosineWindow(x, windowSize=16):
	'''1D version of the modified raised cosine window (Section 4.4 of the IPOL article).
	It is centered and nonzero at x=0 and x=windowSize-1'''
	y = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * (x + 1 / 2.) / windowSize)
	return y

def cosineWindow2Dpatches(patches):
	'''Apply a 2D version of the modified raised cosine window
	to a set of overlapped patches to avoid discontinuities and edge artifacts
	(Section 4.4 of the IPOL article).
	Args:
		patches: 2D array of 2D patches (overlapped by half in each dimension)'''
	assert(len(patches.shape) == 4), 'not a 2D array of image patches'
	windowSize = patches.shape[-1]  # Assumes patches are square
	# Compute the attenuation window on 1 patch dimension
	lineWeights = centeredCosineWindow(np.arange(windowSize), windowSize).reshape(-1, 1).repeat(windowSize, 1)
	columnWeights = lineWeights.T
	# the 2D window is the product of the 1D window in both patches dimensions
	window = np.multiply(lineWeights, columnWeights)
	# Apply the attenuation cosine weighting to all patches
	return np.multiply(patches, window)

def cat2DPatches(patches):
	assert(len(patches.shape) == 4), "not a 2D array of 2D arrays"
	return np.concatenate(np.concatenate(patches, axis=1), axis=1)

def depatchifyOverlap(patches):
	'''recreates a single image out of a 2d arrangement
	of patches overlapped by half in each dimension
	'''
	assert(len(patches.shape) == 4), "not a 2D array of 2D patches"
	patchSize = patches.shape[-1]
	dp = patchSize // 2
	assert(patchSize == patches.shape[-2] and patchSize % 2 == 0), "function only supports square patches of even size"

	# separate the different groups of overlapped patches
	patchSet00 = patches[0::2, 0::2]  # original decomposition
	patchSet01 = patches[0::2, 1::2]  # straddled by patchSize/2 in horizontal axis
	patchSet10 = patches[1::2, 0::2]  # straddled by patchSize/2 in vertical axis
	patchSet11 = patches[1::2, 1::2]  # straddled by patchSize/2 half in both axes

	# recreate sub-images from the different patch groups
	imSet00 = cat2DPatches(patchSet00)
	imSet01 = cat2DPatches(patchSet01)
	imSet10 = cat2DPatches(patchSet10)
	imSet11 = cat2DPatches(patchSet11)

	# reconstruct final image by correctly adding sub-images
	reconstructedImage = np.zeros(((patches.shape[0] + 1) * dp, (patches.shape[1] + 1) * dp), dtype=imSet00.dtype)
	reconstructedImage[0 : imSet00.shape[0]     , 0 : imSet00.shape[1]     ]  = imSet00
	reconstructedImage[0 : imSet01.shape[0]     , dp: imSet01.shape[1] + dp] += imSet01
	reconstructedImage[dp: imSet10.shape[0] + dp, 0 : imSet10.shape[1]     ] += imSet10
	reconstructedImage[dp: imSet11.shape[0] + dp, dp: imSet11.shape[1] + dp] += imSet11
	return reconstructedImage

def patchesRMS(patches):
	'''Computes the Root-Mean-Square of a set of patches/tiles.
	Args:
		patches: nD array (n >= 3) of 2D patches
	'''
	assert len(patches.shape) >= 3, 'not an nD array of patches'
	# flatten each patch
	patches = np.reshape(patches, tuple(patches.shape[:-2] + (patches.shape[-2] * patches.shape[-1],)))
	return np.sqrt(np.mean(np.multiply(patches.astype(np.float64), patches.astype(np.float64)), axis=-1))

def mergeWienerDFTPatches(referenceImagePatchesFFT, alternateImagePatchesFFT, noiseVariance, temporalFactor=8):
	'''Temporally denoise a pair of sets of 2D Frequency-domain patches
	Using a variant of the Wiener filter (Section 4.2 of the IPOL article).
	Args:
		referenceImagePatchesFFT: numpy array of 2D DFT patches of the reference image
		alternateImagePatchesFFT: numpy array of 2D DFT patches of the (aligned) alternate image
		noiseVariance: numpy array of the estimated noise variance for each patch
		temporalFactor: tuning factor that drives the compromise between ghosting and denoising.'''
	# scale the noise variance to match the scale of dSq
	patchSize = referenceImagePatchesFFT.shape[-1]
	noiseScaling = patchSize**2 * 1 / 4**2 * 2  # described in the paper
	# additional scaling according to temporal denoising strength tuning
	noiseScaling *= temporalFactor
	# Call the vectorized function
	mergedImagePatchesFFT = mergeWienerDFTPatches_(referenceImagePatchesFFT, alternateImagePatchesFFT, noiseScaling * noiseVariance)

	return mergedImagePatchesFFT

def temporalDenoisePairPatches(referenceImagePatches, alternateImagePatches, noiseVariance, temporalFactor=8, method='DFTWiener'):
	'''Temporally denoise two sets of 2D patches, and return a single set of denoised patches.'''
	if method == 'keepAlternate':
		# only return alternate image patches (e.g. to later average them)
		mergedImagePatches = alternateImagePatches
	elif method == 'pairedAverage':
		mergedImagePatches = np.multiply(0.5, referenceImagePatches + alternateImagePatches)
	elif method == 'DFTWiener':
		# referenceImagePatches then already contains 2D DFT of patches
		altPatchesFFT = torch.fft.fftn(torch.from_numpy(alternateImagePatches.astype(np.float32)), dim=(2, 3)).numpy()
		mergedImagePatches = mergeWienerDFTPatches(referenceImagePatches, altPatchesFFT, noiseVariance, temporalFactor)
	return mergedImagePatches

def spatialDenoisePatches(patchesFFT, noiseVariance, spatialFactor):
	'''Spatially denoise a set of 2D Frequency-domain patches
	Using a variant of the Wiener filter (Section 4.3 of the IPOL article).
	Args:
		patchesFFT: numpy array of 2D DFT (temporally denoised) patches of the reference image.
		noiseVariance: numpy array of the (updated) estimated noise variance for each patch
		spatialFactor: tuning factor that drives the compromise between less residual noise and loss of high frequency content.'''
	assert(len(patchesFFT.shape) == 4), "not a 2D array of 2D patches"
	# Assumes square patches
	patchSize = patchesFFT.shape[-1]
	# create a patch of distance of spatial frequency module with respect to origin of spatial frequency axes
	# this patch will be used to filter higher frequency content more aggressively
	# WARNING: to avoid computing the FFT shift to the patches then FFT ishift,
	# This shift is applied directly to the distance
	rowDistances = (np.arange(patchSize) - patchSize / 2).reshape(-1, 1).repeat(patchSize, 1)
	columnDistances = rowDistances.T
	distancePatch = np.sqrt(rowDistances**2 + columnDistances**2)
	distPatchShift = np.fft.ifftshift(distancePatch, axes=(-2, -1))
	# Scale the noise variance to match the scale of np.abs(patchesFFT)**2
	noiseScaling = patchSize**2 * 1 / 4**2
	# Additional scaling according to spatial denoising strength tuning
	noiseScaling *= spatialFactor
	# Apply the Wiener filtering
	return spatialDenoisePatches_(patchesFFT, distPatchShift * noiseScaling * noiseVariance)

def mergeChannelHdrplus(referenceChannel, alignedChannelTiles, lambdaS, lambdaR, params, options):
	'''perform per-channel, tile-based, pairwise temporal denoising
	as defined in Algorithm 2 of Section 4.2 of the IPOL article.'''
	# Initialization
	currentTime, verbose = time.time(), options['verbose'] > 3
	referenceChannelTiles = alignedChannelTiles[0]
	noiseVariance = []
	if verbose:
		currentTime = getTime(currentTime, ' ---- Initialization')
	if params['method'] == 'DFTWiener':
		# compute noise model of reference image
		# signal level is considered constant for each patch
		signalLevel = patchesRMS(referenceChannelTiles)
		if verbose:
			currentTime = getTime(currentTime, ' ---- Get signal level')
		# noiseVariance = lambdaS * signalLevel + lambdaR
		# compute the per-patch variance value (same value for all pixels of the patch)
		noiseVariance = (np.multiply(lambdaS, signalLevel[..., np.newaxis, np.newaxis]) + lambdaR).repeat(params['patchSize'], axis=-2).repeat(params['patchSize'], axis=-1)
		if verbose:
			currentTime = getTime(currentTime, ' ---- Get noise variance')
		# compute 2D DFT of reference image tiles
		referenceChannelTiles = torch.fft.fftn(torch.from_numpy(referenceChannelTiles.astype(np.float32)), dim=(2, 3)).numpy()
		if verbose:
			currentTime = getTime(currentTime, ' ---- Compute 2D DFT')

	# pairwise merging of reference image channel with itself = reference image channel
	mergedPairsTilesSum = referenceChannelTiles.copy()
	if verbose:
			currentTime = getTime(currentTime, ' ---- Copy reference')

	if options['temporalFactor'] == 0:
		# no temporal denoising
		mergedChannelTiles = mergedPairsTilesSum
	else:
		# pairwise merging of all other images with reference image
		for i in range(1, len(alignedChannelTiles)):
			mergedPairsTilesSum += temporalDenoisePairPatches(referenceChannelTiles, alignedChannelTiles[i], noiseVariance, options['temporalFactor'], params['method'])
		if verbose:
				currentTime = getTime(currentTime, ' ---- Temporal denoise pair patches')
		# final merge = mean of all pairwise merges
		mergedChannelTiles = mergedPairsTilesSum / len(alignedChannelTiles)
	if verbose:
			currentTime = getTime(currentTime, ' ---- Average tiles')

	if params['method'] == 'DFTWiener':
		if options['spatialFactor'] > 0:
			# update noise variance estimation after temporal denoising
			noiseVariance /= len(alignedChannelTiles)
			# perform fast spatial denoising using a Wiener filter
			mergedChannelTiles = spatialDenoisePatches(mergedChannelTiles, noiseVariance, options['spatialFactor'])
			if verbose:
				currentTime = getTime(currentTime, ' ---- Spatial denoise')

		mergedChannelTiles = torch.fft.ifftn(torch.from_numpy(mergedChannelTiles), dim=(2, 3)).real.numpy()
		if verbose:
			currentTime = getTime(currentTime, ' ---- DFT inverse')

	# apply a cosine window function to the overlapping patches to avoid edge artifacts
	mergedChannelTiles = cosineWindow2Dpatches(mergedChannelTiles)
	if verbose:
			currentTime = getTime(currentTime, ' ---- Cosine window')

	# reconstruct an unsigned integer image from the patches
	mergedChannel = depatchifyOverlap(mergedChannelTiles).astype(referenceChannel.dtype)
	if verbose:
			currentTime = getTime(currentTime, ' ---- Depatchify')

	return mergedChannel

def mergeHdrplus(referenceImage, alignedTiles, padding, tags, blackLevel, whiteLevel, params, options):
	'''Implements the Fourier Tile-based Merging as described in Section 4 of the IPOL artice.
	Args:
		referenceImage: 2D array of the reference (Bayer) Image
		alignedTiles: ndarray of tiles aligned after motion estimation
		padding: tuple of padding applied to all images (will need to be removed)
		tags: dict of image metadata
		blackLevel: [int], per-channel black level of the reference image
		whiteLevel: int, reference image white level
		params: dict containing both algorithm parameters and output choices
		options: dict containing options extracted from the script command (input/output path, mode, verbose)
	'''
	# Initialization
	currentTime, verbose = time.time(), options['verbose'] > 2
	# perform burst merging as pairwise merging of reference image and each alternate image
	tileSize = alignedTiles.shape[3]
	h, w = (alignedTiles.shape[1] + 1) * tileSize // 2, (alignedTiles.shape[2] + 1) * tileSize // 2
	mergedImage = np.empty((h, w), dtype=referenceImage.dtype)
	# get noise curve parameters
	lambdaS, lambdaR = getNoiseParams(tags, blackLevel, whiteLevel, params, options)
	if verbose:
		currentTime = getTime(currentTime, ' --- Initialization')

	# work separately on each channel of the Bayer image
	for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
		# for each channel, compute burst merging as described in the HDR+ paper
		mergedImage[di::2, dj::2] = mergeChannelHdrplus(referenceImage[di::2, dj::2], alignedTiles[..., di::2, dj::2], lambdaS, lambdaR, params, options)
		if options['verbose'] > 3:
			currentTime = getTime(currentTime, ' --- Channel %d/4 merged' % c)

	return mergedImage[padding[0]:h - padding[1], padding[2]:w - padding[3]]


def getNoiseParams(tags,blackLevel,whiteLevel,params,options):return params['noiseCurve']
def mergeWienerDFTPatches_(ref,alt,variance):
    # Float64 noise dispatches the original ufunc to its complex128 signature.
    ref=ref.astype('complex128');alt=alt.astype('complex128')
    diff=ref-alt;dist=diff.real**2+diff.imag**2
    return alt+dist/(dist+variance)*diff
def spatialDenoisePatches_(fft,coefficient):
    power=fft.real.astype('float64')**2+fft.imag.astype('float64')**2
    return fft*power/(power+coefficient)
FACTORS=[1,2,4,4]
ALIGNMENT={'mode':'bayer','tuning':{'factors':FACTORS,'tileSizes':[16,16,16,8],'searchRadia':[1,4,4,4],'distances':['L1','L2','L2','L2'],'subpixels':[False,True,True,True]}}
NOISE=(3.55148388,516.6520187906699)
def reconstruct(data,temporal=75.,spatial=.1,frame_count=10,align=True):
    frames=data['frames'][:frame_count];padding=(16,16,16,16);padded=[np.pad(x,16,mode='symmetric') for x in frames]
    options={'verbose':0,'temporalFactor':temporal,'spatialFactor':spatial}
    if align:motion,tiles,levels=alignHdrplus(padded[0],padded[1:],ALIGNMENT,options)
    else:
        tiles=np.asarray([getTiles(x,32,16) for x in padded]);motion=np.zeros((frame_count-1,)+tiles.shape[1:3]+(2,),dtype=int);levels=[]
    params={'patchSize':16,'method':'DFTWiener','noiseCurve':NOISE}
    result=mergeHdrplus(frames[0],tiles,padding,{},[64]*4,1023,params,options)
    return {'reconstruction':result[None]},{'motion':motion,'levels':levels,'aligned_tiles':tiles,'aligned_sample':tiles[:,:3,:3].copy(),'bayer_average':downsample(frames[0],kernel='bayer'),'reference_pyramid':hdrplusPyramid(downsample(padded[0],kernel='bayer'))}

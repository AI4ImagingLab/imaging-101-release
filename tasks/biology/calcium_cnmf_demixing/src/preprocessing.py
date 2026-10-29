# Derived numerical mathematics: attribution and complete license texts in test/validation.md.


from __future__ import annotations

def load_movie(path='data/raw_data.npz'):
    """Validate and load native float32 observations, removing the batch axis."""
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'movie'}:raise ValueError('Expected only movie')
        movie=archive['movie']
    if movie.shape!=(1,3000,170,170) or movie.dtype!=np.float32 or not np.isfinite(movie).all():
        raise ValueError('Invalid native movie')
    return movie[0]
import logging, itertools, collections
import numpy as np
from numpy.fft import ifftshift
import scipy, scipy.interpolate
import cv2
cv2.setNumThreads(0)

def apply_shift_iteration(img, shift, border_nan=False, border_type=cv2.BORDER_REFLECT):
    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.array([[1, 0, sh_y_n], [0, 1, sh_x_n]], dtype=np.float32)
    min_, max_ = (np.nanmin(img), np.nanmax(img))
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i), flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = (0, 0, 0, 0)
        max_h, max_w = np.ceil(np.maximum((max_h, max_w), shift)).astype(int)
        min_h, min_w = np.floor(np.minimum((min_h, min_w), shift)).astype(int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h - 1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w - 1, np.newaxis]
        else:
            logging.warning(f'Unknown value of border_nan ({border_nan}); treating as False')
    return img

def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    if not hasattr(upsampled_region_size, '__iter__'):
        upsampled_region_size = [upsampled_region_size] * data.ndim
    elif len(upsampled_region_size) != data.ndim:
        raise ValueError("shape of upsampled region sizes must be equal to input data's number of dimensions.")
    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    elif len(axis_offsets) != data.ndim:
        raise ValueError("number of axis offsets must be equal to input data's number of dimensions.")
    col_kernel = np.exp(-1j * 2 * np.pi / (data.shape[1] * upsample_factor) * (ifftshift(np.arange(data.shape[1]))[:, None] - np.floor(data.shape[1] // 2)).dot(np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1]))
    row_kernel = np.exp(-1j * 2 * np.pi / (data.shape[0] * upsample_factor) * (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(ifftshift(np.arange(data.shape[0]))[None, :] - np.floor(data.shape[0] // 2)))
    if data.ndim > 2:
        pln_kernel = np.exp(-1j * 2 * np.pi / (data.shape[2] * upsample_factor) * (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(ifftshift(np.arange(data.shape[2]))[None, :] - np.floor(data.shape[2] // 2)))
    output = np.tensordot(row_kernel, data, axes=[1, 0])
    output = np.tensordot(output, col_kernel, axes=[1, 0])
    if data.ndim > 2:
        output = np.tensordot(output, pln_kernel, axes=[1, 1])
    return output

def _compute_phasediff(cross_correlation_max):
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)

def register_translation(src_image, target_image, upsample_factor=1, space='real', shifts_lb=None, shifts_ub=None, max_shifts=(10, 10), use_cuda=False):
    if src_image.shape != target_image.shape:
        raise ValueError('Error: images must really be same size for register_translation')
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError('Error: register_translation only supports subpixel registration for 2D images')
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    elif space.lower() == 'real':
        src_freq_1 = cv2.dft(src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
        src_freq = np.asarray(src_freq, dtype=np.complex128)
        target_freq_1 = cv2.dft(target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
        target_freq = np.asarray(target_freq, dtype=np.complex128)
    else:
        raise ValueError('Error: register_translation only knows the "real" and "fourier" values for the ``space`` argument.')
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    image_product_cv = np.dstack([np.real(image_product), np.imag(image_product)])
    cross_correlation = cv2.dft(image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
    cross_correlation = cross_correlation[:, :, 0] + 1j * cross_correlation[:, :, 1]
    new_cross_corr = np.abs(cross_correlation)
    if shifts_lb is not None or shifts_ub is not None:
        if shifts_lb[0] < 0 and shifts_ub[0] >= 0:
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0
        if shifts_lb[1] < 0 and shifts_ub[1] >= 0:
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:
        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0
        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0
    maxima = np.unravel_index(np.argmax(new_cross_corr), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size // 2) for axis_size in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
    else:
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = src_freq.size * upsample_factor ** 2
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(), upsampled_region_size, upsample_factor, sample_region_offset).conj()
        cross_correlation /= normalization
        maxima = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape), dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(), 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(), 1, upsample_factor)[0, 0]
        target_amp /= normalization
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0
    return (shifts, src_freq, _compute_phasediff(CCmax))

def get_patch_edges(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...]) -> tuple[list[int], ...]:
    windowSizes = np.add(overlaps, strides)
    return tuple((list(range(0, dim - windowSize, stride)) + [dim - windowSize] for dim, windowSize, stride in zip(dims, windowSizes, strides)))

def get_patch_centers(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...], shifts_opencv=False, upsample_factor_grid=1) -> tuple[list[float], ...]:
    if not shifts_opencv and upsample_factor_grid != 1:
        strides = tuple(np.round(np.divide(strides, upsample_factor_grid)).astype(int))
    patch_edges = get_patch_edges(dims, overlaps, strides)
    windowSizes = np.add(overlaps, strides)
    return tuple(([edge + (sz - 1) / 2 for edge in edges] for edges, sz in zip(patch_edges, windowSizes)))

def sliding_window_dims(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...]):
    windowSizes = np.add(overlaps, strides)
    edge_ranges = get_patch_edges(dims, overlaps, strides)
    for patch in itertools.product(*[enumerate(r) for r in edge_ranges]):
        inds, corners = zip(*patch)
        yield (tuple(inds), tuple(corners), windowSizes)

def sliding_window(image: np.ndarray, overlaps: tuple[int, int], strides: tuple[int, int]):
    if image.ndim != 2:
        raise ValueError('Input to sliding_window must be 2D')
    for inds, corner, size in sliding_window_dims(image.shape, overlaps, strides):
        patch = image[corner[0]:corner[0] + size[0], corner[1]:corner[1] + size[1]]
        yield (inds + corner + (patch,))

def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4, upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None, use_cuda=False, border_nan=True, shifts_interpolate=False):
    logger = logging.getLogger('caiman')
    logger.debug('Starting tile and correct')
    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()
    img = img + add_to_movie
    template = template + add_to_movie
    rigid_shts, sfr_freq, diffphase = register_translation(img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts, use_cuda=use_cuda)
    if max_deviation_rigid == 0:
        new_img = apply_shift_iteration(img, (-rigid_shts[0], -rigid_shts[1]), border_nan=border_nan)
        return (new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1]), None, None)
    else:
        logger.debug('Extracting patches')
        xy_grid: list[tuple[int, int]] = []
        templates: list[np.ndarray] = []
        for xind, yind, _, _, patch in sliding_window(template, overlaps=overlaps, strides=strides):
            xy_grid.append((xind, yind))
            templates.append(patch)
        imgs = [it[-1] for it in sliding_window(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xy_grid[-1], 1))
        num_tiles = len(xy_grid)
        if max_deviation_rigid is not None:
            lb_shifts = np.ceil(np.subtract(rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(np.add(rigid_shts, max_deviation_rigid)).astype(int)
        else:
            lb_shifts = None
            ub_shifts = None
        logger.debug('Extracting shifts for each patch')
        shfts_et_all = [register_translation(a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in zip(imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)
        dims = img.shape
        patch_centers = get_patch_centers(dims, strides=strides, overlaps=overlaps)
        m_reg = apply_pw_shifts_remap_2d(img, shifts_y=shift_img_x, shifts_x=shift_img_y, patch_centers=patch_centers, border_nan=border_nan, shifts_interpolate=shifts_interpolate)
        total_shifts = [(-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
        return (m_reg - add_to_movie, total_shifts, None, None)

def apply_pw_shifts_remap_2d(img: np.ndarray, shifts_y: np.ndarray, shifts_x: np.ndarray, patch_centers: tuple[list[float], ...], border_nan: Union[bool, Literal['copy', 'min']], shifts_interpolate=False) -> np.ndarray:
    patch_grid = tuple((len(centers) for centers in patch_centers))
    shift_img_y = np.reshape(shifts_y, patch_grid)
    shift_img_x = np.reshape(shifts_x, patch_grid)
    dims = img.shape
    x_coords, y_coords = [np.arange(0.0, dims[dim]).astype(np.float32) for dim in (1, 0)]
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    shifts_y = cv2.resize(shift_img_y.astype(np.float32), dims[::-1])
    shifts_x = cv2.resize(shift_img_x.astype(np.float32), dims[::-1])
    if border_nan is False:
        mode = cv2.BORDER_CONSTANT
        value = 0.0
    elif border_nan is True:
        mode = cv2.BORDER_CONSTANT
        value = np.nan
    elif border_nan == 'min':
        mode = cv2.BORDER_CONSTANT
        value = np.min(img)
    elif border_nan == 'copy':
        mode = cv2.BORDER_REPLICATE
        value = 0.0
    else:
        raise ValueError(f'Unknown value of border_nan ({border_nan})')
    return cv2.remap(img, shifts_x + x_grid, shifts_y + y_grid, cv2.INTER_CUBIC, borderMode=mode, borderValue=value)
import warnings

class _TemplateMovie(np.ndarray):

    def __new__(cls, array):
        return np.asarray(array).view(cls)

    def __array_finalize__(self, obj):
        pass

    def motion_correct(self, max_shift_w=5, max_shift_h=5, num_frames_template=None, template=None, method: str='opencv', remove_blanks: bool=False, interpolation: str='cubic') -> tuple[Any, tuple, Any, Any]:
        if template is None:
            if num_frames_template is None:
                num_frames_template = 100000000.0 / (self.shape[1] * self.shape[2])
            frames_to_skip = int(np.maximum(1, self.shape[0] / num_frames_template))
            submov = self[::frames_to_skip, :].copy()
            templ = submov.bin_median()
            shifts, xcorrs = submov.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=templ, method=method)
            submov.apply_shifts(shifts, interpolation=interpolation, method=method)
            template = submov.bin_median()
            del submov
            m = self.copy()
            shifts, xcorrs = m.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method=method)
            m = m.apply_shifts(shifts, interpolation=interpolation, method=method)
            template = m.bin_median()
            del m
        else:
            template = template - np.percentile(template, 8)
        shifts, xcorrs = self.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method=method)
        self = self.apply_shifts(shifts, interpolation=interpolation, method=method)
        if remove_blanks:
            raise Exception('motion_correct(): The remove_blanks parameter was never functional and should not be used')
        return (self, shifts, xcorrs, template)

    def bin_median(self, window: int=10) -> np.ndarray:
        T, d1, d2 = self.shape
        num_windows = int(T // window)
        num_frames = num_windows * window
        return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    def extract_shifts(self, max_shift_w: int=5, max_shift_h: int=5, template=None, method: str='opencv') -> tuple[list, list]:
        logger = logging.getLogger('caiman')
        min_val = np.percentile(self, 1)
        if min_val < -0.1:
            logger.debug(f'min_val in extract_shifts: {min_val}')
            logger.warning('Movie average is negative. Removing 1st percentile.')
            self = self - min_val
        else:
            min_val = 0
        if not isinstance(self[0, 0, 0], np.float32):
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)
        _, h_i, w_i = self.shape
        ms_w = max_shift_w
        ms_h = max_shift_h
        if template is None:
            template = np.median(self, axis=0)
        elif np.percentile(template, 8) < -0.1:
            logger.warning('Movie average is negative. Removing 1st percentile.')
            template = template - np.percentile(template, 1)
        template = template[ms_h:h_i - ms_h, ms_w:w_i - ms_w].astype(np.float32)
        shifts = []
        xcorrs = []
        for i, frame in enumerate(self):
            if i % 100 == 99:
                logger.debug(f'Frame {i + 1}')
            res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
            top_left = cv2.minMaxLoc(res)[3]
            avg_corr = np.mean(res)
            sh_y, sh_x = top_left
            if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
                log_xm1_y = np.log(res[sh_x - 1, sh_y])
                log_xp1_y = np.log(res[sh_x + 1, sh_y])
                log_x_ym1 = np.log(res[sh_x, sh_y - 1])
                log_x_yp1 = np.log(res[sh_x, sh_y + 1])
                four_log_xy = 4 * np.log(res[sh_x, sh_y])
                sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
            else:
                sh_x_n = -(sh_x - ms_h)
                sh_y_n = -(sh_y - ms_w)
            shifts.append([sh_x_n, sh_y_n])
            xcorrs.append([avg_corr])
        self = self + min_val
        return (shifts, xcorrs)

    def apply_shifts(self, shifts, interpolation: str='linear', method: str='opencv', remove_blanks: bool=False):
        logger = logging.getLogger('caiman')
        if not isinstance(self[0, 0, 0], np.float32):
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)
        if interpolation == 'cubic':
            interpolation = cv2.INTER_CUBIC
            logger.debug('cubic interpolation')
        elif interpolation == 'nearest':
            interpolation = cv2.INTER_NEAREST
            logger.debug('nearest interpolation')
        elif interpolation == 'linear':
            interpolation = cv2.INTER_LINEAR
            logger.debug('linear interpolation')
        elif interpolation == 'area':
            interpolation = cv2.INTER_AREA
            logger.debug('area interpolation')
        elif interpolation == 'lanczos4':
            interpolation = cv2.INTER_LANCZOS4
            logger.debug('lanczos/biquartic interpolation')
        else:
            raise Exception('Interpolation method not available')
        _, h, w = self.shape
        for i, frame in enumerate(self):
            if i % 100 == 99:
                logger.debug(f'Frame {i + 1}')
            sh_x_n, sh_y_n = shifts[i]
            M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
            min_, max_ = (np.min(frame), np.max(frame))
            self[i] = np.clip(cv2.warpAffine(frame, M, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT), min_, max_)
        if remove_blanks:
            raise Exception('apply_shifts(): The remove_blanks parameter was never functional and should not be used')
        return self

def motion_correct_movie(movie, progress=None):
    """Full two-pass NoRMCorre, preserving the official 37 temporal chunks."""
    T = movie.shape[0]
    bias = np.float32(-np.min(movie[:400]))
    sampled = _TemplateMovie(movie[::T // 50 + 1].copy())
    template = sampled.motion_correct(6, 6, interpolation='cubic')[0].bin_median()
    rigid_fields = []
    rigid_templates = []
    for ids in np.array_split(np.arange(T), 37):
        block = np.empty((len(ids), *movie.shape[1:]), dtype=np.float32)
        for j, i in enumerate(ids):
            block[j], shift, _, _ = tile_and_correct(movie[i], template, None, None, (6, 6), max_deviation_rigid=0, add_to_movie=bias, shifts_opencv=True, border_nan='copy')
            rigid_fields.append(shift)
        temp = np.nanmean(block, axis=0)
        temp[np.isnan(temp)] = np.nanmin(temp)
        rigid_templates.append(temp)
    rigid_template = np.nanmedian(np.dstack(rigid_templates), axis=-1)
    corrected = np.empty(movie.shape, dtype=np.float32)
    elastic_fields = []
    elastic_templates = []
    for ids in np.array_split(np.arange(T), 37):
        block = np.empty((len(ids), *movie.shape[1:]), dtype=np.float32)
        for j, i in enumerate(ids):
            block[j], shift, _, _ = tile_and_correct(movie[i], rigid_template, (50, 50), (24, 24), (6, 6), max_deviation_rigid=3, add_to_movie=bias, shifts_opencv=True, border_nan='copy')
            elastic_fields.append(shift)
        corrected[ids] = block + bias
        temp = np.nanmean(block, axis=0)
        temp[np.isnan(temp)] = np.nanmin(temp)
        elastic_templates.append(temp)
        if progress is not None:
            progress(int(ids[-1]) + 1, T)
    elastic_template = np.nanmedian(np.dstack(elastic_templates), axis=-1)
    return {'corrected': corrected, 'template_rigid': np.asarray(rigid_template), 'template_elastic': elastic_template, 'shifts_rigid': np.asarray(rigid_fields), 'shifts': np.asarray(elastic_fields), 'bias': bias}

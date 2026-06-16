"""
Inverse solver for s2ISM.

Implements the multi-plane maximum-likelihood expectation-maximisation (ML-EM)
reconstruction that jointly deconvolves all SPAD detector channels to recover
a 3-D fluorescent object from ISM measurements.
"""

import string
import gc
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as torchpad
from torch import real, einsum
from torch.fft import rfftn, irfftn, ifftshift
from tqdm import tqdm


def partial_convolution_rfft(kernel: torch.Tensor, volume: torch.Tensor,
                             dim1: str = 'ijk', dim2: str = 'jkl',
                             axis: str = 'jk', fourier: tuple = (False, False),
                             padding: list = None):
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]

    if fourier[0] == False:
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel

    if fourier[1] == False:
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume

    conv = einsum(f'{dim1},{dim2}->{dim3}', kernel_fft, volume_fft)

    conv = irfftn(conv, dim=axis_list[2], s=padding)
    conv = ifftshift(conv, dim=axis_list[2])
    conv = real(conv)

    return conv


def amd_update_fft(img, obj, psf_fft, psf_m_fft, eps: float):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alphabet = string.ascii_lowercase
    n_dim = psf_fft.ndim

    str_psf = alphabet[:n_dim]
    str_first = alphabet[:n_dim - 1]
    str_second = alphabet[1:n_dim]
    axis_str = alphabet[1:n_dim - 1]

    img_estimate = partial_convolution_rfft(
        psf_fft, obj, dim1=str_psf, dim2=str_first, axis=axis_str, fourier=(1, 0)).sum(0)
    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)
    del img_estimate
    update = partial_convolution_rfft(
        psf_m_fft, fraction, dim1=str_psf, dim2=str_second, axis=axis_str, fourier=(1, 0)).sum(-1)
    del fraction
    obj_new = obj * update

    return obj_new


def amd_stop(o_old, o_new, pre_flag: bool, flag: bool, stop, max_iter: int,
             threshold: float, tot: float, nz: int, k: int):
    int_f_old = (o_old[nz // 2]).sum()
    int_f_new = (o_new[nz // 2]).sum()
    d_int_f = (int_f_new - int_f_old) / tot
    int_bkg_old = o_old.sum() - int_f_old
    int_bkg_new = o_new.sum() - int_f_new
    d_int_bkg = (int_bkg_new - int_bkg_old) / tot

    if isinstance(stop, str) and stop == 'auto':
        if torch.abs(d_int_f) < threshold:
            if not pre_flag:
                flag = False
            else:
                pre_flag = False
        elif k == max_iter:
            flag = False
            print('Reached maximum number of iterations.')
    elif isinstance(stop, str) and stop == 'fixed':
        if k == max_iter:
            flag = False

    return pre_flag, flag, torch.Tensor([int_f_new, int_bkg_new]), \
        torch.Tensor([d_int_f, d_int_bkg])


def max_likelihood_reconstruction(dset, psf, stop='fixed', max_iter: int = 100,
                                  threshold: float = 1e-3, rep_to_save: str = 'last',
                                  initialization: str = 'flat', process: str = 'gpu'):
    cpu_device = torch.device("cpu")
    data = torch.from_numpy(dset * 1.0).type(torch.float32).to(cpu_device)
    h = torch.from_numpy(psf * 1.0).type(torch.float32).to(cpu_device)

    oddeven_check_x = data.shape[0] % 2
    oddeven_check_y = data.shape[1] % 2
    check_x = False
    check_y = False

    data_check = data
    if oddeven_check_x == 0:
        check_x = True
        data_check = data_check[1:]
    if oddeven_check_y == 0:
        check_y = True
        data_check = data_check[:, 1:]

    Nz = h.shape[0]
    shape_data = data_check.shape
    Nx = shape_data[0]
    Ny = shape_data[1]
    shape_init = (Nz,) + shape_data[:-1]

    crop_pad_x = int((shape_data[0] - h.shape[1]) / 2)
    crop_pad_y = int((shape_data[1] - h.shape[2]) / 2)

    if crop_pad_x > 0 or crop_pad_y > 0:
        pad_array = np.zeros(2 * h.ndim, dtype='int')
        pad_array[2:4] = crop_pad_x
        pad_array[4:6] = crop_pad_y
        pad_array = tuple(np.flip(pad_array))

        h = torchpad.pad(h, pad_array, 'constant')
    elif crop_pad_x < 0 or crop_pad_y < 0:
        raise Exception('The PSF is bigger than the image. Warning.')

    flip_ax = list(np.arange(1, data_check.ndim))
    norm_ax = tuple(np.arange(1, h.ndim))

    h = h.type(torch.float64).to(cpu_device)
    h = h / (h.sum(keepdim=True, axis=norm_ax))
    ht = torch.flip(h, flip_ax)

    O = torch.ones(shape_init).to(cpu_device)
    b = torch.finfo(torch.float).eps

    if initialization == 'sum':
        S = data_check.sum(-1) / Nz
        for z in range(Nz):
            O[z, ...] = S
    elif initialization == 'flat':
        O *= data_check.sum() / Nz / Nx / Ny
    else:
        raise Exception('Initialization mode unknown.')

    k = 0

    counts = torch.zeros([2, max_iter + 1]).to(cpu_device)
    diff = torch.zeros([2, max_iter + 1]).to(cpu_device)
    tot = data.sum()

    if isinstance(rep_to_save, str) and rep_to_save == 'all':
        size = [max_iter + 1] + list(O.shape)
        O_all = torch.empty(size).to(cpu_device)
    elif isinstance(rep_to_save, Iterable):
        l = len(rep_to_save)
        size_b = [l] + list(O.shape)
        O_all = torch.empty(size_b).to(cpu_device)

    pre_flag = True
    flag = True

    if stop != 'auto':
        total = max_iter
    else:
        total = None
    cont = 0

    padding = [h.size(d) for d in flip_ax]

    h_tmp = h / (h.sum(keepdim=True, axis=norm_ax))
    ht_tmp = torch.flip(h_tmp, flip_ax)

    h_fft = rfftn(h, dim=flip_ax, s=padding)
    del h
    ht_fft = rfftn(ht, dim=flip_ax, s=padding)
    del ht

    if process == 'gpu' and torch.cuda.is_available():
        try:
            gpu_device = torch.device("cuda:0")
            h_fft = h_fft.to(gpu_device)
            ht_fft = ht_fft.to(gpu_device)
            data_check = data_check.to(gpu_device)
            O = O.to(gpu_device)
            _ = amd_update_fft(data_check, O, h_fft, ht_fft, b)
            print('The algorithm will run on the GPU.')
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                print("Warning: The algorithms goes in Out Of Memory with CUDA. "
                      "The algorithm will run on the CPU.")
                h_fft = h_fft.to(cpu_device)
                ht_fft = ht_fft.to(cpu_device)
                data_check = data_check.to(cpu_device)
                O = O.to(cpu_device)
            else:
                raise e

    pbar = tqdm(total=total, desc='Progress', position=0)

    while flag:
        O_new = amd_update_fft(data_check, O, h_fft, ht_fft, b)
        pre_flag, flag, counts[:, k], diff[:, k] = amd_stop(
            O, O_new, pre_flag, flag, stop, max_iter, threshold, tot, Nz, k)

        if isinstance(rep_to_save, Iterable) and not isinstance(rep_to_save, str):
            if k in rep_to_save:
                O_all[cont, ...] = O.clone()
                cont += 1
        elif isinstance(rep_to_save, str) and rep_to_save == 'all':
            O_all[k, ...] = O.clone()

        O = O_new.clone()

        k += 1
        pbar.update(1)
    pbar.close()

    if check_x:
        if rep_to_save == 'last':
            pad_arr = np.zeros(2 * len(O.shape), dtype='int')
            pad_arr[-4] = 1
            O = torchpad.pad(O, tuple(pad_arr), 'constant')
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype='int')
            pad_arr[-6] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), 'constant')
    if check_y:
        if rep_to_save == 'last':
            pad_arr = np.zeros(2 * len(O.shape), dtype='int')
            pad_arr[-6] = 1
            O = torchpad.pad(O, tuple(pad_arr), 'constant')
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype='int')
            pad_arr[-8] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), 'constant')

    if isinstance(rep_to_save, str) and rep_to_save == 'last':
        obj = O.detach().cpu().numpy()
    else:
        obj = O_all.detach().cpu().numpy()

    counts = counts[:, :k].detach().cpu().numpy()
    diff = diff[:, :k].detach().cpu().numpy()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return obj, counts, diff, k

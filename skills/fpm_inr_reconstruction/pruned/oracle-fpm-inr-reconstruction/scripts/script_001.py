#!/usr/bin/env python
"""Normal-Variance all-in-focus composite from an amplitude stack.

Standalone deterministic helper. Reads an (H,W,Z) amplitude stack from an npz
(key 'amplitude_stack') and writes an all-in-focus (H,W) array. Optional; does
not alter the graded submission.

Usage:
  public/envs/runtime/.venv-posix/bin/python \
      ${CLAUDE_SKILL_DIR}/scripts/script_001.py outputs/case_000.npz workspace/aif.npy
"""
import sys
import numpy as np


def create_balance_map(image_size, patch_size=64, patch_pace=16):
    n = int((image_size - patch_size) / patch_pace + 1)
    bal = np.zeros((image_size, image_size))
    for i in range(n):
        for j in range(n):
            a = i * patch_pace; b = j * patch_pace
            bal[a:a+patch_size, b:b+patch_size] += 1.0
    return n, 1.0 / bal


def all_in_focus_normal_variance(z_stack, patch_size=64, patch_pace=16):
    imgs = z_stack.copy()
    m, n, framenum = imgs.shape
    o_sum = np.mean(imgs, axis=2)
    intensity, edges = np.histogram(np.abs(o_sum), bins="auto")
    background = edges[int(np.argmax(intensity))]
    o_fusion = np.zeros((m, n))
    n_patches, balance_map = create_balance_map(m, patch_size, patch_pace)
    for i in range(n_patches):
        for j in range(n_patches):
            a = i * patch_pace; b = j * patch_pace
            NV = []
            for k in range(framenum):
                crop = imgs[a:a+patch_size, b:b+patch_size, k]
                if background > 0.1:
                    crop[crop > background] = background
                imgs[a:a+patch_size, b:b+patch_size, k] = crop
                mu = np.mean(crop)
                NV.append(np.sum((crop - mu) ** 2) / (patch_size**2 * mu))
            loc = int(np.argmax(NV))
            o_fusion[a:a+patch_size, b:b+patch_size] += imgs[a:a+patch_size, b:b+patch_size, loc]
    return o_fusion * balance_map


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/case_000.npz"
    dst = sys.argv[2] if len(sys.argv) > 2 else "workspace/aif.npy"
    stack = np.load(src)["amplitude_stack"].astype("float32")
    stack = (stack - stack.min()) / (stack.max() - stack.min() + 1e-8)
    aif = all_in_focus_normal_variance(stack)
    np.save(dst, aif)
    print("wrote", dst, aif.shape)


if __name__ == "__main__":
    main()

# Seismic LSRTM Original (From-Scratch) - Progress

## 环境

- **venv**: `source /home/pisquare/jiangxiaoyi/GIDD_xr/gidd_mini/.venv/bin/activate`
- **GPU**: RTX 5090 (`CUDA_VISIBLE_DEVICES=1`)

## Parity Test (`test_parity_deepwave.py`)

Scratch C-PML Born solver vs deepwave (both accuracy=4):

| 测试项 | rel_l2 |
|--------|--------|
| Forward (homogeneous) | 1.45e-06 |
| Born (homogeneous) | 4.17e-06 |
| Born (Marmousi) | 9.04e-06 |

**PASS**: 所有差异在浮点精度范围内 (< 1e-4)。

## 运行结果 (3 L-BFGS epochs, 81.1s)

| Epoch | Loss | Scatter range |
|-------|------|---------------|
| 1 | 3.10e+01 | [-312, 209] |
| 2 | 2.04e+01 | [-857, 1162] |
| 3 | 1.51e+01 | [-1634, 2342] |

| 指标 | 值 |
|------|-----|
| Data rel. L2 (mean) | 0.2170 |
| Final loss | 1.51e+01 |

与 deepwave 版本 (seismic_lsrtm_main) 的 0.2188 非常接近。

## 测试

`python -m pytest evaluation/tests/ -v` — **30/30 passed**

## Notebook

`notebooks/seismic_lsrtm_original.ipynb` — `jupyter nbconvert --execute` 通过

## Completion Checklist

- [x] `python main.py` runs to completion
- [x] `python -m pytest evaluation/tests/ -v` — all 30 tests pass
- [x] `notebooks/seismic_lsrtm_original.ipynb` exists and executes without errors
- [x] `evaluation/reference_outputs/metrics.json` exists
- [x] `README.md`, `plan/approach.md`, `plan/design.md` exist
- [x] `requirements.txt` lists all dependencies
- [x] Parity test vs deepwave: all < 1e-4 rel L2

## 备注

- 使用 4 阶 FD（scratch 实现），deepwave 版本 (seismic_lsrtm_main) 使用 8 阶
- Born 耦合项: `2*v*scatter*dt^2 * W_bg` + 源点散射注入 `-2*v(xs)*scatter(xs)*dt^2`
- 运行时间 ~81s vs deepwave 版 ~15s（scratch 版无 C/CUDA kernel 加速）

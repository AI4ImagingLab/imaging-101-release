"""Generic end-to-end reconstruction scoring (canonical Inverse-101 design).

Ported from the reference implementation. The benchmark contract is:

  * the agent always writes a SINGLE ``output/reconstruction.npy`` (one real
    numeric array) — no per-task keys, formats, or complex/phase requirements;
  * the scorer auto-discovers the task's reference array (``.npy`` or ``.npz``),
    selecting the right ``.npz`` key by (a) matching the agent reconstruction's
    shape and (b) a ranked list of canonical names;
  * arrays are squeezed (ndim>2), complex is reduced to magnitude, and a single
    metric set (flux-normalized NRMSE/NCC/MSE/PSNR/SSIM) is computed.

This replaces the per-task ``task_scoring`` recipes for the live end-to-end
score so the agent only ever has to emit ``output/reconstruction.npy``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

REFERENCE_ARRAY_KEYS = (
    "ground_truth", "gt", "truth", "target", "reference",
    "reference_reconstruction", "reconstruction", "phantom", "image",
    "volume", "delta_n", "tissue_map", "bone_map", "y_pred", "spectrum",
)


def _reference_path_candidates(task_dir: Path) -> list[Path]:
    ref_dir = task_dir / "evaluation" / "reference_outputs"
    data_dir = task_dir / "data"
    ordered = [
        ref_dir / "ground_truth.npy", ref_dir / "ground_truth.npz",
        data_dir / "ground_truth.npy", data_dir / "ground_truth.npz",
        ref_dir / "reference.npy", ref_dir / "reference.npz",
        ref_dir / "reference_reconstruction.npy", ref_dir / "reference_reconstruction.npz",
        ref_dir / "reconstruction.npy", ref_dir / "reconstruction.npz",
        ref_dir / "reconstructions.npy", ref_dir / "reconstructions.npz",
        data_dir / "reference.npy", data_dir / "reference.npz",
        data_dir / "baseline_reference.npy", data_dir / "baseline_reference.npz",
    ]
    if ref_dir.is_dir():
        tokens = ("ground_truth", "reference", "reconstruction", "recon", "gt")
        ordered.extend(
            p for p in sorted(ref_dir.iterdir())
            if p.suffix.lower() in (".npy", ".npz")
            and any(tok in p.stem.lower() for tok in tokens)
        )
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in ordered:
        if p not in seen:
            unique.append(p); seen.add(p)
    return unique


def _prepare_metric_array(array: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == object:
        raise ValueError(f"{label} is not a valid numeric array")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{label} is not numeric")
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError(f"{label} has wrong dimensions: 0 (expected at least 1)")
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float64)


def _key_rank(key: str) -> tuple[int, int, str]:
    lowered = key.lower()
    for i, pref in enumerate(REFERENCE_ARRAY_KEYS):
        if lowered == pref:
            return (0, i, lowered)
    for i, pref in enumerate(REFERENCE_ARRAY_KEYS):
        if pref in lowered:
            return (1, i, lowered)
    return (2, len(REFERENCE_ARRAY_KEYS), lowered)


def _load_array_file(path: Path, target_shape: tuple[int, ...] | None = None) -> np.ndarray:
    loaded = np.load(str(path), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            candidates: list[tuple[tuple[int, int, str], str, np.ndarray]] = []
            errors: list[str] = []
            for key in loaded.files:
                try:
                    arr = _prepare_metric_array(loaded[key], label=f"{path.name}:{key}")
                except ValueError as exc:
                    errors.append(str(exc)); continue
                if target_shape is not None and arr.shape != target_shape:
                    continue
                candidates.append((_key_rank(key), key, arr))

            if not candidates and target_shape is not None:
                for key in loaded.files:
                    try:
                        arr = _prepare_metric_array(loaded[key], label=f"{path.name}:{key}")
                    except ValueError:
                        continue
                    candidates.append((_key_rank(key), key, arr))
                if len(candidates) == 1:
                    return candidates[0][2]
                shapes = {key: list(arr.shape) for _r, key, arr in candidates}
                raise ValueError(
                    f"{path.name} has no array matching output shape {target_shape}; "
                    f"available shapes: {shapes}")

            if not candidates:
                detail = "; ".join(errors[:3]) if errors else "no arrays found"
                raise ValueError(f"{path.name} does not contain a usable numeric array ({detail})")

            candidates.sort(key=lambda item: item[0])
            return candidates[0][2]
        finally:
            loaded.close()
    return _prepare_metric_array(loaded, label=path.name)


def load_reference_array(task_dir: Path, target_shape: tuple[int, ...] | None = None
                         ) -> tuple[np.ndarray, Path]:
    candidates = _reference_path_candidates(task_dir)
    existing = [p for p in candidates if p.exists()]
    errors: list[str] = []
    for p in existing:
        try:
            return _load_array_file(p, target_shape=target_shape), p
        except Exception as exc:
            errors.append(f"{p.name}: {exc}")
    tried = ", ".join(p.name for p in candidates[:10])
    if errors:
        raise ValueError(f"reference array not usable; tried {tried}; errors: {' | '.join(errors[:4])}")
    raise FileNotFoundError(f"reference array not found; tried {tried}")


def compute_quality_metrics_from_arrays(out: np.ndarray, gt: np.ndarray) -> dict:
    if out.shape != gt.shape:
        return {"error": f"Shape mismatch: reconstruction {out.shape} vs expected {gt.shape}",
                "expected_shape": list(gt.shape)}
    # NRMSE = range-normalized RMSE = RMSE / (max(gt) - min(gt)), matching each
    # task's compute_metrics convention and the metrics.json *_boundary values.
    # No flux/scale normalization: the agent must output a correctly-scaled
    # reconstruction (NCC below is cosine and thus scale-invariant regardless).
    mse = float(np.mean((out - gt) ** 2))
    rng = float(gt.max() - gt.min())
    nrmse = float(np.sqrt(mse) / rng) if rng > 0 else float("inf")
    ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
    max_val = float(gt.max())
    psnr = float(20 * np.log10(max_val / np.sqrt(mse))) if mse > 0 else float("inf")

    def _ssim_2d(a, b, data_range):
        c1 = (0.01 * data_range) ** 2; c2 = (0.03 * data_range) ** 2
        mu_a, mu_b = a.mean(), b.mean(); sig_a2, sig_b2 = a.var(), b.var()
        sig_ab = np.mean((a - mu_a) * (b - mu_b))
        num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (sig_a2 + sig_b2 + c2)
        return float(num / den)

    return {"nrmse": round(nrmse, 6), "ncc": round(ncc, 6), "mse": round(mse, 10),
            "psnr": round(psnr, 2), "ssim": round(_ssim_2d(out, gt, max_val), 6)}


def score_reconstruction(task_dir: Path, recon_dir: Path) -> dict:
    """Score output/reconstruction.npy in recon_dir (or recon_dir/output) against
    the task's auto-discovered reference array. Returns metrics or {'error': ...}."""
    recon_dir = Path(recon_dir)
    out_path = None
    for base in (recon_dir, recon_dir / "output"):
        p = base / "reconstruction.npy"
        if p.exists():
            out_path = p; break
    if out_path is None:
        return {"error": "output/reconstruction.npy not found"}
    try:
        out = _prepare_metric_array(np.load(out_path, allow_pickle=True), label="reconstruction")
    except Exception as exc:
        return {"error": str(exc)}
    try:
        gt, _ = load_reference_array(Path(task_dir), target_shape=out.shape)
    except Exception as exc:
        return {"error": f"Failed to load reference array: {exc}"}
    metrics = compute_quality_metrics_from_arrays(out, gt)
    if "error" not in metrics:
        _apply_boundaries(Path(task_dir), metrics)
    return metrics


def _apply_boundaries(task_dir: Path, metrics: dict) -> None:
    """Attach metrics.json boundaries + pass/fail (NCC>=ncc_boundary AND
    NRMSE<=nrmse_boundary) when both boundaries are defined for the task."""
    import json as _json
    mj = None
    for rel in ("evaluation/metrics.json", "evaluation/reference_outputs/metrics.json"):
        p = task_dir / rel
        if p.exists():
            try:
                mj = _json.loads(p.read_text())
            except Exception:
                mj = None
            break
    if not mj:
        return
    nb, rb = mj.get("ncc_boundary"), mj.get("nrmse_boundary")
    metrics["boundaries"] = {"ncc_boundary": nb, "nrmse_boundary": rb}
    if nb is not None and rb is not None:
        metrics["passed"] = bool(metrics["ncc"] >= nb and metrics["nrmse"] <= rb)
    else:
        metrics["passed"] = None

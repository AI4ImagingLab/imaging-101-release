# End-to-End Evaluation Guide

How to run the end-to-end (`end2end`) mechanism and how the results are scored
and aggregated.

In end-to-end mode an LLM agent is given a task's problem statement and data and
must build a **full reconstruction pipeline** that writes a single output file,
`output/reconstruction.npy`. The harness then scores that output against the
task's reference array and its pass/fail boundaries. No unit tests are run in
this mode — reconstruction quality is the sole criterion.

---

## 1. Prerequisites

- Python env with the harness deps (`numpy`, `requests`) plus, for in-process
  scoring, `scipy`/`scikit-image` are optional (only `numpy` is required by the
  generic scorer). The per-task solver runs in an isolated venv the harness
  builds automatically from `tasks/<task>/requirements.txt`.
- Task assets present under `tasks/<task>/data/` and
  `tasks/<task>/evaluation/` (download with `python scripts/download_assets.py
  --task <task>` or `--all`).
- An OpenAI-compatible endpoint + key for the model under test.

Set credentials (or pass them as flags):

```bash
export MODEL="Vendor2/GPT-5.5"
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="…"
```

---

## 2. Run one task

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism end2end \
  --level L1 \
  --framework react \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --max-iterations 5 \
  --max-tokens 32768 \
  --timeout 600 \
  --output results/end2end
```

Key options:

| Flag | Meaning |
| --- | --- |
| `--mechanism` | `end2end` (aliases: `end_to_end`, `e2e`) |
| `--level` | `L1` (README+data only), `L2` (+`approach.md`), `L3` (+`design.md`) |
| `--framework` | `react` (single-agent loop) or `multi_agent` (Plan→Architect→Coder→Judge) |
| `--max-iterations` | ReAct: single steps **per phase**; multi_agent: pipeline **cycles** |
| `--max-tokens` | Max completion tokens per LLM call (lower ⇒ faster for verbose reasoning models) |
| `--timeout` | Per-shell-command timeout (seconds) for solver execution |
| `--output` | Directory for the result JSON |

Use `--dry-run` to validate a command without calling the model.

### The agent output contract

The agent must save its reconstruction to **`output/reconstruction.npy`** as a
single real numeric array. It does **not** need to match any task-specific key
or layout — but its **shape must equal the reference array's shape** (after
squeezing singleton dims). Complex outputs are reduced to magnitude by the
scorer.

---

## 3. How scoring works (`evaluation_harness/reference_scoring.py`)

After the agent finishes, `Scorer._compute_quality_metrics` calls
`reference_scoring.score_reconstruction(task_dir, sandbox_dir)` in-process:

1. **Find the agent output** — `reconstruction.npy` in the sandbox root or its
   `output/` subdir. Missing ⇒ `{"error": "output/reconstruction.npy not found"}`.
2. **Auto-discover the reference array** — searches, in priority order,
   `evaluation/reference_outputs/` then `data/` for
   `ground_truth.{npy,npz}`, `reference*`, `reference_reconstruction*`,
   `reconstruction(s)*`, `baseline_reference*`, plus any `*gt/reference/recon*`
   named array in `reference_outputs/`.
3. **Select the array inside an `.npz`** — among all keys, pick the one whose
   (squeezed) shape matches the agent output, tie-broken by a ranked list of
   canonical names (`ground_truth, gt, …, phantom, image, volume, delta_n,
   tissue_map, bone_map, y_pred, spectrum`).
4. **Normalize both arrays** — squeeze dims > 2, take `np.abs` if complex, cast
   to float64.
5. **Shape check** — mismatch ⇒
   `{"error": "Shape mismatch: reconstruction (…) vs expected (…)"}`.
6. **Compute metrics** on the aligned arrays.

### Metric definitions

```
NRMSE = sqrt(mean((out - gt)^2)) / (max(gt) - min(gt))      # range-normalized
NCC   = <out, gt> / (||out|| · ||gt||)                       # cosine similarity
MSE   = mean((out - gt)^2)
PSNR  = 20 · log10(max(gt) / sqrt(MSE))
SSIM  = global single-window SSIM
```

- **NRMSE is range-normalized** (divided by the ground-truth dynamic range) to
  match each task's own `compute_metrics` convention and the `*_boundary`
  values. There is **no flux/scale normalization** — the agent must output a
  correctly-scaled reconstruction. (NCC is cosine and therefore scale-invariant.)

### Pass / fail

Boundaries are read from `tasks/<task>/evaluation/metrics.json`:

```
passed = (NCC >= ncc_boundary) AND (NRMSE <= nrmse_boundary)
```

If both boundaries are present the result gets `"passed": true/false`; otherwise
`"passed": null`.

---

## 4. Result files

Each run writes one JSON to `--output`, named
`<task>_end_to_end_<level>_<framework>_<model>_<timestamp>.json`. The relevant
field is `quality_metrics`, e.g.:

```json
"quality_metrics": {
  "ncc": 0.9933, "nrmse": 0.0512, "mse": 1.3e-07, "psnr": 40.05, "ssim": 0.9966,
  "boundaries": {"ncc_boundary": 0.894, "nrmse_boundary": 0.0563},
  "passed": true,
  "scorer": "generic"
}
```

On failure the field is `{"error": "...", "scorer": "generic"}` (e.g.
`output/reconstruction.npy not found`, `Shape mismatch: …`).

A single task is a **success** iff `quality_metrics.passed == true`.

---

## 5. Calculating / aggregating results

A task passes when `passed == true`. To tally a model's success count across a
results directory:

```python
import glob, json

def tally(results_dir):
    n = produced = nccpass = passed = 0
    for f in glob.glob(f"{results_dir}/*.json"):
        q = (json.load(open(f)).get("quality_metrics") or {})
        n += 1
        if "error" in q or q.get("ncc") is None:
            continue
        produced += 1                                   # a scorable reconstruction
        b = q.get("boundaries") or {}
        if b.get("ncc_boundary") is not None and q["ncc"] >= b["ncc_boundary"]:
            nccpass += 1                                # correlation clears bound
        if q.get("passed"):
            passed += 1                                 # both NCC & NRMSE clear
    return dict(tasks=n, produced=produced, ncc_pass=nccpass, success=passed)

print(tally("results/end2end"))
```

Report three levels, most-informative last:

- **produced** — the agent emitted a scorable `reconstruction.npy` of the right shape;
- **ncc_pass** — the reconstruction is well-correlated (`NCC ≥ ncc_boundary`);
- **success** — passes **both** boundaries (`passed == true`).

### Batch over tasks / models

```bash
for t in ct_fan_beam mri_grappa eht_black_hole_original; do
  python -m evaluation_harness run --task "$t" --mechanism end2end --level L1 \
    --framework multi_agent --model "$MODEL" --base-url "$BASE_URL" \
    --api-key "$API_KEY" --max-iterations 5 --output results/end2end
done
```

Then run `tally("results/end2end")`. For a model-vs-model comparison, keep each
model's runs in its own `--output` dir and tally each.

---

## 6. Caveats to keep in mind

- **NRMSE is scale-sensitive.** A reconstruction that is correct up to a global
  scale factor passes NCC (≈1.0) but can fail NRMSE. If you want to credit such
  results, add a scale-invariant NRMSE (least-squares fit a global factor
  `s = Σ(out·gt)/Σ(out²)` before the RMSE) — not enabled by default.
- **Shape must match the reference.** Multi-frame / multi-channel / volumetric /
  RGB tasks require the agent to produce the exact reference layout; a flattened
  or transposed array yields a `Shape mismatch` error even if the content is
  right.
- **Boundaries are per task** and can be strict; some tasks demand near-perfect
  reconstructions.
- **Run-to-run variance is high** at `temperature > 0`; a task's pass/fail can
  flip between runs. For a defensible number, run multiple seeds and report a
  pass-rate rather than a single run.
- `results/code_archive/<ts>_<task>/` keeps the agent's generated `src/`,
  `main.py`, and `output/` after each run, so a run can be re-scored later
  without re-invoking the model:
  `python -c "from evaluation_harness import reference_scoring as rs; \
   print(rs.score_reconstruction('tasks/<task>', 'results/code_archive/<ts>_<task>'))"`.

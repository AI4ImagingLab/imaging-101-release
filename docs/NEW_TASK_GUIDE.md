# How to Add a New Task

Step-by-step guide for contributing a new computational imaging task to the imaging-101 benchmark.

---

## Overview

Each task represents a **computational imaging inverse problem**: given observed data $y$, recover the underlying signal $x$ using a known forward model $A$ where $y = A(x) + \text{noise}$.

A complete task includes:
- Problem definition and reference implementation
- Raw data (real or synthetic)
- Unit tests and reference outputs for automated evaluation
- A tutorial notebook for human review

---

## Step 1: Create the Directory Structure

```bash
mkdir -p tasks/your_task_name/{data,plan,src,evaluation/{tests,fixtures,reference_outputs},notebooks}
```

This creates:

```
tasks/your_task_name/
├── README.md                          # Problem definition
├── requirements.txt                   # Python dependencies
├── main.py                           # Pipeline entry point
├── data/
│   ├── raw_data.npz                  # Observation data
│   └── meta_data.json                # Imaging parameters
├── plan/
│   ├── approach.md                   # Solution methodology
│   └── design.md                     # Code architecture + function signatures
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Raw data -> processed observations
│   ├── physics_model.py              # Forward model: x -> y
│   ├── solvers.py                    # Inverse solver: y -> x_hat
│   ├── visualization.py              # Plotting utilities
│   └── generate_data.py              # (optional) Synthetic data generation
├── evaluation/
│   ├── metrics.json                  # Baseline method metrics (NCC/NRMSE)
│   ├── reference_outputs/            # Ground truth, checkpoints, samples
│   │   ├── ground_truth.npy
│   │   └── metrics.json
│   ├── fixtures/                     # Per-function test data
│   │   └── *.npz
│   └── tests/                        # Unit tests
│       ├── __init__.py
│       ├── test_physics_model.py
│       ├── test_preprocessing.py
│       └── test_solvers.py
└── notebooks/
    └── your_task_name.ipynb          # Tutorial notebook
```

---

## Step 2: Write the README

The README is the **primary document an agent reads** to understand the task. Follow this structure:

```markdown
# Your Task Name

One-line summary of the imaging problem.

- **Domain**: Medicine / Astronomy / Biology / Physics / Chemistry / Earth Science
- **Keywords**: comma-separated technical keywords
- **Difficulty**: Easy / Medium / Hard

## Background

2-3 paragraphs explaining the physics and motivation. Include key equations.

## Problem Description

What the agent must implement: given inputs, produce outputs.
Define the forward model mathematically.

## Data Description

### Input
- `data/raw_data.npz`: describe each array and its shape/dtype
- `data/meta_data.json`: describe each parameter

### Output
- `output/reconstruction.npy`: expected output shape and meaning

## Method Hints

Conceptual guidance (not implementation details):
- Which algorithm family to use
- Key regularization approaches
- Important numerical considerations

## References

1. Author et al. (Year) Title. Journal.
```

**Rules:**
- Do NOT include implementation code or function signatures in the README
- Method Hints should be conceptual, not prescriptive
- Use LaTeX math notation: `$...$` for inline, `$$...$$` for display

---

## Step 3: Prepare the Data

### Option A: Real observation data

```python
# Save observation data as .npz
import numpy as np
np.savez('tasks/your_task/data/raw_data.npz',
         observations=obs_data,       # shape: describe
         sampling_mask=mask)           # shape: describe
```

### Option B: Synthetic data

Write `src/generate_data.py`:

```python
def generate_synthetic_data(meta_data: dict) -> dict:
    """Generate synthetic observation data.
    
    Returns:
        dict with keys matching raw_data.npz contents
    """
    # 1. Create ground truth phantom
    # 2. Apply forward model
    # 3. Add noise
    return {"observations": y, "ground_truth": x_true}
```

### Metadata file

Create `data/meta_data.json` with imaging parameters only (no solver parameters):

```json
{
    "image_size": 128,
    "num_measurements": 256,
    "noise_level": 0.01,
    "wavelength": 0.5e-6
}
```

---

## Step 4: Implement the Source Code

### `src/physics_model.py`

The forward model $A: x \to y$:

```python
def forward(x, meta_data):
    """Apply forward model.
    
    Args:
        x: Image/signal to transform, shape (N, N)
        meta_data: dict from meta_data.json
    Returns:
        y: Measurements, shape (M,)
    """
    ...

def adjoint(y, meta_data):
    """Apply adjoint operator A^H.
    
    Args:
        y: Measurements, shape (M,)
        meta_data: dict from meta_data.json
    Returns:
        x: Back-projected image, shape (N, N)
    """
    ...
```

### `src/preprocessing.py`

Load and prepare raw data:

```python
def load_and_preprocess(data_dir, meta_data):
    """Load raw_data.npz and return processed observations.
    
    Returns:
        dict with processed arrays ready for the solver
    """
    ...
```

### `src/solvers.py`

Reconstruction algorithm:

```python
def reconstruct(observations, forward_op, adjoint_op, meta_data, **kwargs):
    """Run reconstruction algorithm.
    
    Returns:
        reconstruction: np.ndarray, the recovered image/signal
    """
    ...
```

### `src/visualization.py`

```python
def plot_results(reconstruction, ground_truth, save_dir):
    """Generate comparison figures."""
    ...
```

### `main.py`

Pipeline entry point that ties everything together:

```python
import matplotlib
matplotlib.use('Agg')  # Only set backend in main.py, never in src/

from src.preprocessing import load_and_preprocess
from src.physics_model import forward, adjoint
from src.solvers import reconstruct
from src.visualization import plot_results

def main():
    # 1. Load data
    # 2. Preprocess
    # 3. Reconstruct
    # 4. Save output/reconstruction.npy
    # 5. Visualize
    ...

if __name__ == "__main__":
    main()
```

---

## Step 5: Generate Reference Outputs

Run the pipeline and save all outputs:

```bash
cd tasks/your_task_name
pip install -r requirements.txt
python main.py
```

Save to `evaluation/reference_outputs/`:
- `ground_truth.npy` — the true signal
- `reconstruction.npy` — reference method output  
- `metrics.json` — baseline NCC/NRMSE values
- Any intermediate outputs needed for tests

### metrics.json format

```json
{
    "your_method": {
        "ncc": 0.95,
        "nrmse": 0.15
    },
    "baseline_method": {
        "ncc": 0.80,
        "nrmse": 0.45
    }
}
```

---

## Step 6: Write Unit Tests

### Generate fixtures

Create input-output pairs for each function:

```python
# generate_fixtures.py
import numpy as np
from src.physics_model import forward, adjoint

# Small test case
x_test = np.random.randn(16, 16)
y_test = forward(x_test, meta_data)
np.savez('evaluation/fixtures/physics_model.npz',
         x_input=x_test, y_expected=y_test, meta_data=meta_data)
```

### Write test files

`evaluation/tests/test_physics_model.py`:

```python
import numpy as np
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"

class TestForwardModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        data = np.load(FIXTURES / "physics_model.npz", allow_pickle=True)
        self.x_input = data["x_input"]
        self.y_expected = data["y_expected"]
        self.meta_data = data["meta_data"].item()

    def test_forward_shape(self):
        from src.physics_model import forward
        y = forward(self.x_input, self.meta_data)
        assert y.shape == self.y_expected.shape

    def test_forward_values(self):
        from src.physics_model import forward
        y = forward(self.x_input, self.meta_data)
        np.testing.assert_allclose(y, self.y_expected, rtol=1e-5)

    def test_adjoint_shape(self):
        from src.physics_model import adjoint
        x = adjoint(self.y_expected, self.meta_data)
        assert x.shape == self.x_input.shape
```

Run tests to verify:

```bash
cd tasks/your_task_name
python -m pytest evaluation/tests/ -v
```

---

## Step 7: Write the Plan Documents

### `plan/approach.md`

Describe the solution methodology (what an agent at L2 difficulty would see):

```markdown
# Approach

## Algorithm Overview
Describe the reconstruction algorithm at a conceptual level.

## Key Steps
1. Data loading and preprocessing
2. Forward model construction  
3. Iterative reconstruction with regularization
4. Post-processing

## Mathematical Formulation
Key equations for the optimization problem.
```

### `plan/design.md`

Code architecture with function signatures (what an agent at L3 difficulty would see):

```markdown
# Design

## File Structure
- `src/preprocessing.py`: Data I/O and preparation
- `src/physics_model.py`: Forward/adjoint operators
- `src/solvers.py`: Reconstruction algorithm

## Function Signatures

### preprocessing.py
- `load_and_preprocess(data_dir, meta_data) -> dict`

### physics_model.py  
- `forward(x, meta_data) -> np.ndarray`
- `adjoint(y, meta_data) -> np.ndarray`

### solvers.py
- `reconstruct(observations, ...) -> np.ndarray`
```

---

## Step 8: Create the Notebook

`notebooks/your_task_name.ipynb` is the **primary user-facing deliverable**:

- Loads precomputed results from `evaluation/reference_outputs/` (runs in seconds)
- Shows: ground truth, reconstruction, comparison figures, metrics
- Includes commented-out code for running the full pipeline from scratch
- Must execute without errors: `jupyter nbconvert --execute notebooks/your_task_name.ipynb`

---

## Step 9: Write requirements.txt

List all Python dependencies:

```
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
scikit-image>=0.19
```

---

## Step 10: Verify and Evaluate

### Completion checklist

```bash
# All of these must succeed:
cd tasks/your_task_name

python main.py                              # Pipeline runs
python -m pytest evaluation/tests/ -v       # All tests pass
jupyter nbconvert --execute notebooks/*.ipynb  # Notebook runs
```

Verify these files exist:
- [ ] `README.md`, `main.py`, `requirements.txt`
- [ ] `data/raw_data.npz`, `data/meta_data.json`
- [ ] `plan/approach.md`, `plan/design.md`
- [ ] `src/physics_model.py`, `src/preprocessing.py`, `src/solvers.py`
- [ ] `evaluation/metrics.json` or `evaluation/reference_outputs/metrics.json`
- [ ] `evaluation/tests/test_*.py` (at least one)
- [ ] `notebooks/your_task_name.ipynb`

### Run the benchmark on your new task

```bash
# Function-mode: test if an agent can implement each module
python -m evaluation_harness run \
    --task your_task_name \
    --mode function \
    --target-function physics_model \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key $API_KEY \
    --framework react \
    --output results/function_mode \
    -v

# End-to-end: test if an agent can build the full pipeline
python -m evaluation_harness run \
    --task your_task_name \
    --mode end_to_end \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key $API_KEY \
    --framework react \
    --level L1 \
    --output results/end_to_end \
    -v
```

---

## Common Pitfalls

1. **Solver parameters in meta_data.json** — Only include imaging/physics parameters, not algorithm hyperparameters
2. **Information leakage** — Don't reference packages published after the original paper
3. **`matplotlib.use('Agg')` in src/** — Only call this in `main.py`, never in library code (breaks notebooks)
4. **Coordinate convention mismatch** — Ensure `generate_data.py` and `physics_model.py` use the same pixel coordinate convention. Verify by round-tripping: forward-model the ground truth and check $\chi^2/\text{DOF} \approx 1$
5. **Cross-task imports** — Each task must be fully self-contained

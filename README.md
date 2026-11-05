# Prior-data fitted networks for variable selection

> Two pretrained prior-data fitted networks that return posterior inclusion probabilities for every covariate in one forward pass: **CD-PFN** estimates whether each covariate is conditionally dependent on the response, and **PC-PFN** estimates whether each covariate is a parent or child of the response in a directed acyclic graph.

## Associated paper

- **Prior-data fitted networks for variable selection**  
	[arXiv](https://arxiv.org/abs/2611.05966)

## Installation
```
pip install varselpfn
```

## Usage
```python
import numpy as np
from varselpfn import CDPFN, PCPFN

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 50))
y = X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(size=200)

# CD-PFN
cd = CDPFN()
pips = cd.predict(X, y)  # shape (50,)
selected = cd.select(X, y, alpha=0.1)  # indices selected at target FDR 0.1

# PC-PFN
pc = PCPFN()
pips = pc.predict(X, y)  # shape (50, 2): parent and child PIPs
selected = pc.select(X, y, alpha=0.1)  # {'parents': [...], 'children': [...]}
```

- `X` is an `(n, p)` array or DataFrame of numeric covariates; `y` is continuous or binary (0/1). No preprocessing is needed, but missing values are not allowed.
- The networks were trained on datasets with 16 to 1024 observations and 8 to 512 covariates.
- Pass `device='cuda'` (or `'mps'`) to run on a GPU.

## Fine-tuning

To adapt a network to your data, copy its config ([`cd_pfn.json`](https://github.com/omelikechi/varselpfn/blob/main/varselpfn/configs/cd_pfn.json) or [`pc_pfn.json`](https://github.com/omelikechi/varselpfn/blob/main/varselpfn/configs/pc_pfn.json)), edit the `prior` block (options in [`prior.py`](https://github.com/omelikechi/varselpfn/blob/main/varselpfn/prior.py)), and continue training from the released weights:

```
python -m varselpfn.train --config my_prior.json --init cd_pfn --run-dir my_run --max-minutes 120
```

Then load it with `CDPFN(weights='my_run/best.pt')`. For PC-PFN, use `--init pc_pfn` and `PCPFN(weights=...)`.

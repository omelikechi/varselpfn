# varselpfn

Variable selection with prior-data fitted networks. Given a dataset `(X, y)`, two pretrained networks return posterior inclusion probabilities (PIPs) for every covariate in one forward pass, with nothing fitted to your data:

- **CD-PFN:** the probability that `y` depends on each covariate given the other covariates (conditional dependence).
- **PC-PFN:** the probability that each covariate is a parent (direct cause) or a child (direct effect) of `y`.

Selections at a target false discovery rate (FDR) use the Bayesian FDR rule. The method is described in the paper *Prior-data fitted networks for variable selection* (Melikechi, 2026).

## Installation

```
pip install git+https://github.com/omelikechi/varselpfn
```

This installs the package with both networks' weights (about 21 MB each). It requires Python 3.10 or later and PyTorch; a GPU is not needed.

## Usage

```python
import numpy as np
from varselpfn import CDPFN, PCPFN

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 50))
y = X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(size=200)

cd = CDPFN()
pips = cd.predict(X, y)              # shape (50,): one PIP per covariate
selected = cd.select(X, y, alpha=0.1)  # indices selected at target FDR 0.1

pc = PCPFN()
pips = pc.predict(X, y)              # shape (50, 2): parent and child PIPs
selected = pc.select(X, y, alpha=0.1)  # {'parents': [...], 'children': [...]}
```

- `X` is an `(n, p)` array or pandas DataFrame of numeric covariates (continuous, binary, or ordinal), and `y` is a continuous or binary (0/1) response. With a DataFrame, results are labeled by column name.
- Columns are standardized inside the networks, so no preprocessing is needed. Missing values are not allowed.
- The networks were trained on datasets with 16 to 1024 observations and 8 to 512 covariates. They also work on larger datasets, but run time and memory grow with their size.
- `bayes_select(pips, alpha)` applies the selection rule to any PIPs: it returns the largest set of covariates with the highest PIPs whose average of `1 - PIP` is at most `alpha`.
- Pass `device='cuda'` (or `'mps'`) to run on a GPU.

## Fine-tuning

Each network was trained on simulated datasets from a prior over data-generating models, specified in a JSON config: [`cd_pfn.json`](varselpfn/configs/cd_pfn.json) and [`pc_pfn.json`](varselpfn/configs/pc_pfn.json). To adapt a network to your application, copy its config, change the `prior` block to describe your data, and continue training from the released weights. For example, to fine-tune CD-PFN for small samples, set `"n_range": [16, 128]` and run

```
python -m varselpfn.train --config my_prior.json --init cd_pfn --run-dir my_run --max-minutes 120
```

Then use the fine-tuned network with `CDPFN(weights='my_run/best.pt')`. The prior's options are documented in [`prior.py`](varselpfn/prior.py), and the training options in [`train.py`](varselpfn/train.py). Training runs on a CPU but is much faster on a GPU. For PC-PFN, start from `pc_pfn.json` with `--init pc_pfn`, and use `PCPFN(weights='my_run/best.pt')`.

A prior can also be a different simulator altogether. [`gprior.py`](varselpfn/gprior.py) is a linear regression prior with Zellner's g-prior, whose exact PIPs it also computes; set `"family": "gprior"` in the config's `prior` block to fine-tune on it, as in the paper's posterior accuracy check.

## Reproducing the paper

The code for every table and figure in the paper, and the configs used to train both networks, are in [varselpfn-paper](https://github.com/omelikechi/varselpfn-paper).

## Citation

```bibtex
@article{melikechi2026,
  title = {Prior-data fitted networks for variable selection},
  author = {Melikechi, Omar},
  year = {2026}
}
```

## License

MIT. The graph sampler in `prior.py` is adapted from TabICL (BSD-3-Clause); see [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).

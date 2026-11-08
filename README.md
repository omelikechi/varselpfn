# varselpfn

[![Paper](https://img.shields.io/badge/Paper-arXiv-1f6feb?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.05966)

Variable selection with pretrained prior-data fitted networks. In one forward pass:

- **CD-PFN** gives the posterior probability that the response depends on each covariate given the others.
- **PC-PFN** gives the posterior probabilities that each covariate is a parent or a child of the response in a directed acyclic graph.

## Installation

```
pip install varselpfn
```

## Quick start

```python
import numpy as np
from varselpfn import CDPFN, PCPFN

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 50))
y = X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(size=200)

cd = CDPFN()
pips = cd.predict(X, y)  # shape (50,)
selected = cd.select(X, y, alpha=0.1)  # array([0, 1]), by the Bayesian FDR rule at level 0.1

pc = PCPFN()
pips = pc.predict(X, y)  # shape (50, 2): parent and child
selected = pc.select(X, y, alpha=0.1)  # {'parents': [...], 'children': [...]}
```

With a DataFrame, results are indexed by column name. Any column types work (numeric, categorical, text, dates), and missing values are filled in.

## Notes

- `y` can be continuous, binary, or (for CD-PFN) multiclass. For a multiclass `y`, the scores are the mean of the one-vs-rest PIPs.
- The networks were trained on 16 to 1,024 rows and 8 to 512 covariates. More rows give better rankings, but PIPs from more than 1,024 rows can be overconfident, and `select()` warns when that happens. Pass `max_rows=1024` to stay within the training range.
- Rare classes and covariates with extreme values are handled automatically. See `help(CDPFN)` for all options.
- Pass `device='cuda'` (or `'mps'`) to run on a GPU.

## Fine-tuning

To adapt a network to your data, copy its config ([`cd_pfn.json`](https://github.com/omelikechi/varselpfn/blob/main/varselpfn/configs/cd_pfn.json) or [`pc_pfn.json`](https://github.com/omelikechi/varselpfn/blob/main/varselpfn/configs/pc_pfn.json)), edit the `prior` block (options in [`prior.py`](https://github.com/omelikechi/varselpfn/blob/main/varselpfn/prior.py)), and continue training from the released weights:

```
python -m varselpfn.train --config my_prior.json --init cd_pfn --run-dir my_run --max-minutes 120
```

Then load it with `CDPFN(weights='my_run/best.pt')`. For PC-PFN, use `--init pc_pfn` and `PCPFN(weights=...)`.

## Citation

```bibtex
@article{melikechi2026pfn,
  title={Prior-data fitted networks for variable selection},
  author={Melikechi, Omar},
  journal={arXiv preprint arXiv:2611.05966},
  year={2026}
}
```

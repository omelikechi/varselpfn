# CD-PFN and PC-PFN
'''
CDPFN and PCPFN wrap the pretrained networks. predict(X, y) returns estimated posterior inclusion probabilities (PIPs)
in one forward pass, with nothing fitted to the data; select(X, y, alpha) applies the Bayesian FDR rule to them.

Weights are the bundled ones ('cd_pfn', 'pc_pfn') unless a path is given: a .safetensors file or a checkpoint written
by varselpfn.train (best.pt), for example after fine-tuning.
'''

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
import torch

from varselpfn.model import Network

package_dir = Path(__file__).resolve().parent
bundled_weights = {'cd_pfn': package_dir / 'weights' / 'cd_pfn.safetensors', 'pc_pfn': package_dir / 'weights' / 'pc_pfn.safetensors'}

#----------------------------------------------------------------
# Weights
#----------------------------------------------------------------
def load_weights(weights):
    '''Returns (model config, state dict) for a bundled name or a path to a .safetensors file or a training checkpoint.'''
    path = bundled_weights.get(weights, Path(weights).expanduser())
    if path.suffix == '.safetensors':
        with safe_open(path, 'pt') as f:
            model_config = json.loads(f.metadata()['model'])
        return model_config, load_file(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint['config']['model'], checkpoint['model']

@lru_cache(maxsize=None)
def load_network(weights, device):
    '''Loaded once per (weights, device) and reused.'''
    model_config, state = load_weights(weights)
    model = Network(**(model_config | {'grad_checkpoint': False}))
    model.load_state_dict(state)
    return model.to(device).eval()

#----------------------------------------------------------------
# Selection
#----------------------------------------------------------------
def bayes_select(pips, alpha=0.1):
    '''
    Bayesian FDR rule: the largest set of covariates with the highest PIPs whose mean (1 - PIP), the estimated FDR, is
    at most alpha. Returns their indices, in decreasing order of PIP (ties by index).
    '''
    pips = np.asarray(pips, dtype=float)
    p = len(pips)
    order = np.lexsort((np.arange(p), -pips))
    running = np.cumsum(1 - pips[order]) / np.arange(1, p + 1)
    ok = np.flatnonzero(running <= alpha)
    k = ok.max() + 1 if len(ok) else 0
    return order[:k]

#----------------------------------------------------------------
# Models
#----------------------------------------------------------------
class PFN:
    '''Shared by CDPFN and PCPFN: input checks, the forward pass, and covariate names for pandas input.'''
    default_weights = None
    outputs = None

    def __init__(self, weights=None, device='cpu'):
        self.weights = str(weights) if weights is not None else self.default_weights
        self.device = torch.device(device)

    def probabilities(self, X, y):
        names = list(X.columns) if hasattr(X, 'columns') else None
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f'X must be (n, p) and y (n,); got X {X.shape} and y {y.shape}')
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError('X and y must not contain missing or infinite values')
        model = load_network(self.weights, self.device)
        with torch.no_grad():
            logits = model(torch.from_numpy(X)[None].to(self.device), torch.from_numpy(y)[None].to(self.device))[0]
        pips = torch.sigmoid(logits[:, self.outputs].float()).cpu().numpy().astype(float)
        return pips, names

class CDPFN(PFN):
    '''
    CD-PFN: for each covariate, the estimated posterior probability that y depends on it given the other covariates.

    X: (n, p) array or DataFrame of numeric covariates (continuous, binary, or ordinal). y: (n,) continuous or binary
    response. Trained on 16 <= n <= 1024 and 8 <= p <= 512; columns are standardized inside the model.
    '''
    default_weights = 'cd_pfn'
    outputs = 0

    def predict(self, X, y):
        '''PIPs, shape (p,); a Series indexed by the column names if X is a DataFrame.'''
        pips, names = self.probabilities(X, y)
        if names is None:
            return pips
        import pandas as pd
        return pd.Series(pips, index=names, name='pip')

    def select(self, X, y, alpha=0.1):
        '''Covariates selected at target FDR alpha: indices, or column names if X is a DataFrame.'''
        pips, names = self.probabilities(X, y)
        selected = bayes_select(pips, alpha)
        return selected if names is None else [names[j] for j in selected]

class PCPFN(PFN):
    '''
    PC-PFN: for each covariate, the estimated posterior probabilities that it is a parent (direct cause) and a child
    (direct effect) of y. Inputs as for CDPFN.
    '''
    default_weights = 'pc_pfn'
    outputs = [1, 2]

    def predict(self, X, y):
        '''PIPs, shape (p, 2) with columns (parent, child); a DataFrame indexed by the column names if X is one.'''
        pips, names = self.probabilities(X, y)
        if names is None:
            return pips
        import pandas as pd
        return pd.DataFrame(pips, index=names, columns=['parent', 'child'])

    def select(self, X, y, alpha=0.1):
        '''Parents and children selected at target FDR alpha, each by the Bayesian FDR rule: {'parents': ..., 'children': ...}.'''
        pips, names = self.probabilities(X, y)
        out = {'parents': bayes_select(pips[:, 0], alpha), 'children': bayes_select(pips[:, 1], alpha)}
        return out if names is None else {role: [names[j] for j in selected] for role, selected in out.items()}

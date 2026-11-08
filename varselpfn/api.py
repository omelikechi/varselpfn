# CD-PFN and PC-PFN
'''
CDPFN and PCPFN wrap the pretrained networks. predict(X, y) returns estimated posterior inclusion probabilities (PIPs)
in one forward pass; select(X, y, alpha) applies the Bayesian FDR rule to them. Inputs
are prepared automatically (covariates_matrix, response_vector, row_budget).

Weights are the bundled ones ('cd_pfn', 'pc_pfn') unless a path is given: a .safetensors file or a checkpoint written
by varselpfn.train (best.pt), for example after fine-tuning.
'''

from functools import lru_cache
import json
from pathlib import Path
import warnings

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

def case_control_rows(y, rng, threshold=0.2, ratio=3, min_rows=200):
    '''
    Rows for a binary response whose rarer class is under threshold of the rows (None otherwise): every row of the rarer
    class and ratio times as many random rows of the other (at least min_rows in all, while rows last). Sampling on y
    alone leaves unchanged whether y depends on a covariate given the others, and the networks were trained on binary
    responses whose rarer class has 20% to 50% of the rows.
    '''
    levels, counts = np.unique(y, return_counts=True)
    if len(levels) != 2 or counts.min() >= threshold * len(y):
        return None
    rare = np.flatnonzero(y == levels[np.argmin(counts)])
    common = np.flatnonzero(y != levels[np.argmin(counts)])
    m = min(len(common), max(ratio * len(rare), min_rows - len(rare)))
    if m == len(common):
        return None
    return np.sort(np.concatenate([rare, rng.choice(common, m, replace=False)]))

balance_ratio, balance_chunks = 3, 10

def case_control_chunks(y, rng, threshold=0.2, ratio=None, min_rows=200, max_chunks=None):
    '''
    Row sets for a binary response whose rarer class is under threshold of the rows (None otherwise): the rows of the
    other class are shuffled and split into disjoint chunks of about ratio times the rarer count (at least min_rows in
    all), and each set is every row of the rarer class plus one chunk; at most max_chunks sets, chosen at random. Every
    row is used at most once across the sets, and each set's rarer class has at least 1 / (1 + ratio) of its rows,
    within the 20% to 50% of training.
    '''
    ratio = balance_ratio if ratio is None else ratio
    max_chunks = balance_chunks if max_chunks is None else max_chunks
    levels, counts = np.unique(y, return_counts=True)
    if len(levels) != 2 or counts.min() >= threshold * len(y):
        return None
    rare = np.flatnonzero(y == levels[np.argmin(counts)])
    common = np.flatnonzero(y != levels[np.argmin(counts)])
    m = max(ratio * len(rare), min_rows - len(rare))
    if m >= len(common):
        return None
    chunks = np.array_split(rng.permutation(common), int(np.ceil(len(common) / m)))
    if len(chunks) > max_chunks:
        chunks = [chunks[i] for i in np.sort(rng.choice(len(chunks), max_chunks, replace=False))]
    return [np.sort(np.concatenate([rare, c])) for c in chunks]

def extreme_columns(X, z=50, min_values=10):
    '''
    Columns with at least min_values distinct values and a value more than z robust standard deviations (1.4826 times
    the median absolute deviation) from the median; for these the median absolute deviation is replaced by the 10%-90%
    interquantile range divided by 2.563 when it is 0.
    '''
    out = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        x = X[:, j]
        if len(np.unique(x)) < min_values:
            continue
        med = np.median(x)
        scale = 1.4826 * np.median(np.abs(x - med))
        if scale == 0:
            lo, hi = np.quantile(x, [0.1, 0.9])
            scale = (hi - lo) / 2.563
        out[j] = scale > 0 and np.abs(x - med).max() > z * scale
    return out

def rank_columns(X, columns):
    '''Columns replaced by their ranks (average ranks for ties) divided by n.'''
    from scipy.stats import rankdata
    X = X.copy()
    X[:, columns] = (rankdata(X[:, columns], axis=0) / len(X)).astype(X.dtype)
    return X

#----------------------------------------------------------------
# Inputs
#----------------------------------------------------------------
training_rows = 1024
budget_rows, budget_p = 10000, 100

def covariates_matrix(X):
    '''
    Float matrix (n, p) and column names (None for an array) from an array or a DataFrame of any column types. Numeric
    and boolean columns are used as they are; datetimes become seconds since 1970; text columns whose every non-missing
    value is a number become numeric. Other columns (text, and columns of categorical dtype) become integer codes of
    their sorted values, with the values seen only once pooled into one further code; so a column whose values are all
    distinct (an identifier, which carries no information) becomes constant. Missing and infinite values are replaced by the column
    median (0 if the column has none), so the PIPs refer to the covariates as filled in.
    '''
    names = None
    if hasattr(X, 'columns'):
        import pandas as pd
        names = list(X.columns)
        columns = []
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_datetime64_any_dtype(col):
                columns.append((col - pd.Timestamp(0, tz=col.dt.tz)).dt.total_seconds().to_numpy(dtype=float))
            elif pd.api.types.is_bool_dtype(col) or (pd.api.types.is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype)):
                columns.append(col.to_numpy(dtype=float, na_value=np.nan))
            else:
                if not isinstance(col.dtype, pd.CategoricalDtype):
                    parsed = pd.to_numeric(col, errors='coerce')
                    if col.notna().any() and parsed.notna().sum() == col.notna().sum():
                        columns.append(parsed.to_numpy(dtype=float, na_value=np.nan))
                        continue
                values = col.astype(str).where(col.notna())
                counts = values.value_counts()
                codes = {v: i for i, v in enumerate(sorted(counts.index[counts >= 2]))}
                codes.update({v: len(codes) for v in counts.index[counts == 1]})
                columns.append(values.map(codes).to_numpy(dtype=float, na_value=np.nan))
        X = np.column_stack(columns) if columns else np.zeros((len(X), 0))
    else:
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ValueError('X has non-numeric values; pass a pandas DataFrame to have them encoded') from None
    if X.ndim != 2:
        raise ValueError(f'X must be (n, p); got shape {X.shape}')
    X = np.where(np.isfinite(X), X, np.nan)
    missing = np.isnan(X)
    if missing.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            medians = np.nan_to_num(np.nanmedian(X, axis=0))
        X = np.where(missing, medians, X)
    return X.astype(np.float32), names

def response_vector(y, response='auto'):
    '''
    Response values and type: 'regression' (float values), 'binary' (0/1, with 1 for the last of the sorted classes),
    or 'multiclass' (codes 0, ..., K - 1 of the sorted classes). With response='auto', a non-numeric or boolean y (strings,
    categories) is binary for two classes and multiclass otherwise, and a numeric y is binary if it takes two values
    and regression otherwise (pass response='multiclass' for numeric class labels).
    '''
    if response not in ('auto', 'regression', 'binary', 'multiclass'):
        raise ValueError(f"response must be 'auto', 'regression', 'binary', or 'multiclass'; got {response!r}")
    values = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    values = values.ravel()
    try:
        numeric = values.dtype.kind in 'iuf' and not (hasattr(y, 'dtype') and str(y.dtype) == 'category')
        if numeric:
            values = values.astype(float)
            missing = ~np.isfinite(values)
        else:
            missing = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in values], dtype=bool)
    except TypeError:
        numeric, missing = False, np.zeros(len(values), dtype=bool)
    if missing.any():
        raise ValueError('y must not contain missing or infinite values')
    if response == 'auto':
        response = ('binary' if len(np.unique(values)) == 2 else 'regression') if numeric else 'classification'
    if response == 'regression':
        if not numeric:
            raise ValueError("response='regression' needs a numeric y")
        return values.astype(np.float32), 'regression'
    levels = sorted(set(values.tolist()), key=lambda v: (str(type(v)), v))
    if len(levels) < 2:
        raise ValueError('y has a single class')
    if response == 'classification':
        response = 'binary' if len(levels) == 2 else 'multiclass'
    if response == 'binary' and len(levels) != 2:
        raise ValueError(f"response='binary' needs two classes; y has {len(levels)}")
    codes = {v: i for i, v in enumerate(levels)}
    return np.array([codes[v] for v in values.tolist()], dtype=np.float32), response

def row_budget(p):
    '''
    Rows kept for p covariates: the largest n with n (n + p) p (the cost of the attention across rows and across
    covariates) at most that of 10,000 rows and 100 covariates, but never fewer than the 1,024 rows of training.
    '''
    budget = budget_rows * (budget_rows + budget_p) * budget_p
    p = max(p, 1)
    n = (-p * p + np.sqrt(p ** 4 + 4 * p * budget)) / (2 * p)
    return max(training_rows, int(n))

def subsample_rows(y, kind, m, rng):
    '''m random rows (sorted); for a classification response, each class in proportion to its count (at least one row).'''
    if kind == 'regression':
        return np.sort(rng.choice(len(y), m, replace=False))
    classes, counts = np.unique(y, return_counts=True)
    take = np.maximum(1, np.floor(m * counts / len(y)).astype(int))
    for i in np.argsort(-(m * counts / len(y) - take))[:max(0, m - take.sum())]:
        take[i] += 1
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), min(t, cnt), replace=False) for c, t, cnt in zip(classes, take, counts)]))

#----------------------------------------------------------------
# Models
#----------------------------------------------------------------
class PFN:
    '''
    Shared by CDPFN and PCPFN: input preparation, the row budget, the forward pass, and covariate names for pandas
    input.
    '''
    default_weights = None
    outputs = None
    balance = 'off'
    transform = 'off'
    multiclass = False

    def __init__(self, weights=None, device='cpu', response='auto', max_rows='auto', seed=0):
        self.weights = str(weights) if weights is not None else self.default_weights
        self.device = torch.device(device)
        self.response = response
        self.max_rows = max_rows
        self.seed = seed
        self.rows_used = None

    def probabilities(self, X, y):
        X, names = covariates_matrix(X)
        y, kind = response_vector(y, self.response)
        if len(X) != len(y):
            raise ValueError(f'X has {len(X)} rows and y {len(y)}')
        for option in ['balance', 'transform']:
            if getattr(self, option) not in ('auto', 'off'):
                raise ValueError(f"{option} must be 'auto' or 'off'; got {getattr(self, option)!r}")
        if kind == 'multiclass' and not self.multiclass:
            raise ValueError(f'{type(self).__name__} does not handle a multiclass response')
        rng = np.random.default_rng(self.seed)
        if self.max_rows is not None:
            m = row_budget(int((np.ptp(X, axis=0) > 0).sum())) if self.max_rows == 'auto' else int(self.max_rows)
            if len(y) > m:
                rows = subsample_rows(y, kind, m, rng)
                X, y = X[rows], y[rows]
        self.rows_used = 0
        if kind == 'multiclass':
            # mean one-vs-rest scores: one pass per class on the indicator of that class
            pips = np.mean([self.forward(X, (y == c).astype(np.float32), rng) for c in np.unique(y)], axis=0)
        else:
            pips = self.forward(X, y, rng)
        return pips, names

    def forward(self, X, y, rng):
        if self.balance == 'auto':
            sets = case_control_chunks(y, rng)
            if sets is not None:
                # mean PIPs over the case-control sets, which together use every row of the other class once (up to
                # balance_chunks sets)
                return np.mean([self.forward_once(X[rows], y[rows]) for rows in sets], axis=0)
        return self.forward_once(X, y)

    def forward_once(self, X, y):
        self.rows_used = max(self.rows_used, len(y))
        # constant covariates carry no information: their PIPs are 0, and they are left out of the forward pass so that
        # they do not affect the other covariates' PIPs
        varying = np.ptp(X, axis=0) > 0 if len(X) else np.zeros(X.shape[1], dtype=bool)
        pips = np.zeros((X.shape[1],) + np.shape(self.outputs))
        if varying.any():
            model = load_network(self.weights, self.device)
            Xv = np.ascontiguousarray(X[:, varying])
            if self.transform == 'auto':
                extreme = extreme_columns(Xv)
                if extreme.any():
                    Xv = rank_columns(Xv, extreme)
            with torch.no_grad():
                logits = model(torch.from_numpy(Xv)[None].to(self.device), torch.from_numpy(y)[None].to(self.device))[0]
            pips[varying] = torch.sigmoid(logits[:, self.outputs].float()).cpu().numpy().astype(float)
        return pips

    def warn_rows(self):
        if self.rows_used is not None and self.rows_used > training_rows:
            warnings.warn(f'the PIPs were computed from {self.rows_used:,} rows, beyond the {training_rows:,} of training: rankings improve with more rows, but in simulations the realized FDR of the Bayesian FDR rule reached 15-27% at alpha = 0.1 with 3,000-10,000 rows. Pass max_rows={training_rows} for PIPs from at most {training_rows:,} rows.', stacklevel=3)

class CDPFN(PFN):
    '''
    CD-PFN: for each covariate, the estimated posterior probability that y depends on it given the other covariates.

    X: (n, p) array or DataFrame; columns of any type are encoded (covariates_matrix: categories as codes, missing
    values by the median). y: (n,) continuous, binary, or multiclass response (response_vector). Trained on
    16 <= n <= 1024 and 8 <= p <= 512; columns are standardized inside the model. Constant covariates get PIP 0 and are
    left out of the forward pass.

    response: 'auto' (default), 'regression', 'binary', or 'multiclass'. For a multiclass y, the scores are the mean of
    the one-vs-rest PIPs (one pass per class on its indicator), not PIPs of y itself: they rank covariates well, and
    select() applies the usual cutoff to them, but its FDR behavior on them has not been established.

    max_rows: 'auto' (default), an integer, or None (all rows). With 'auto', datasets with more rows than row_budget(p)
    (10,000 rows for p <= 100, fewer for larger p, never under 1,024) are subsampled at random (stratified on a
    classification response) to bound the compute. More rows than training gives better rankings but overconfident
    PIPs (select() warns); max_rows=1024 keeps the input within the training range.

    balance: 'auto' (default) or 'off'. With 'auto', a binary response (or class indicator) whose rarer class has under
    20% of the rows (rarer than any in training) is case-control subsampled (case_control_chunks): the rows of the other
    class are split into disjoint chunks of about 3 times the rarer count (at least 200 rows with the rarer class), each
    chunk is passed with every row of the rarer class, and the PIPs are averaged over the passes (at most 10). Sampling
    on y alone leaves the conditional dependence of y on each covariate unchanged.

    transform: 'auto' (default) or 'off'. With 'auto', each covariate with extreme values (one more than 50 robust
    standard deviations from its median; extreme_columns) is replaced by its ranks, which removes heavy tails and
    outliers. A strictly increasing function of a covariate leaves the conditional dependence unchanged.

    seed: for the random subsamples.
    '''
    default_weights = 'cd_pfn'
    outputs = 0
    multiclass = True

    def __init__(self, weights=None, device='cpu', response='auto', max_rows='auto', balance='auto', transform='auto', seed=0):
        super().__init__(weights, device, response, max_rows, seed)
        self.balance = balance
        self.transform = transform

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
        self.warn_rows()
        selected = bayes_select(pips, alpha)
        return selected if names is None else [names[j] for j in selected]

class PCPFN(PFN):
    '''
    PC-PFN: for each covariate, the estimated posterior probabilities that it is a parent (direct cause) and a child
    (direct effect) of y. Inputs as for CDPFN, except that y is continuous or binary.
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
        self.warn_rows()
        out = {'parents': bayes_select(pips[:, 0], alpha), 'children': bayes_select(pips[:, 1], alpha)}
        return out if names is None else {role: [names[j] for j in selected] for role, selected in out.items()}

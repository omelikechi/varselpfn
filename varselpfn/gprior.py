# Linear g-prior with exact PIPs
'''
A prior whose posterior inclusion probabilities can be computed exactly, for checking how closely a network
approximates the posterior, and an example of fine-tuning on a new prior (config "family": "gprior"). Covariates are
Gaussian with factor-model correlation. The model is y = alpha + X beta + sigma eps, with a beta-binomial(1, 1) prior on
the active set gamma, Zellner's g-prior beta_gamma | g, sigma ~ N(0, g sigma^2 (Xc_gamma' Xc_gamma)^{-1}) on centered
covariates Xc, g = c n with c uniform on a log-spaced grid, a flat prior on alpha, and a 1/sigma^2 prior on sigma^2.
Under the last two, the posterior of gamma is invariant to the location and scale of y and of each covariate, so it can
be computed from the standardized data the network sees; data are therefore simulated with alpha = 0 and sigma = 1.
Exact PIPs enumerate all 2^p models, using the marginal likelihood of Liang et al. (2008),
p(y | gamma, g) propto (1 + g)^{(n - 1 - k) / 2} (1 + g (1 - R^2_gamma))^{-(n - 1) / 2}.
Labels have the three columns of prior.py (conditional dependence, parent, child): gamma, gamma, and zero.
'''

import itertools

import numpy as np
from scipy.special import logsumexp

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
c_grid = np.geomspace(0.01, 1.0, 9)

#----------------------------------------------------------------
# Sampling
#----------------------------------------------------------------
def log_uniform_int(rng, low, high):
    return int(np.round(np.exp(rng.uniform(np.log(low), np.log(high)))))

def sample_n(rng, prior):
    return log_uniform_int(rng, *prior['n_range'])

def sample_p(rng, prior):
    return int(rng.integers(prior['p_range'][0], prior['p_range'][1] + 1))

def sample_covariates(rng, n, p):
    '''Gaussian covariates with 0 to 3 common factors; loadings vary, so correlations range from none to strong.'''
    k = rng.integers(0, 4)
    loadings = rng.normal(0, rng.uniform(0.3, 1.5), size=(p, k)) * (rng.random((p, k)) < 0.7)
    return rng.normal(size=(n, k)) @ loadings.T + rng.normal(size=(n, p))

def sample_dataset(rng, prior, n=None, p=None, keep_info=False):
    '''Returns X (n, p), y (n,), labels (p, 3), failures (always 0), and info (gamma and g, or None).'''
    n = sample_n(rng, prior) if n is None else n
    p = sample_p(rng, prior) if p is None else p
    X = sample_covariates(rng, n, p)
    gamma = rng.random(p) < rng.beta(1, 1)
    g = rng.choice(c_grid) * n
    y = rng.normal(size=n)
    if gamma.any():
        Xc = X[:, gamma] - X[:, gamma].mean(axis=0)
        R = np.linalg.qr(Xc, mode='r')
        beta = np.sqrt(g) * np.linalg.solve(R, rng.normal(size=gamma.sum()))
        y = y + Xc @ beta
    labels = np.column_stack([gamma, gamma, np.zeros(p, bool)]).astype(np.float32)
    info = {'gamma': gamma, 'g': g} if keep_info else None
    return X.astype(np.float32), y.astype(np.float32), labels, 0, info

def sample_batch(rng, prior, batch_size, n=None, p=None, keep_info=False):
    '''All datasets in a batch share n and p, as in prior.py.'''
    n = sample_n(rng, prior) if n is None else n
    p = sample_p(rng, prior) if p is None else p
    draws = [sample_dataset(rng, prior, n=n, p=p, keep_info=keep_info) for _ in range(batch_size)]
    return (np.stack([d[0] for d in draws]), np.stack([d[1] for d in draws]), np.stack([d[2] for d in draws]), 0,
        [d[4] for d in draws])

#----------------------------------------------------------------
# Exact posterior
#----------------------------------------------------------------
def exact_pips(X, y):
    '''Exact PIPs by enumerating all 2^p models, averaging the marginal likelihood over the grid of g.'''
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    A = Xc.T @ Xc
    b = Xc.T @ yc
    yy = yc @ yc
    models = np.array(list(itertools.product([0, 1], repeat=p)), dtype=bool)
    log_post = np.empty(len(models))
    g = c_grid * n
    for i, gamma in enumerate(models):
        k = gamma.sum()
        r2 = b[gamma] @ np.linalg.solve(A[np.ix_(gamma, gamma)], b[gamma]) / yy if k else 0.0
        log_marginal = logsumexp((n - 1 - k) / 2 * np.log1p(g) - (n - 1) / 2 * np.log1p(g * (1 - r2))) - np.log(len(g))
        # beta-binomial(1, 1) prior: 1 / ((p + 1) C(p, k))
        log_prior = -np.log(p + 1) - (np.sum(np.log(np.arange(1, p + 1))) - np.sum(np.log(np.arange(1, k + 1))) - np.sum(np.log(np.arange(1, p - k + 1))))
        log_post[i] = log_marginal + log_prior
    weights = np.exp(log_post - logsumexp(log_post))
    return weights @ models

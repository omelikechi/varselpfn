# Training prior
'''
Broad prior over data-generating models, used to simulate datasets labeled with conditional dependence, parent, and
child membership of each observed predictor (covariate) relative to an observed response. The prior is a mixture of two families
(component_weights):

- scm: a partially observed DAG over the response, predictors, and latent nodes (described below).
- xfirst: predictors are drawn first from a structure family (independent, latent-factor modules, global-plus-module
  factors, AR, or an SCM among the predictors), with optional monotone marginal transforms and discretization; then
  Y = f(X_S, noise) with a sparse random S; optionally a few predictors are regenerated after Y as its children (with
  other predictors as spouses). Predictor correlation is therefore drawn independently of the response's
  neighborhood. Labels: conditional dependence = parents of Y, children, and other parents of children; exact under positivity and
  faithfulness, whatever the dependence among predictors, because the conditional density of Y given X involves only
  these terms.

With scm_hubs in the prior, a share of scm datasets also gets latent hubs (global factors plus modules, as in the
X-first hierarchical structure) driving many predictors and latent nodes but never the response, so predictors are
correlated as in omics data while children of the response stay embedded in that correlation. Hubs are latent nodes
of the graph, so the labels account for them.

Convention: A[u, v] = 1 means u -> v.

Graph: p observed predictors, one response, and a random number of latent nodes. Edges follow a random topological
order, either Erdos-Renyi or with heterogeneous Cauchy-distributed node propensities (hubs), the latter adapted from
TabICL's RandomCauchyDAG (https://github.com/soda-inria/tabicl, BSD-3-Clause, commit c91f00d).

In the scm family, mechanisms (one family per node): additive edge functions (identity, tanh, sin, cos, square, abs,
Gaussian bump, relu, softplus, sigmoid, each with a random input scale and shift), small random MLPs over the parent
vector, random Fourier-feature (GP-like) functions, products of two parent functions (interactions; for the X-first
response, up to xfirst max_product_terms products of disjoint pairs, plus additive terms in the remaining parents),
linear maps, and additive random trees (piecewise constant; 'tree'). Step functions are also an additive edge function
when additive_weights includes them.

Noise is additive with a random noise fraction r: z = sqrt(1 - r) f + sqrt(r) eps, where f and eps are standardized.
Noise families: Gaussian, Laplace, uniform, Student-t (random degrees of freedom), two-component Gaussian mixture.
Every node is standardized during generation (internally standardized SCM, Ormaniec et al. 2025), with moments taken
from an independent reference sample of n_ref rows.

Observed predictors are continuous, binary, or ordinal; the response is continuous or binary. A discrete variable
is usually causal as observed (it is discretized during generation, and its children see the discrete value). With
probability measurement_discretization_prob it is instead a coarse measurement of a continuous node: the observed
variable is a child of the continuous node, which stays in the graph as a latent node.

Every edge is functionally active: after a mechanism is drawn, each parent is replaced in turn by a shuffled copy
(reference rows, same random parameters); if the output does not change, the mechanism is redrawn, and after 20
failed draws the inactive edges are removed from the graph before labels are computed. This enforces the part of
faithfulness the simulator controls (no edge without dependence through its mechanism); cancellations along
multiple paths are not checked.

Optional (off unless set in the prior): scm_y_neighbors gives, with probability prob, the response of an scm dataset
numbers of parents and children drawn uniformly from 0 to max_parents and max_children (each at most max_fraction of p
when set), independently of the graph's density (the response is placed in the order with room for them, and its edges
are redrawn); otherwise the response is a random node and inherits the graph's sparsity. linear_gaussian_prob draws that share of all datasets from the
scm family with the overrides in linear_gaussian (for example linear mechanisms and Gaussian noise only).

Labels for predictor j:
- conditional dependence: the observed response is not d-separated from observed X_j given all other observed
  predictors, in the graph including latent and discretization nodes (equals Markov-blanket membership when everything
  is observed and continuous). Assumes faithfulness.
- parent / child: a direct edge from j's underlying node to the response's underlying node, or the reverse.
'''

import copy

import numpy as np
from scipy.sparse.csgraph import connected_components

additive_functions = {
    'identity': lambda x: x,
    'tanh': np.tanh,
    'sin': np.sin,
    'cos': np.cos,
    'square': lambda x: x**2,
    'abs': np.abs,
    'bump': lambda x: np.exp(-x**2 / 2),
    'relu': lambda x: np.maximum(x, 0),
    'softplus': lambda x: np.logaddexp(0, x),
    'sigmoid': lambda x: 0.5 * (1 + np.tanh(x / 2)),
    'step': lambda x: (x > 0).astype(float),
}
mlp_activations = ['tanh', 'relu', 'sin', 'bump', 'softplus', 'abs']

#----------------------------------------------------------------
# Graph
#----------------------------------------------------------------
def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def log_uniform_int(rng, low, high):
    return int(np.clip(np.rint(np.exp(rng.uniform(np.log(low - 0.5), np.log(high + 0.5)))), low, high))

def sample_n(rng, prior):
    '''Sample size: uniform on n_range with probability n_uniform_prob, otherwise log-uniform.'''
    if rng.random() < prior.get('n_uniform_prob', 0.0):
        return int(rng.integers(prior['n_range'][0], prior['n_range'][1] + 1))
    return log_uniform_int(rng, *prior['n_range'])

def sample_p(rng, prior):
    '''
    Number of predictors: with probability p_large_prob (if set), uniform on p_large_range; otherwise uniform on p_range
    with probability p_uniform_prob, else log-uniform on p_range.
    '''
    if 'p_large_prob' in prior and rng.random() < prior['p_large_prob']:
        return int(rng.integers(prior['p_large_range'][0], prior['p_large_range'][1] + 1))
    if rng.random() < prior.get('p_uniform_prob', 0.0):
        return int(rng.integers(prior['p_range'][0], prior['p_range'][1] + 1))
    return log_uniform_int(rng, *prior['p_range'])

def sample_graph(m, rng, prior):
    '''Random DAG over m nodes with expected in-degree about lambda (log-uniform), ER or Cauchy-heterogeneous.'''
    lam = log_uniform(rng, *prior['indegree_range'])
    target = min(1.0, 2 * lam / max(m - 1, 1))
    upper_mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    if rng.random() < prior['cauchy_graph_prob']:
        # exp overflow for extreme Cauchy logits gives probability 0 or 1, which is intended
        np.seterr(over='ignore')
        logits = rng.standard_cauchy(m)[:, None] + rng.standard_cauchy(m)[None, :]
        # shift logits so the mean edge probability over forward pairs matches the target density
        lo, hi = -50.0, 50.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(logits[upper_mask] + mid)))).mean() < target:
                lo = mid
            else:
                hi = mid
        prob = 1 / (1 + np.exp(-(logits + mid)))
    else:
        prob = np.full((m, m), target)
    upper = (rng.random((m, m)) < prob) & upper_mask
    if 'max_in_degree' in prior:
        # at most max_in_degree parents per node (random ones kept): Cauchy propensities otherwise create colliders
        # with hundreds of parents, whose co-parents would all be spouses of the response
        for v in np.flatnonzero(upper.sum(axis=0) > prior['max_in_degree']):
            drop = rng.choice(np.flatnonzero(upper[:, v]), size=int(upper[:, v].sum()) - prior['max_in_degree'], replace=False)
            upper[drop, v] = False
    order = rng.permutation(m)
    A = np.zeros((m, m), dtype=np.int8)
    A[np.ix_(order, order)] = upper
    return A, order, lam

#----------------------------------------------------------------
# Mechanisms and noise
#----------------------------------------------------------------
def draw_noise(rng, family, size):
    if family == 'gaussian':
        return rng.standard_normal(size)
    if family == 'laplace':
        return rng.laplace(size=size)
    if family == 'uniform':
        return rng.uniform(-1, 1, size)
    if family == 'student_t':
        return rng.standard_t(rng.uniform(2.5, 10), size)
    if family == 'mixture':
        shift = rng.uniform(1, 4)
        return rng.standard_normal(size) + shift * (rng.random(size) < rng.uniform(0.1, 0.5))
    raise ValueError(f'unknown noise family: {family}')

def additive_part(rng, P, function_set, function_weights=None):
    '''
    Sum over parent columns of w * g(s * x + b), scaled by 1/sqrt(k). function_weights (a dict over function names)
    sets the probabilities of the edge functions; without it they are uniform over function_set.
    '''
    k = P.shape[1]
    if function_weights:
        function_set = list(function_weights)
        probs = np.array([function_weights[g] for g in function_set], dtype=float)
        probs /= probs.sum()
    out = np.zeros(P.shape[0])
    for u in range(k):
        g = function_set[rng.choice(len(function_set), p=probs)] if function_weights else function_set[rng.integers(len(function_set))]
        scale = log_uniform(rng, 0.5, 2.0)
        shift = 0.0 if g == 'identity' else rng.uniform(-1, 1)
        weight = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        out += weight * additive_functions[g](scale * P[:, u] + shift)
    return out / np.sqrt(k)

def random_tree_part(rng, P):
    '''Sum over parents of a random piecewise-constant function with 2-8 levels (the 'tree' family).'''
    out = np.zeros(P.shape[0])
    for u in range(P.shape[1]):
        cuts = np.sort(rng.uniform(-1.5, 1.5, size=rng.integers(1, 8)))
        values = rng.standard_normal(len(cuts) + 1)
        out += values[np.searchsorted(cuts, P[:, u])]
    return out

def mechanism(rng, family, P, prior):
    '''f(parents) for one node; P is (rows, k) of standardized parent values.'''
    k = P.shape[1]
    if family == 'product' and k < 2:
        family = 'additive'
    if family == 'linear':
        return P @ (rng.choice([-1, 1], size=k) * rng.uniform(0.5, 1.5, size=k)) / np.sqrt(k)
    if family == 'additive':
        return additive_part(rng, P, prior['additive_set'], prior.get('additive_weights'))
    if family == 'tree':
        return random_tree_part(rng, P)
    if family == 'mlp':
        h = P
        for _ in range(rng.integers(1, 3)):
            width = int(rng.integers(8, 33))
            W = rng.standard_normal((h.shape[1], width)) / np.sqrt(h.shape[1])
            activations = prior.get('mlp_activations', mlp_activations)
            act = activations[rng.integers(len(activations))]
            h = additive_functions[act](h @ W + 0.5 * rng.standard_normal(width))
        return h @ rng.standard_normal(h.shape[1]) / np.sqrt(h.shape[1])
    if family == 'gp':
        n_features = 64
        length = log_uniform(rng, *prior.get('gp_length_range', [0.3, 3.0]))
        omega = rng.standard_normal((k, n_features)) / length
        phase = rng.uniform(0, 2 * np.pi, n_features)
        return np.cos(P @ omega + phase) @ rng.standard_normal(n_features) / np.sqrt(n_features)
    if family == 'product':
        idx = rng.permutation(k)
        functions = prior.get('product_functions', ['identity', 'tanh', 'sin', 'sigmoid', 'bump'])
        max_terms = min(prior.get('max_product_terms', 1), k // 2)
        terms = 1 if max_terms == 1 else int(rng.integers(1, max_terms + 1))
        out = 0
        for t in range(terms):
            pair = P[:, idx[2 * t:2 * t + 2]]
            g1, g2 = (additive_functions[rng.choice(functions)] for _ in range(2))
            out = out + g1(pair[:, 0] + rng.uniform(-1, 1)) * g2(pair[:, 1] + rng.uniform(-1, 1))
        if k > 2 * terms:
            out = out + additive_part(rng, P[:, idx[2 * terms:]], prior['additive_set'], prior.get('additive_weights'))
        return out
    raise ValueError(f'unknown mechanism family: {family}')

def inactive_parents(state, family, P, f, n, prior):
    '''Indices of parent columns whose shuffling (same mechanism parameters) leaves the reference-row output unchanged.'''
    shuffle = np.random.default_rng(0)
    scale = np.std(f[n:]) + 1e-12
    dead = []
    for a in range(P.shape[1]):
        P2 = P.copy()
        P2[:, a] = shuffle.permutation(P2[:, a])
        f2 = mechanism(copy.deepcopy(state), family, P2, prior)
        if not np.std(f[n:] - f2[n:]) > 1e-6 * scale:
            dead.append(a)
    return dead

def standardize_by_reference(z, n):
    ref = z[n:]
    sd = ref.std()
    if not np.isfinite(sd) or sd < 1e-8:
        return None
    return (z - ref.mean()) / sd

#----------------------------------------------------------------
# Simulation
#----------------------------------------------------------------
def sample_hubs(rng, nodes, n, prior):
    '''
    Latent hubs for the scm family (scm_hubs in the prior): a few global factors shared by most hub-driven nodes plus
    modules, as in the X-first hierarchical structure. A share of the given nodes (never the response) is hub-driven;
    node v's hub signal is sqrt(g_v) (global loadings) . F_global + sqrt(1 - g_v) F_module(v), and simulate mixes it in
    with weight w_v. Returns the hub values (rows, hubs), the hub -> node edges as (hub index, node) pairs, and
    node -> (signal, w).
    '''
    hp = prior['scm_hubs']
    rows = n + prior['n_ref']
    share = rng.uniform(*hp['node_share_range'])
    driven = [int(v) for v in nodes if rng.random() < share]
    k_global = int(rng.integers(1, 4))
    n_modules = log_uniform_int(rng, 2, max(2, len(nodes) // 3))
    noise_family = prior['noise_families'][rng.integers(len(prior['noise_families']))]
    H = np.column_stack([standardize_by_reference(draw_noise(rng, noise_family, rows), n) for _ in range(k_global + n_modules)])
    g_mean = rng.uniform(0.3, 0.9)
    w_mean = rng.uniform(*hp['weight_range'])
    edges, signals = [], {}
    for v in driven:
        loadings = rng.standard_normal(k_global)
        loadings /= np.linalg.norm(loadings)
        module = k_global + int(rng.integers(n_modules))
        g = rng.beta(4 * g_mean + 0.05, 4 * (1 - g_mean) + 0.05)
        # w at least 0.05, so every hub edge has a real effect (faithfulness)
        w = float(np.clip(rng.beta(4 * w_mean + 0.05, 4 * (1 - w_mean) + 0.05), 0.05, 0.97))
        # errstate: Accelerate BLAS on macOS raises spurious floating-point warnings in matmul
        with np.errstate(all='ignore'):
            signal = np.sqrt(g) * (H[:, :k_global] @ loadings) + np.sqrt(1 - g) * H[:, module]
        signals[v] = (signal, w)
        edges += [(h, v) for h in range(k_global)] + [(module, v)]
    return H, edges, signals

def simulate(A, order, n, rng, prior, causal_levels, hub_signals=None):
    '''
    Returns Z and Z_ref (internally standardized node values for the data and reference rows), discrete codes for
    causally discretized nodes, per-node mechanism families, the failure count, the number of mechanism redraws for
    inactive edges, and the number of edges pruned (A is modified in place when edges are pruned). causal_levels maps
    node -> number of levels for nodes discretized during generation; their children see the standardized discrete
    value. hub_signals maps node -> (signal, w): the node becomes sqrt(1 - w) z + sqrt(w) signal (restandardized),
    where z is its own mechanism plus noise (see sample_hubs).
    '''
    m = A.shape[0]
    codes = {}
    redraws = 0
    pruned = 0
    rows = n + prior['n_ref']
    families = list(prior['mechanism_weights'])
    weights = np.array([prior['mechanism_weights'][f] for f in families], dtype=float)
    weights /= weights.sum()
    Z = np.zeros((rows, m))
    node_family = [''] * m
    failures = 0
    for v in order:
        parents = np.flatnonzero(A[:, v])
        noise_family = prior['noise_families'][rng.integers(len(prior['noise_families']))]
        for attempt in range(20):
            eps = standardize_by_reference(draw_noise(rng, noise_family, rows), n)
            if len(parents) == 0:
                z = eps
                node_family[v] = 'root'
            else:
                family = families[rng.choice(len(families), p=weights)]
                # non-finite values are caught below and counted as failures; this only silences spurious warnings
                with np.errstate(all='ignore'):
                    state = copy.deepcopy(rng)
                    f_raw = mechanism(rng, family, Z[:, parents], prior)
                    dead = inactive_parents(state, family, Z[:, parents], f_raw, n, prior) if np.isfinite(f_raw).all() else []
                    f = standardize_by_reference(f_raw, n)
                if dead and attempt < 19:
                    redraws += 1
                    continue
                if dead:
                    # last resort: remove edges the mechanism ignores, so labels match the data-generating process
                    A[parents[dead], v] = 0
                    pruned += len(dead)
                r = rng.uniform(*prior['noise_fraction_range'])
                z = None if f is None or eps is None else np.sqrt(1 - r) * f + np.sqrt(r) * eps
                node_family[v] = family
            z = None if z is None else standardize_by_reference(z, n)
            if z is not None and np.isfinite(z).all():
                break
            failures += 1
        if hub_signals is not None and v in hub_signals and z is not None:
            signal, w = hub_signals[v]
            mixed = standardize_by_reference(np.sqrt(1 - w) * z + np.sqrt(w) * signal, n)
            z = z if mixed is None else mixed
        if v in causal_levels:
            d = discretize(z, z[n:], rng, causal_levels[v])
            codes[v] = d[:n]
            # a constant discrete node (all reference rows in one level) carries no information to children
            z = standardize_by_reference(d, n)
            z = np.zeros(rows) if z is None else z
        Z[:, v] = z
    return Z[:n], Z[n:], codes, node_family, failures, redraws, pruned

def discretize(values, reference, rng, levels, rare=False):
    '''
    Ordinal coding at random reference quantiles: binary if levels == 2. With rare, a binary cut is at a
    logit-uniform quantile (about 0.02 to 0.98), so one level can be rare.
    '''
    if levels == 2 and rare:
        qs = np.array([1 / (1 + np.exp(-rng.uniform(-4, 4)))])
    else:
        qs = np.sort(rng.uniform(0.1, 0.9, size=levels - 1)) if levels > 2 else np.array([rng.uniform(0.2, 0.8)])
    cuts = np.quantile(reference, qs)
    return np.searchsorted(cuts, values).astype(float)

#----------------------------------------------------------------
# Labels
#----------------------------------------------------------------
def conditional_dependence(A, observed, y, predictors):
    '''
    For each predictor j: True unless y and j are d-separated given all other observed nodes.
    Uses the moral graph of the ancestral set of the observed nodes: after removing the conditioning set, only
    y, j, and latent nodes remain, so y and j are connected iff they are adjacent or both touch the same connected
    component of latent nodes.
    '''
    M = A.shape[0]
    A = A.astype(bool)
    anc = observed.copy()
    while True:
        new = anc | A[:, anc].any(axis=1)
        if (new == anc).all():
            break
        anc = new
    sub = np.flatnonzero(anc)
    index = -np.ones(M, dtype=int)
    index[sub] = np.arange(len(sub))
    As = A[np.ix_(sub, sub)].astype(np.int32)
    moral = (As + As.T + As @ As.T) > 0
    np.fill_diagonal(moral, False)
    latent = ~observed[sub]
    lat_idx = np.flatnonzero(latent)
    yi = index[y]
    pj = index[predictors]
    direct = moral[yi, pj]
    if len(lat_idx) == 0:
        return direct
    n_comp, comp = connected_components(moral[np.ix_(lat_idx, lat_idx)], directed=False)
    onehot = np.zeros((len(lat_idx), n_comp), dtype=bool)
    onehot[np.arange(len(lat_idx)), comp] = True
    touches = (moral[:, lat_idx].astype(np.int32) @ onehot.astype(np.int32)) > 0
    via_latent = (touches[pj] & touches[yi][None, :]).any(axis=1)
    return direct | via_latent

#----------------------------------------------------------------
# X-first family
#----------------------------------------------------------------
marginal_transforms = {
    # right or left skew of varying strength
    'exp': lambda x, rng: np.exp(rng.choice([-1, 1], p=[0.2, 0.8]) * log_uniform(rng, 0.2, 1.5) * x),
    # many zeros and a skewed positive part, like counts or amounts
    'zero_inflated': lambda x, rng: np.exp(np.maximum(x - rng.uniform(0, 2), 0) * log_uniform(rng, 0.2, 1.5)) - 1,
    'cube': lambda x, rng: x**3,
    'sqrt': lambda x, rng: np.sign(x) * np.sqrt(np.abs(x)),
    'uniform': lambda x, rng: 0.5 * (1 + np.tanh(x * 0.8)),
    'bimodal': lambda x, rng: np.tanh(rng.uniform(2, 5) * (x - rng.uniform(-0.5, 0.5))),
}

def sample_predictors(rng, n, p, prior, graph=False):
    '''
    Predictors for the X-first family over n data rows plus n_ref reference rows. Returns recorded values X_obs,
    internally standardized values X_par (what mechanisms see), the feature types (0 continuous, 1 binary, 2 ordinal),
    and the structure kind. With graph, also returns the generating structure: the number L of latent structure nodes
    (factors, modules, or unobserved SCM nodes) and the edges (u, v) among nodes 0..L-1 (latent) and L + j (the
    continuous value of predictor j before any discretization). Recording it draws no random numbers.
    '''
    xp = prior['xfirst']
    rows = n + prior['n_ref']
    kinds = list(xp['structure_weights'])
    weights = np.array([xp['structure_weights'][k] for k in kinds], dtype=float)
    kind = kinds[rng.choice(len(kinds), p=weights / weights.sum())]
    n_struct, edges = 0, []
    if kind == 'independent':
        X = rng.standard_normal((rows, p))
    elif kind == 'factor':
        # x_j = h_j * (direction of its loadings) . F + sqrt(1 - h_j^2) e_j, so corr(x_i, x_j) = h_i h_j cos(angle);
        # the communalities h_j^2 have dataset-level mean s ~ U(0, 0.95), so mean |corr| spans weak to near-collinear
        k = log_uniform_int(rng, 1, max(1, p // 2))
        if rng.random() < 0.3:
            # clusters: each predictor loads on exactly one factor
            loadings = np.zeros((p, k))
            loadings[np.arange(p), rng.integers(k, size=p)] = 1.0
        else:
            loadings = rng.standard_normal((p, k)) * (rng.random((p, k)) < rng.uniform(0.2, 1.0))
            empty = ~loadings.any(axis=1)
            loadings[empty, rng.integers(k, size=empty.sum())] = 1.0
        loadings /= np.linalg.norm(loadings, axis=1, keepdims=True)
        s = rng.uniform(0, 0.95)
        h2 = rng.beta(4 * s + 0.05, 4 * (1 - s) + 0.05, size=p)
        noise_family = prior['noise_families'][rng.integers(len(prior['noise_families']))]
        F = np.column_stack([standardize_by_reference(draw_noise(rng, noise_family, rows), n) for _ in range(k)])
        E = np.column_stack([standardize_by_reference(draw_noise(rng, noise_family, rows), n) for _ in range(p)])
        # errstate: Accelerate BLAS on macOS raises spurious floating-point warnings in matmul
        with np.errstate(all='ignore'):
            X = np.sqrt(h2) * (F @ loadings.T) + np.sqrt(1 - h2) * E
        n_struct = k
        edges = [(f, k + j) for j, f in zip(*np.nonzero(loadings))]
    elif kind == 'hierarchical':
        # omics-like: a few global factors shared by most predictors plus modules (clusters). The common part of x_j is
        # sqrt(g_j) (global loadings) . F_global + sqrt(1 - g_j) F_module(j), with global weights g_j around a dataset-level
        # mean; communalities are capped at 0.97, so no two predictors are exactly collinear
        k_global = int(rng.integers(1, 4))
        n_modules = log_uniform_int(rng, 2, max(2, p // 3))
        global_loadings = rng.standard_normal((p, k_global))
        global_loadings /= np.linalg.norm(global_loadings, axis=1, keepdims=True)
        g_mean = rng.uniform(0.3, 0.9)
        g = rng.beta(4 * g_mean + 0.05, 4 * (1 - g_mean) + 0.05, size=p)
        s = rng.uniform(0.4, 0.95)
        h2 = np.minimum(rng.beta(4 * s + 0.05, 4 * (1 - s) + 0.05, size=p), 0.97)
        noise_family = prior['noise_families'][rng.integers(len(prior['noise_families']))]
        F_global = np.column_stack([standardize_by_reference(draw_noise(rng, noise_family, rows), n) for _ in range(k_global)])
        F_module = np.column_stack([standardize_by_reference(draw_noise(rng, noise_family, rows), n) for _ in range(n_modules)])
        E = np.column_stack([standardize_by_reference(draw_noise(rng, noise_family, rows), n) for _ in range(p)])
        module = rng.integers(n_modules, size=p)
        with np.errstate(all='ignore'):
            common = np.sqrt(g) * (F_global @ global_loadings.T) + np.sqrt(1 - g) * F_module[:, module]
            X = np.sqrt(h2) * common + np.sqrt(1 - h2) * E
        n_struct = k_global + n_modules
        edges = [(h, n_struct + j) for j in range(p) for h in range(k_global)] + [(k_global + module[j], n_struct + j) for j in range(p)]
    elif kind == 'ar':
        rho = rng.uniform(0, 0.95)
        X = np.empty((rows, p))
        X[:, 0] = rng.standard_normal(rows)
        for j in range(1, p):
            X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(rows)
        perm = rng.permutation(p)
        X = X[:, perm]
        # column c holds chain position perm[c]
        column = np.argsort(perm)
        edges = [(int(column[j - 1]), int(column[j])) for j in range(1, p)]
    else:
        # an SCM among the predictors (and latent nodes), using the scm family's graph and mechanisms
        n_latent = 0 if rng.random() < prior['latent_zero_prob'] else int(round(p * rng.uniform(*prior['latent_fraction_range'])))
        A, order, _ = sample_graph(p + n_latent, rng, prior)
        Z, Z_ref, _, _, _, _, _ = simulate(A, order, n, rng, prior, {})
        nodes = rng.permutation(p + n_latent)
        X = np.vstack([Z, Z_ref])[:, nodes[:p]]
        # graph node v -> structure index: observed nodes after the latent ones
        index = np.empty(p + n_latent, dtype=int)
        index[nodes[:p]] = n_latent + np.arange(p)
        index[nodes[p:]] = np.arange(n_latent)
        n_struct = n_latent
        edges = [(int(index[u]), int(index[v])) for u, v in zip(*np.nonzero(A))]

    X_obs = np.zeros((rows, p))
    X_par = np.zeros((rows, p))
    feature_type = np.zeros(p, dtype=int)
    names = list(xp['transform_weights'])
    # per-dataset transform probabilities around the prior weights, so some datasets are dominated by one kind
    transform_p = rng.dirichlet(np.array([xp['transform_weights'][t] for t in names], dtype=float) / sum(xp['transform_weights'].values()) * len(names))
    # dataset-level share of discrete predictors: none, or uniform on (0, 1); half binary, half ordinal
    u = rng.random()
    discrete_share = 0.0 if u < xp['continuous_prob'] else 1.0 if u < xp['continuous_prob'] + xp['all_discrete_prob'] else rng.uniform(0, 1)
    # dataset-level share of marginally transformed (skewed, heavy-tailed, bounded or bimodal) predictors
    u = rng.random()
    transform_share = 0.0 if u > xp['transform_prob'] else 1.0 if u < xp['all_transform_prob'] else rng.uniform(0, 1)
    replaced = []
    for j in range(p):
        x = standardize_by_reference(X[:, j], n)
        if x is None:
            # a degenerate column is replaced by independent noise, so it loses its structure edges
            replaced.append(n_struct + j)
            x = rng.standard_normal(rows)
        if rng.random() < transform_share:
            with np.errstate(all='ignore'):
                # input clipped at 6 sd so heavy-tailed noise under exp or cube does not give absurd outliers
                t = standardize_by_reference(marginal_transforms[names[rng.choice(len(names), p=transform_p)]](np.clip(x, -6, 6), rng), n)
            x = x if t is None or not np.isfinite(t).all() else t
        u = rng.random()
        if u < discrete_share / 2:
            feature_type[j], levels = 1, 2
        elif u < discrete_share:
            feature_type[j], levels = 2, int(rng.integers(prior['ordinal_levels'][0], prior['ordinal_levels'][1] + 1))
        else:
            levels = 0
        if levels:
            codes = discretize(x, x[n:], rng, levels, rare=True)
            z = standardize_by_reference(codes, n)
            X_obs[:, j], X_par[:, j] = codes, np.zeros(rows) if z is None else z
        else:
            X_obs[:, j], X_par[:, j] = x, x
    if graph:
        return X_obs, X_par, feature_type, kind, (n_struct, [(u, v) for u, v in edges if v not in replaced])
    return X_obs, X_par, feature_type, kind

def generate_node(rng, P, n, prior, background=None, background_weight=0.0):
    '''
    One internally standardized node with parent values P (rows, k), every parent functionally active (see
    inactive_parents). Returns None if no valid mechanism is found in 20 draws. With background (rows,), the noise is
    sqrt(1 - background_weight) eps + sqrt(background_weight) background (restandardized), so background acts as a
    further, latent parent that enters through the noise term and leaves the mechanism's weight unchanged.
    '''
    families = list(prior['mechanism_weights'])
    weights = np.array([prior['mechanism_weights'][f] for f in families], dtype=float)
    weights /= weights.sum()
    noise_family = prior['noise_families'][rng.integers(len(prior['noise_families']))]
    for _ in range(20):
        eps = standardize_by_reference(draw_noise(rng, noise_family, P.shape[0]), n)
        if background is not None and eps is not None:
            eps = standardize_by_reference(np.sqrt(1 - background_weight) * eps + np.sqrt(background_weight) * background, n)
        family = families[rng.choice(len(families), p=weights)]
        with np.errstate(all='ignore'):
            state = copy.deepcopy(rng)
            f_raw = mechanism(rng, family, P, prior)
            dead = inactive_parents(state, family, P, f_raw, n, prior) if np.isfinite(f_raw).all() else [0]
            f = standardize_by_reference(f_raw, n)
        if dead or f is None or eps is None:
            continue
        r = rng.uniform(*prior['noise_fraction_range'])
        z = standardize_by_reference(np.sqrt(1 - r) * f + np.sqrt(r) * eps, n)
        if z is not None and np.isfinite(z).all():
            return z
    return None

def sample_dataset_xfirst(rng, prior, n, p, keep_info=False):
    '''
    One dataset from the X-first family; same outputs as sample_dataset. With xfirst child_keep_range, each child of y
    is generated from y and its spouses with noise sqrt(1 - w) eps + sqrt(w) (its own earlier value), w uniform on the
    range, so it keeps part of its place in the predictor correlation structure while its mechanism keeps the usual
    weight. The earlier value is then a latent parent of the child, correlated with other predictors, so conditioning on
    the child can make y depend on further predictors: the parent and child labels stay exact, but the label (parents,
    children, spouses) is only a subset of the conditional dependence set, and training with this option requires
    head_weights[0] = 0. With keep_info, info['graph'] holds the generating graph (adjacency A over latent structure
    nodes, earlier and final predictor values, discretization nodes, and the response; the observed nodes; the response
    node; and each output column's node), and info['blanket_exact'] says whether the conditional dependence label is
    exact.
    '''
    xp = prior['xfirst']
    rows = n + prior['n_ref']
    failures = 0
    while True:
        if keep_info:
            X_obs, X_par, feature_type, kind, (n_struct, edges) = sample_predictors(rng, n, p, prior, graph=True)
            # graph bookkeeping only (no random numbers): node of each predictor's current value
            n_nodes = n_struct + p
            obs_node = list(range(n_struct, n_struct + p))
            for j in np.flatnonzero(feature_type):
                edges.append((obs_node[j], n_nodes))
                obs_node[j] = n_nodes
                n_nodes += 1
        else:
            X_obs, X_par, feature_type, kind = sample_predictors(rng, n, p, prior)
        k_max = max(1, int(p * xp['max_parent_fraction']))
        # neighborhood size: none (null_prob), else uniform on 1..k_max with probability parent_uniform_prob (dense
        # supports), otherwise log-uniform (mostly sparse); configs without parent_uniform_prob draw as before
        if rng.random() < xp['null_prob']:
            k = 0
        elif xp.get('parent_uniform_prob', 0.0) > 0 and rng.random() < xp['parent_uniform_prob']:
            k = int(rng.integers(1, k_max + 1))
        else:
            k = log_uniform_int(rng, 1, k_max)
        if 'child_share' in xp:
            # k counts direct neighbors (parents and children); a share of them become children, so roles change
            # without adding neighbors
            n_children = int(rng.binomial(k, xp['child_share']))
            k -= n_children
        parents = rng.choice(p, size=k, replace=False)
        if k == 0:
            z_y = standardize_by_reference(draw_noise(rng, prior['noise_families'][rng.integers(len(prior['noise_families']))], rows), n)
        else:
            # the response's product mechanisms may have several terms (xfirst max_product_terms); other nodes have one
            z_y = generate_node(rng, X_par[:, parents], n, prior | {'max_product_terms': xp.get('max_product_terms', 1)})
        if z_y is None:
            failures += 1
            continue
        y_binary = rng.random() < prior['binary_y_prob']
        if y_binary:
            y_obs = discretize(z_y, z_y[n:], rng, 2)
            y_par = standardize_by_reference(y_obs, n)
            y_par = np.zeros(rows) if y_par is None else y_par
        else:
            y_obs, y_par = z_y, z_y
        if keep_info:
            y_node = n_nodes
            n_nodes += 1
            edges += [(obs_node[j], y_node) for j in parents]
        kept_old = False

        # children of y: predictors outside the parent set, regenerated from y and up to max_spouses other predictors
        children, spouses, descendants = [], set(), set()
        if 'child_share' not in xp:
            n_children = int(rng.integers(1, xp['max_children'] + 1)) if rng.random() < xp['children_prob'] else 0
        candidates = np.setdiff1d(np.arange(p), parents)
        chosen = rng.choice(candidates, size=min(n_children, len(candidates)), replace=False)
        # spouses come from predictors that are not children, so no spouse is later overwritten
        others = np.setdiff1d(np.arange(p), chosen)
        ok = True
        for c in chosen:
            n_spouses = min(len(others), int(rng.integers(0, xp['max_spouses'] + 1)))
            if xp.get('spouses_near_child', False) and n_spouses:
                # spouses among the predictors most correlated with the replaced column, so the child keeps
                # dependence on its neighborhood through observed parents (no latent parent of a child)
                corr = np.abs(X_par[n:, others].T @ X_par[n:, c]) / prior['n_ref']
                pool = others[np.argsort(-corr)[:max(3 * n_spouses, 5)]]
                extra = rng.choice(pool, size=min(n_spouses, len(pool)), replace=False)
            else:
                extra = rng.choice(others, size=n_spouses, replace=False)
            # with child_keep_range, the child's earlier value replaces a share w of its noise (a latent parent)
            keep = 'child_keep_range' in xp and X_par[n:, c].std() > 0
            w_keep = rng.uniform(*xp['child_keep_range']) if keep else 0.0
            z_c = generate_node(rng, np.column_stack([y_par, X_par[:, extra]]), n, prior,
                background=X_par[:, c] if keep else None, background_weight=w_keep)
            if z_c is None:
                ok = False
                break
            kept_old = kept_old or keep
            if keep_info:
                edges += [(y_node, n_nodes)] + [(obs_node[e], n_nodes) for e in extra] + ([(obs_node[c], n_nodes)] if keep else [])
                obs_node[c] = n_nodes
                n_nodes += 1
            X_obs[:, c], X_par[:, c], feature_type[c] = z_c, z_c, 0
            children.append(int(c))
            spouses |= set(int(e) for e in extra)
            # descendants: predictors that also depend on the child; they are outside the blanket because the child
            # is observed. They exclude parents, children, and spouses already used, whose values must stay as generated.
            if rng.random() < xp.get('descendant_prob', 0.0):
                free = np.setdiff1d(np.arange(p), np.concatenate([parents, chosen, list(spouses)]).astype(int))
                for d in rng.choice(free, size=min(len(free), int(rng.integers(1, xp['max_descendants'] + 1))), replace=False):
                    g = generate_node(rng, X_par[:, [c]], n, prior)
                    if g is None:
                        continue
                    w = rng.uniform(0.2, 0.8)
                    z_d = standardize_by_reference(np.sqrt(1 - w) * X_par[:, d] + np.sqrt(w) * g, n)
                    if z_d is not None:
                        X_obs[:, d], X_par[:, d], feature_type[d] = z_d, z_d, 0
                        descendants.add(int(d))
                        if keep_info:
                            edges += [(obs_node[d], n_nodes), (obs_node[c], n_nodes)]
                            obs_node[d] = n_nodes
                            n_nodes += 1
        if ok:
            break
        failures += 1

    blanket = np.zeros(p, dtype=bool)
    blanket[list(parents) + children + list(spouses)] = True
    is_parent = np.zeros(p, dtype=bool)
    is_parent[parents] = True
    is_child = np.zeros(p, dtype=bool)
    is_child[children] = True
    perm = rng.permutation(p)
    labels = np.stack([blanket[perm], is_parent[perm], is_child[perm]], axis=1).astype(np.float32)
    info = None
    if keep_info:
        info = {'family': 'xfirst', 'structure': kind, 'parents': perm.argsort()[parents], 'children': perm.argsort()[children],
            'descendants': perm.argsort()[sorted(descendants)], 'spouses': perm.argsort()[sorted(spouses)],
            'feature_type': feature_type[perm], 'y_binary': bool(y_binary), 'X_par': X_par[:n, perm], 'blanket_exact': not kept_old}
        A = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
        observed = np.zeros(n_nodes, dtype=bool)
        observed[obs_node + [y_node]] = True
        info['graph'] = {'A': A, 'observed': observed, 'y': y_node, 'predictors': np.array(obs_node)[perm]}
    return X_obs[:n, perm].astype(np.float32), y_obs[:n].astype(np.float32), labels, failures, info

#----------------------------------------------------------------
# Datasets and batches
#----------------------------------------------------------------
def sample_dataset(rng, prior, n=None, p=None, keep_info=False):
    '''
    One dataset from the mixture prior. Returns X (n, p), y (n,), labels (p, 3) with columns (conditional dependence,
    parent, child), the number of numerical failures, and optionally an info dict.
    '''
    n = sample_n(rng, prior) if n is None else n
    p = sample_p(rng, prior) if p is None else p
    if prior.get('linear_gaussian_prob', 0.0) > 0 and rng.random() < prior['linear_gaussian_prob']:
        return sample_dataset_scm(rng, prior | prior['linear_gaussian'], n, p, keep_info)
    weights = prior.get('component_weights', {'scm': 1.0})
    if rng.random() < weights.get('xfirst', 0.0) / sum(weights.values()):
        return sample_dataset_xfirst(rng, prior, n, p, keep_info)
    return sample_dataset_scm(rng, prior, n, p, keep_info)

def sample_dataset_scm(rng, prior, n, p, keep_info=False):
    '''One dataset from the scm family; same outputs as sample_dataset.'''
    n_latent = 0 if rng.random() < prior['latent_zero_prob'] else int(round(p * rng.uniform(*prior['latent_fraction_range'])))
    m = p + 1 + n_latent
    A, order, lam = sample_graph(m, rng, prior)

    # roles: response, predictors (in random order), latents
    perm = rng.permutation(m)
    yn = prior.get('scm_y_neighbors')
    if yn is not None and rng.random() < yn['prob']:
        # the response's own numbers of parents and children, placed in the order with room for both
        cap = int(yn['max_fraction'] * p) if 'max_fraction' in yn else None
        max_pa, max_ch = (yn['max_parents'], yn['max_children']) if cap is None else (min(yn['max_parents'], cap), min(yn['max_children'], cap))
        k_pa, k_ch = int(rng.integers(max_pa + 1)), int(rng.integers(max_ch + 1))
        k_pa, k_ch = min(k_pa, m - 1), min(k_ch, max(m - 1 - k_pa, 0))
        pos = int(rng.integers(k_pa, m - k_ch))
        y = order[pos]
        perm[np.flatnonzero(perm == y)[0]], perm[0] = perm[0], y
        A[:, y] = 0
        A[y, :] = 0
        A[rng.choice(order[:pos], size=k_pa, replace=False), y] = 1
        A[y, rng.choice(order[pos + 1:], size=k_ch, replace=False)] = 1
    y_node, predictor_nodes = perm[0], perm[1:p + 1]

    # observation types: 0 continuous, 1 binary, 2 ordinal; discrete ones are causal or measurement-only
    levels = {}
    feature_type = np.zeros(p, dtype=int)
    for j, v in enumerate(predictor_nodes):
        u = rng.random()
        if u < prior['binary_feature_prob']:
            feature_type[j], levels[v] = 1, 2
        elif u < prior['binary_feature_prob'] + prior['ordinal_feature_prob']:
            feature_type[j], levels[v] = 2, int(rng.integers(prior['ordinal_levels'][0], prior['ordinal_levels'][1] + 1))
    y_binary = rng.random() < prior['binary_y_prob']
    if y_binary:
        levels[y_node] = 2
    measured = {v for v in levels if rng.random() < prior['measurement_discretization_prob']}
    causal_levels = {v: k for v, k in levels.items() if v not in measured}

    # latent hubs (scm_hubs): correlated predictors with children of the response embedded in the correlation structure
    hub_edges, hub_signals = [], None
    if 'scm_hubs' in prior and rng.random() < prior['scm_hubs']['prob']:
        _, hub_edges, hub_signals = sample_hubs(rng, [v for v in range(m) if v != y_node], n, prior)
    n_hubs = 1 + max([h for h, _ in hub_edges], default=-1)

    Z, Z_ref, codes, node_family, failures, redraws, pruned = simulate(A, order, n, rng, prior, causal_levels, hub_signals)

    def observe(v):
        if v in causal_levels:
            return codes[v]
        if v in measured:
            return discretize(Z[:, v], Z_ref[:, v], rng, levels[v])
        return Z[:, v]
    X = np.stack([observe(v) for v in predictor_nodes], axis=1)
    y = observe(y_node)

    # augmented graph: each measurement-only discrete variable is an observed child of its (latent) continuous node;
    # hubs are latent roots after these
    measured = sorted(measured)
    M = m + len(measured) + n_hubs
    A_aug = np.zeros((M, M), dtype=np.int8)
    A_aug[:m, :m] = A
    for h, v in hub_edges:
        A_aug[m + len(measured) + h, v] = 1
    obs_node = {v: v for v in list(predictor_nodes) + [y_node]}
    for i, v in enumerate(measured):
        A_aug[v, m + i] = 1
        obs_node[v] = m + i
    observed = np.zeros(M, dtype=bool)
    observed[[obs_node[v] for v in predictor_nodes]] = True
    observed[obs_node[y_node]] = True
    blanket = conditional_dependence(A_aug, observed, obs_node[y_node], np.array([obs_node[v] for v in predictor_nodes]))
    # parent and child are edges between the underlying nodes in G, whether or not either variable is coarsely measured
    parents = A[predictor_nodes, y_node] == 1
    children = A[y_node, predictor_nodes] == 1
    labels = np.stack([blanket, parents, children], axis=1).astype(np.float32)

    info = None
    if keep_info:
        info = {'family': 'scm', 'A': A, 'order': order, 'lam': lam, 'y_node': int(y_node), 'predictors': predictor_nodes, 'n_latent': n_latent,
            'feature_type': feature_type, 'y_binary': bool(y_binary), 'measured': measured, 'node_family': node_family, 'Z': Z,
            'mechanism_redraws': redraws, 'pruned_edges': pruned, 'n_hubs': n_hubs}
    return X.astype(np.float32), y.astype(np.float32), labels, failures, info

def sample_batch(rng, prior, batch_size, n=None, p=None, keep_info=False):
    '''All datasets in a batch share n and p (drawn once per batch if not given).'''
    n = sample_n(rng, prior) if n is None else n
    p = sample_p(rng, prior) if p is None else p
    draws = [sample_dataset(rng, prior, n=n, p=p, keep_info=keep_info) for _ in range(batch_size)]
    X = np.stack([d[0] for d in draws])
    y = np.stack([d[1] for d in draws])
    labels = np.stack([d[2] for d in draws])
    failures = sum(d[3] for d in draws)
    infos = [d[4] for d in draws]
    return X, y, labels, failures, infos

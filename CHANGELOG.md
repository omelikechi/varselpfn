# Changelog

## 0.2.0

The network weights are unchanged. PIPs are identical to 0.1.0 on numeric inputs with no missing values, no covariate with extreme values, and a response that is continuous or has a rarer class of at least 20% of the rows, as long as the rows fit the row budget (`max_rows`).

### Added

- Any DataFrame as input: categorical and text columns are encoded, text columns of numbers are read as numbers, dates become timestamps, and missing values are filled with the column median.
- Multiclass responses for `CDPFN`, scored by the mean of the one-vs-rest PIPs.
- `response` option (`'auto'`, `'regression'`, `'binary'`, `'multiclass'`); `'auto'` detects the type from `y`.
- `max_rows` option. By default, rows beyond a compute budget (10,000 rows at 100 covariates, never fewer than 1,024) are subsampled. `select()` warns when PIPs come from more than 1,024 rows.
- `balance='auto'`: a binary response, or class indicator, whose rarer class has under 20% of the rows is split into case-control subsets (every rare row plus disjoint chunks of the others, 3 to 1), and the PIPs are averaged.
- `transform='auto'`: covariates with extreme values are replaced by their ranks.

### Changed

- Constant covariates and identifier columns get PIP 0 and are left out of the forward pass.
- Faster, lighter inference: fused attention and inference in slices (peak memory 2.0 GB instead of 13.4 GB at 10,000 rows and 100 covariates).

## 0.1.0

Initial release.

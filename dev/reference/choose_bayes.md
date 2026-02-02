# Finding the Set of Hyperparameters of Bayesian Model

**\[deprecated\]** This function chooses the set of hyperparameters of
Bayesian model using
[`stats::optim()`](https://rdrr.io/r/stats/optim.html) function.

## Usage

``` r
choose_bayes(
  bayes_bound = bound_bvhar(),
  ...,
  eps = 1e-04,
  y,
  order = c(5, 22),
  include_mean = TRUE,
  parallel = list()
)
```

## Arguments

- bayes_bound:

  Empirical Bayes optimization bound specification defined by
  [`bound_bvhar()`](bound_bvhar.md).

- ...:

  Additional arguments for
  [`stats::optim()`](https://rdrr.io/r/stats/optim.html).

- eps:

  Hyperparameter `eps` is fixed. By default, `1e-04`.

- y:

  Time series data

- order:

  Order for BVAR or BVHAR. `p` of
  [`bvar_minnesota()`](bvar_minnesota.md) or `har` of
  [`bvhar_minnesota()`](bvhar_minnesota.md). By default, `c(5, 22)` for
  `har`.

- include_mean:

  Add constant term (Default: `TRUE`) or not (`FALSE`)

- parallel:

  List the same argument of
  [`optimParallel::optimParallel()`](https://rdrr.io/pkg/optimParallel/man/optimParallel.html).
  By default, this is empty, and the function does not execute parallel
  computation.

## Value

`bvharemp` [class](https://rdrr.io/r/base/class.html) is a list that has

- ...:

  Many components of
  [`stats::optim()`](https://rdrr.io/r/stats/optim.html) or
  [`optimParallel::optimParallel()`](https://rdrr.io/pkg/optimParallel/man/optimParallel.html)

- spec:

  Corresponding `bvharspec`

- fit:

  Chosen Bayesian model

- ml:

  Marginal likelihood of the final model

## References

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for
Vector Autoregressions*. Review of Economics and Statistics, 97(2).

Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous
autoregressive modeling*. Journal of Statistical Computation and
Simulation, 94(6), 1139-1157.

## See also

- [`bound_bvhar()`](bound_bvhar.md) to define L-BFGS-B optimization
  bounds.

- Individual functions: [`choose_bvar()`](choose_bvar.md)

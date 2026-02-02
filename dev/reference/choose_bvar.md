# Finding the Set of Hyperparameters of Individual Bayesian Model

**\[deprecated\]** Instead of these functions, you can use
[`choose_bayes()`](choose_bayes.md).

## Usage

``` r
choose_bvar(
  bayes_spec = set_bvar(),
  lower = 0.01,
  upper = 10,
  ...,
  eps = 1e-04,
  y,
  p,
  include_mean = TRUE,
  parallel = list()
)

choose_bvhar(
  bayes_spec = set_bvhar(),
  lower = 0.01,
  upper = 10,
  ...,
  eps = 1e-04,
  y,
  har = c(5, 22),
  include_mean = TRUE,
  parallel = list()
)

# S3 method for class 'bvharemp'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.bvharemp(x)

# S3 method for class 'bvharemp'
knit_print(x, ...)
```

## Arguments

- bayes_spec:

  Initial Bayes model specification.

- lower:

  **\[experimental\]** Lower bound. By default, `.01`.

- upper:

  **\[experimental\]** Upper bound. By default, `10`.

- ...:

  not used

- eps:

  Hyperparameter `eps` is fixed. By default, `1e-04`.

- y:

  Time series data

- p:

  BVAR lag

- include_mean:

  Add constant term (Default: `TRUE`) or not (`FALSE`)

- parallel:

  List the same argument of
  [`optimParallel::optimParallel()`](https://rdrr.io/pkg/optimParallel/man/optimParallel.html).
  By default, this is empty, and the function does not execute parallel
  computation.

- har:

  Numeric vector for weekly and monthly order. By default, `c(5, 22)`.

- x:

  Any object

- digits:

  digit option to print

## Value

`bvharemp` [class](https://rdrr.io/r/base/class.html) is a list that has

- [`stats::optim()`](https://rdrr.io/r/stats/optim.html) or
  [`optimParallel::optimParallel()`](https://rdrr.io/pkg/optimParallel/man/optimParallel.html)

- chosen `bvharspec` set

- Bayesian model fit result with chosen specification

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

## Details

Empirical Bayes method maximizes marginal likelihood and selects the set
of hyperparameters. These functions implement `L-BFGS-B` method of
[`stats::optim()`](https://rdrr.io/r/stats/optim.html) to find the
maximum of marginal likelihood.

If you want to set `lower` and `upper` option more carefully, deal with
them like as in [`stats::optim()`](https://rdrr.io/r/stats/optim.html)
in order of [`set_bvar()`](set_bvar.md), [`set_bvhar()`](set_bvar.md),
or [`set_weight_bvhar()`](set_bvar.md)'s argument (except `eps`). In
other words, just arrange them in a vector.

## References

Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). *A limited memory
algorithm for bound constrained optimization*. SIAM Journal on
scientific computing, 16(5), 1190-1208.

Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013).
*Bayesian data analysis*. Chapman and Hall/CRC.

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for
Vector Autoregressions*. Review of Economics and Statistics, 97(2).

Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous
autoregressive modeling*. Journal of Statistical Computation and
Simulation, 94(6), 1139-1157.

# Fitting Bayesian VHAR with Coefficient and Covariance Prior

**\[maturing\]** This function fits BVHAR. Covariance term can be
homoskedastic or heteroskedastic (stochastic volatility). It can have
Minnesota, SSVS, and Horseshoe prior.

## Usage

``` r
vhar_bayes(
  y,
  har = c(5, 22),
  exogen = NULL,
  s = 0,
  factor_spec = set_factor(),
  num_chains = 1,
  num_iter = 1000,
  num_burn = floor(num_iter/2),
  thinning = 1,
  coef_spec = set_bvhar(),
  contem_spec = coef_spec,
  cov_spec = set_ldlt(),
  intercept = set_intercept(),
  exogen_spec = coef_spec,
  loading_spec = coef_spec,
  include_mean = TRUE,
  minnesota = c("longrun", "short", "no"),
  ggl = TRUE,
  save_init = FALSE,
  convergence = NULL,
  verbose = FALSE,
  num_thread = 1
)

# S3 method for class 'bvharsv'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvharldlt'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvharsv'
knit_print(x, ...)

# S3 method for class 'bvharldlt'
knit_print(x, ...)
```

## Arguments

- y:

  Time series data of which columns indicate the variables

- har:

  Numeric vector for weekly and monthly order. By default, `c(5, 22)`.

- exogen:

  Unmodeled variables

- s:

  Lag of exogeneous variables in VARX(p, s). By default, `s = 0`.

- factor_spec:

  **\[experimental\]** Factor augmentation specification by
  [`set_factor()`](set_factor.md).

- num_chains:

  Number of MCMC chains

- num_iter:

  MCMC iteration number

- num_burn:

  Number of burn-in (warm-up). Half of the iteration is the default
  choice.

- thinning:

  Thinning every thinning-th iteration

- coef_spec:

  Coefficient prior specification by [`set_bvar()`](set_bvar.md),
  [`set_ssvs()`](set_ssvs.md), or [`set_horseshoe()`](set_horseshoe.md).

- contem_spec:

  Contemporaneous coefficient prior specification by
  [`set_bvar()`](set_bvar.md), [`set_ssvs()`](set_ssvs.md), or
  [`set_horseshoe()`](set_horseshoe.md).

- cov_spec:

  **\[experimental\]** SV specification by [`set_sv()`](set_ldlt.md).

- intercept:

  **\[experimental\]** Prior for the constant term by
  [`set_intercept()`](set_intercept.md).

- exogen_spec:

  Exogenous coefficient prior specification.

- loading_spec:

  **\[experimental\]** Factor loading prior specification.

- include_mean:

  Add constant term (Default: `TRUE`) or not (`FALSE`)

- minnesota:

  Apply cross-variable shrinkage structure (Minnesota-way). Two type:
  `short` type and `longrun` (default) type. You can also set `no`.

- ggl:

  If `TRUE` (default), use additional group shrinkage parameter for
  group structure. Otherwise, use group shrinkage parameter instead of
  global shirnkage parameter. Applies to HS, NG, and DL priors.

- save_init:

  Save every record starting from the initial values (`TRUE`). By
  default, exclude the initial values in the record (`FALSE`), even when
  `num_burn = 0` and `thinning = 1`. If `num_burn > 0` or
  `thinning != 1`, this option is ignored.

- convergence:

  Convergence threshold for rhat \< convergence. By default, `NULL`
  which means no warning.

- verbose:

  Print the progress bar in the console. By default, `FALSE`.

- num_thread:

  Number of threads

- x:

  `bvharldlt` object

- digits:

  digit option to print

- ...:

  not used

## Value

`vhar_bayes()` returns an object named `bvharsv`
[class](https://rdrr.io/r/base/class.html). It is a list with the
following components:

- coefficients:

  Posterior mean of coefficients.

- chol_posterior:

  Posterior mean of contemporaneous effects.

- param:

  Every set of MCMC trace.

- param_names:

  Name of every parameter.

- group:

  Indicators for group.

- num_group:

  Number of groups.

- df:

  Numer of Coefficients: `3m + 1` or `3m`

- p:

  3 (The number of terms. It contains this element for usage in other
  functions.)

- week:

  Order for weekly term

- month:

  Order for monthly term

- m:

  Dimension of the data

- obs:

  Sample size used when training = `totobs` - `p`

- totobs:

  Total number of the observation

- call:

  Matched call

- process:

  Description of the model, e.g. `VHAR_SSVS_SV`, `VHAR_Horseshoe_SV`, or
  `VHAR_minnesota-part_SV`

- type:

  include constant term (`const`) or not (`none`)

- spec:

  Coefficients prior specification

- sv:

  log volatility prior specification

- init:

  Initial values

- intercept:

  Intercept prior specification

- chain:

  The numer of chains

- iter:

  Total iterations

- burn:

  Burn-in

- thin:

  Thinning

- HARtrans:

  VHAR linear transformation matrix

- y0:

  \\Y_0\\

- design:

  \\X_0\\

- y:

  Raw input

If it is SSVS or Horseshoe:

- pip:

  Posterior inclusion probabilities.

## Details

Cholesky stochastic volatility modeling for VHAR based on
\$\$\Sigma_t^{-1} = L^T D_t^{-1} L\$\$

## References

Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous
autoregressive modeling*. Journal of Statistical Computation and
Simulation, 94(6), 1139-1157.

Kim, Y. G., and Baek, C. (n.d.). Working paper.

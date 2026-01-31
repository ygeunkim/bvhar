# Fitting Bayesian VAR with Coefficient and Covariance Prior

**\[maturing\]** This function fits BVAR. Covariance term can be
homoskedastic or heteroskedastic (stochastic volatility). It can have
Minnesota, SSVS, and Horseshoe prior.

## Usage

``` r
var_bayes(
  y,
  p,
  exogen = NULL,
  s = 0,
  factor_spec = set_factor(),
  num_chains = 1,
  num_iter = 1000,
  num_burn = floor(num_iter/2),
  thinning = 1,
  coef_spec = set_bvar(),
  contem_spec = coef_spec,
  cov_spec = set_ldlt(),
  intercept = set_intercept(),
  exogen_spec = coef_spec,
  loading_spec = coef_spec,
  include_mean = TRUE,
  minnesota = TRUE,
  ggl = TRUE,
  save_init = FALSE,
  convergence = NULL,
  verbose = FALSE,
  num_thread = 1
)

# S3 method for class 'bvarsv'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvarldlt'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvarsv'
knit_print(x, ...)

# S3 method for class 'bvarldlt'
knit_print(x, ...)
```

## Arguments

- y:

  Time series data of which columns indicate the variables

- p:

  VAR lag

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

  Apply cross-variable shrinkage structure (Minnesota-way). By default,
  `TRUE`.

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

  `bvarldlt` object

- digits:

  digit option to print

- ...:

  not used

## Value

`var_bayes()` returns an object named `bvarsv`
[class](https://rdrr.io/r/base/class.html).

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

  VAR lag

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

- intercept:

  Intercept prior specification

- init:

  Initial values

- chain:

  The numer of chains

- iter:

  Total iterations

- burn:

  Burn-in

- thin:

  Thinning

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

Cholesky stochastic volatility modeling for VAR based on
\$\$\Sigma_t^{-1} = L^T D_t^{-1} L\$\$, and implements corrected
triangular algorithm for Gibbs sampler.

## References

Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022).
*Corrigendum to “Large Bayesian vector autoregressions with stochastic
volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019)
137-154\]*. Journal of Econometrics, 227(2), 506-512.

Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian
Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge:
Cambridge University Press.

Cogley, T., & Sargent, T. J. (2005). *Drifts and volatilities: monetary
policies and outcomes in the post WWII US*. Review of Economic Dynamics,
8(2), 262-302.

Gruber, L., & Kastner, G. (2022). *Forecasting macroeconomic data with
Bayesian VARs: Sparse or dense? It depends!* arXiv.

Huber, F., Koop, G., & Onorante, L. (2021). *Inducing Sparsity and
Shrinkage in Time-Varying Parameter Models*. Journal of Business &
Economic Statistics, 39(3), 669-683.

Korobilis, D. (2022). A new algorithm for structural restrictions in
Bayesian vector autoregressions. European Economic Review, 148, 104241.

Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage
and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4),
230-354.

Ray, P., & Bhattacharya, A. (2018). *Signal Adaptive Variable Selector
for the Horseshoe Prior*. arXiv.

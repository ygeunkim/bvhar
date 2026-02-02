# Hyperparameters for Bayesian Models

Set hyperparameters of Bayesian VAR and VHAR models.

## Usage

``` r
set_bvar(sigma, lambda = 0.1, delta, eps = 1e-04)

set_bvar_flat(U)

set_bvhar(sigma, lambda = 0.1, delta, eps = 1e-04)

set_weight_bvhar(sigma, lambda = 0.1, eps = 1e-04, daily, weekly, monthly)

# S3 method for class 'bvharspec'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.bvharspec(x)

# S3 method for class 'bvharspec'
knit_print(x, ...)
```

## Arguments

- sigma:

  Standard error vector for each variable (Default: sd)

- lambda:

  Tightness of the prior around a random walk or white noise (Default:
  .1)

- delta:

  Persistence (Default: Litterman sets 1 = random walk prior, White
  noise prior = 0)

- eps:

  Very small number (Default: 1e-04)

- U:

  Positive definite matrix. By default, identity matrix of dimension
  ncol(X0)

- daily:

  Same as delta in VHAR type (Default: 1 as Litterman)

- weekly:

  Fill the second part in the first block (Default: 1)

- monthly:

  Fill the third part in the first block (Default: 1)

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Value

Every function returns `bvharspec`
[class](https://rdrr.io/r/base/class.html). It is the list of which the
components are the same as the arguments provided. If the argument is
not specified, `NULL` is assigned here. The default values mentioned
above will be considered in each fitting function.

- process:

  Model name: `BVAR`, `BVHAR`

- prior:

  Prior name: `Minnesota` (Minnesota prior for BVAR), `Hierarchical`
  (Hierarchical prior for BVAR), `MN_VAR` (BVHAR-S), `MN_VHAR`
  (BVHAR-L), `Flat` (Flat prior for BVAR)

- sigma:

  Vector value (or `bvharpriorspec` class) assigned for sigma

- lambda:

  Value (or `bvharpriorspec` class) assigned for lambda

- delta:

  Vector value assigned for delta

- eps:

  Value assigned for epsilon

`set_weight_bvhar()` has different component with `delta` due to its
different construction.

- daily:

  Vector value assigned for daily weight

- weekly:

  Vector value assigned for weekly weight

- monthly:

  Vector value assigned for monthly weight

## Details

- Missing arguments will be set to be default values in each model
  function mentioned above.

- `set_bvar()` sets hyperparameters for
  [`bvar_minnesota()`](bvar_minnesota.md).

- Each `delta` (vector), `lambda` (length of 1), `sigma` (vector), `eps`
  (vector) corresponds to \\\delta_j\\, \\\lambda\\, \\\delta_j\\,
  \\\epsilon\\.

\\\delta_i\\ are related to the belief to random walk.

- If \\\delta_i = 1\\ for all i, random walk prior

- If \\\delta_i = 0\\ for all i, white noise prior

\\\lambda\\ controls the overall tightness of the prior around these two
prior beliefs.

- If \\\lambda = 0\\, the posterior is equivalent to prior and the data
  do not influence the estimates.

- If \\\lambda = \infty\\, the posterior mean becomes OLS estimates
  (VAR).

\\\sigma_i^2 / \sigma_j^2\\ in Minnesota moments explain the data
scales.

- `set_bvar_flat` sets hyperparameters for
  [`bvar_flat()`](bvar_flat.md).

&nbsp;

- `set_bvhar()` sets hyperparameters for
  [`bvhar_minnesota()`](bvhar_minnesota.md) with VAR-type Minnesota
  prior, i.e. BVHAR-S model.

&nbsp;

- `set_weight_bvhar()` sets hyperparameters for
  [`bvhar_minnesota()`](bvhar_minnesota.md) with VHAR-type Minnesota
  prior, i.e. BVHAR-L model.

## Note

By using [`set_psi()`](set_lambda.md) and
[`set_lambda()`](set_lambda.md) each, hierarchical modeling is
available.

## References

Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector
auto regressions*. Journal of Applied Econometrics, 25(1).

Litterman, R. B. (1986). *Forecasting with Bayesian Vector
Autoregressions: Five Years of Experience*. Journal of Business &
Economic Statistics, 4(1), 25.

Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional
Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal
of the American Statistical Association, 114(526).

Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous
autoregressive modeling*. Journal of Statistical Computation and
Simulation, 94(6), 1139-1157.

Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous
autoregressive modeling*. Journal of Statistical Computation and
Simulation, 94(6), 1139-1157.

## See also

- lambda hyperprior specification [`set_lambda()`](set_lambda.md)

- sigma hyperprior specification [`set_psi()`](set_lambda.md)

## Examples

``` r
# Minnesota BVAR specification------------------------
bvar_spec <- set_bvar(
  sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
  lambda = .2, # lambda = .2
  delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
  eps = 1e-04 # eps = 1e-04
)
class(bvar_spec)
#> [1] "bvharspec"
str(bvar_spec)
#> List of 7
#>  $ process     : chr "BVAR"
#>  $ prior       : chr "Minnesota"
#>  $ sigma       : num [1:3] 0.03 0.02 0.01
#>  $ lambda      : num 0.2
#>  $ delta       : num [1:3] 0.1 0.1 0.1
#>  $ eps         : num 1e-04
#>  $ hierarchical: logi FALSE
#>  - attr(*, "class")= chr "bvharspec"
# Flat BVAR specification-------------------------
# 3-dim
# p = 5 with constant term
# U = 500 * I(mp + 1)
bvar_flat_spec <- set_bvar_flat(U = 500 * diag(16))
class(bvar_flat_spec)
#> [1] "bvharspec"
str(bvar_flat_spec)
#> List of 3
#>  $ process: chr "BVAR"
#>  $ prior  : chr "Flat"
#>  $ U      : num [1:16, 1:16] 500 0 0 0 0 0 0 0 0 0 ...
#>  - attr(*, "class")= chr "bvharspec"
# BVHAR-S specification-----------------------
bvhar_var_spec <- set_bvhar(
  sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
  lambda = .2, # lambda = .2
  delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
  eps = 1e-04 # eps = 1e-04
)
class(bvhar_var_spec)
#> [1] "bvharspec"
str(bvhar_var_spec)
#> List of 7
#>  $ process     : chr "BVHAR"
#>  $ prior       : chr "MN_VAR"
#>  $ sigma       : num [1:3] 0.03 0.02 0.01
#>  $ lambda      : num 0.2
#>  $ delta       : num [1:3] 0.1 0.1 0.1
#>  $ eps         : num 1e-04
#>  $ hierarchical: logi FALSE
#>  - attr(*, "class")= chr "bvharspec"
# BVHAR-L specification---------------------------
bvhar_vhar_spec <- set_weight_bvhar(
  sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
  lambda = .2, # lambda = .2
  eps = 1e-04, # eps = 1e-04
  daily = rep(.2, 3), # daily1 = .2, daily2 = .2, daily3 = .2
  weekly = rep(.1, 3), # weekly1 = .1, weekly2 = .1, weekly3 = .1
  monthly = rep(.05, 3) # monthly1 = .05, monthly2 = .05, monthly3 = .05
)
class(bvhar_vhar_spec)
#> [1] "bvharspec"
str(bvhar_vhar_spec)
#> List of 9
#>  $ process     : chr "BVHAR"
#>  $ prior       : chr "MN_VHAR"
#>  $ sigma       : num [1:3] 0.03 0.02 0.01
#>  $ lambda      : num 0.2
#>  $ eps         : num 1e-04
#>  $ daily       : num [1:3] 0.2 0.2 0.2
#>  $ weekly      : num [1:3] 0.1 0.1 0.1
#>  $ monthly     : num [1:3] 0.05 0.05 0.05
#>  $ hierarchical: logi FALSE
#>  - attr(*, "class")= chr "bvharspec"
```

# Fitting Bayesian VAR(p) of Flat Prior

This function fits BVAR(p) with flat prior.

## Usage

``` r
bvar_flat(
  y,
  p,
  num_chains = 1,
  num_iter = 1000,
  num_burn = floor(num_iter/2),
  thinning = 1,
  bayes_spec = set_bvar_flat(),
  include_mean = TRUE,
  verbose = FALSE,
  num_thread = 1
)

# S3 method for class 'bvarflat'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvarflat'
logLik(object, ...)

# S3 method for class 'bvarflat'
AIC(object, ...)

# S3 method for class 'bvarflat'
BIC(object, ...)

is.bvarflat(x)

# S3 method for class 'bvarflat'
knit_print(x, ...)
```

## Arguments

- y:

  Time series data of which columns indicate the variables

- p:

  VAR lag

- num_chains:

  Number of MCMC chains

- num_iter:

  MCMC iteration number

- num_burn:

  Number of burn-in (warm-up). Half of the iteration is the default
  choice.

- thinning:

  Thinning every thinning-th iteration

- bayes_spec:

  A BVAR model specification by [`set_bvar_flat()`](set_bvar.md).

- include_mean:

  Add constant term (Default: `TRUE`) or not (`FALSE`)

- verbose:

  Print the progress bar in the console. By default, `FALSE`.

- num_thread:

  Number of threads

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

- object:

  A `bvarflat` object

## Value

`bvar_flat()` returns an object `bvarflat`
[class](https://rdrr.io/r/base/class.html). It is a list with the
following components:

- coefficients:

  Posterior Mean matrix of Matrix Normal distribution

- fitted.values:

  Fitted values

- residuals:

  Residuals

- mn_prec:

  Posterior precision matrix of Matrix Normal distribution

- iw_scale:

  Posterior scale matrix of posterior inverse-wishart distribution

- iw_shape:

  Posterior shape of inverse-wishart distribution

- df:

  Numer of Coefficients: mp + 1 or mp

- p:

  Lag of VAR

- m:

  Dimension of the time series

- obs:

  Sample size used when training = `totobs` - `p`

- totobs:

  Total number of the observation

- process:

  Process string in the `bayes_spec`: `BVAR_Flat`

- spec:

  Model specification (`bvharspec`)

- type:

  include constant term (`const`) or not (`none`)

- call:

  Matched call

- prior_mean:

  Prior mean matrix of Matrix Normal distribution: zero matrix

- prior_precision:

  Prior precision matrix of Matrix Normal distribution: \\U^{-1}\\

- y0:

  \\Y_0\\

- design:

  \\X_0\\

- y:

  Raw input (`matrix`)

## Details

Ghosh et al. (2018) gives flat prior for residual matrix in BVAR.

Under this setting, there are many models such as hierarchical or
non-hierarchical. This function chooses the most simple non-hierarchical
matrix normal prior in Section 3.1.

\$\$A \mid \Sigma_e \sim MN(0, U^{-1}, \Sigma_e)\$\$ where U: precision
matrix (MN: [matrix
normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution)).
\$\$p (\Sigma_e) \propto 1\$\$

## References

Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional
Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal
of the American Statistical Association, 114(526).

Litterman, R. B. (1986). *Forecasting with Bayesian Vector
Autoregressions: Five Years of Experience*. Journal of Business &
Economic Statistics, 4(1), 25.

## See also

- [`set_bvar_flat()`](set_bvar.md) to specify the hyperparameters of
  BVAR flat prior.

- [`coef.bvarflat()`](coef.md), [`residuals.bvarflat()`](residuals.md),
  and [`fitted.bvarflat()`](fitted.md)

- [`predict.bvarflat()`](predict.md) to forecast the BVHAR process

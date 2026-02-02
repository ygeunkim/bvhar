# Summarizing Bayesian Multivariate Time Series Model

`summary` method for `normaliw` class.

## Usage

``` r
# S3 method for class 'normaliw'
summary(
  object,
  num_chains = 1,
  num_iter = 1000,
  num_burn = floor(num_iter/2),
  thinning = 1,
  verbose = FALSE,
  num_thread = 1,
  ...
)

# S3 method for class 'summary.normaliw'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'summary.normaliw'
knit_print(x, ...)
```

## Arguments

- object:

  A `normaliw` object

- num_chains:

  Number of MCMC chains

- num_iter:

  MCMC iteration number

- num_burn:

  Number of burn-in (warm-up). Half of the iteration is the default
  choice.

- thinning:

  Thinning every thinning-th iteration

- verbose:

  Print the progress bar in the console. By default, `FALSE`.

- num_thread:

  Number of threads

- ...:

  not used

- x:

  `summary.normaliw` object

- digits:

  digit option to print

## Value

`summary.normaliw` [class](https://rdrr.io/r/base/class.html) has the
following components:

- names:

  Variable names

- totobs:

  Total number of the observation

- obs:

  Sample size used when training = `totobs` - `p`

- p:

  Lag of VAR

- m:

  Dimension of the data

- call:

  Matched call

- spec:

  Model specification (`bvharspec`)

- mn_mean:

  MN Mean of posterior distribution (MN-IW)

- mn_prec:

  MN Precision of posterior distribution (MN-IW)

- iw_scale:

  IW scale of posterior distribution (MN-IW)

- iw_shape:

  IW df of posterior distribution (MN-IW)

- iter:

  Number of MCMC iterations

- burn:

  Number of MCMC burn-in

- thin:

  MCMC thinning

- alpha_record (BVAR) and phi_record (BVHAR):

  MCMC record of coefficients vector

- psi_record:

  MCMC record of upper cholesky factor

- omega_record:

  MCMC record of diagonal of cholesky factor

- eta_record:

  MCMC record of upper part of cholesky factor

- param:

  MCMC record of every parameter

- coefficients:

  Posterior mean of coefficients

- covmat:

  Posterior mean of covariance

## Details

From Minnesota prior, set of coefficient matrices and residual
covariance matrix have matrix Normal Inverse-Wishart distribution.

BVAR:

\$\$(A, \Sigma_e) \sim MNIW(\hat{A}, \hat{V}^{-1}, \hat\Sigma_e,
\alpha_0 + n)\$\$ where \\\hat{V} = X\_\ast^T X\_\ast\\ is the posterior
precision of MN.

BVHAR:

\$\$(\Phi, \Sigma_e) \sim MNIW(\hat\Phi, \hat{V}\_H^{-1}, \hat\Sigma_e,
\nu + n)\$\$ where \\\hat{V}\_H = X\_{+}^T X\_{+}\\ is the posterior
precision of MN.

## References

Litterman, R. B. (1986). *Forecasting with Bayesian Vector
Autoregressions: Five Years of Experience*. Journal of Business &
Economic Statistics, 4(1), 25.

Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector
auto regressions*. Journal of Applied Econometrics, 25(1).

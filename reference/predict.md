# Forecasting Multivariate Time Series

Forecasts multivariate time series using given model.

## Usage

``` r
# S3 method for class 'varlse'
predict(object, n_ahead, level = 0.05, newxreg, ...)

# S3 method for class 'vharlse'
predict(object, n_ahead, level = 0.05, newxreg, ...)

# S3 method for class 'bvarmn'
predict(object, n_ahead, n_iter = 100L, level = 0.05, num_thread = 1, ...)

# S3 method for class 'bvharmn'
predict(object, n_ahead, n_iter = 100L, level = 0.05, num_thread = 1, ...)

# S3 method for class 'bvarflat'
predict(object, n_ahead, n_iter = 100L, level = 0.05, num_thread = 1, ...)

# S3 method for class 'bvarldlt'
predict(
  object,
  n_ahead,
  level = 0.05,
  newxreg,
  stable = FALSE,
  num_thread = 1,
  sparse = FALSE,
  med = FALSE,
  warn = FALSE,
  ...
)

# S3 method for class 'bvharldlt'
predict(
  object,
  n_ahead,
  level = 0.05,
  newxreg,
  stable = FALSE,
  num_thread = 1,
  sparse = FALSE,
  med = FALSE,
  warn = FALSE,
  ...
)

# S3 method for class 'bvarsv'
predict(
  object,
  n_ahead,
  level = 0.05,
  newxreg,
  stable = FALSE,
  num_thread = 1,
  use_sv = TRUE,
  sparse = FALSE,
  med = FALSE,
  warn = FALSE,
  ...
)

# S3 method for class 'bvharsv'
predict(
  object,
  n_ahead,
  level = 0.05,
  newxreg,
  stable = FALSE,
  num_thread = 1,
  use_sv = TRUE,
  sparse = FALSE,
  med = FALSE,
  warn = FALSE,
  ...
)

# S3 method for class 'predbvhar'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.predbvhar(x)

# S3 method for class 'predbvhar'
knit_print(x, ...)
```

## Arguments

- object:

  Model object

- n_ahead:

  step to forecast

- level:

  Specify alpha of confidence interval level 100(1 - alpha) percentage.
  By default, .05.

- newxreg:

  New values for exogenous variables. Should have the same row numbers
  with `n_ahead`.

- ...:

  not used

- n_iter:

  Number to sample residual matrix from inverse-wishart distribution. By
  default, 100.

- num_thread:

  Number of threads

- stable:

  **\[experimental\]** Filter only stable coefficient draws in MCMC
  records.

- sparse:

  **\[experimental\]** Apply restriction. By default, `FALSE`. Give CI
  level (e.g. `.05`) instead of `TRUE` to use credible interval across
  MCMC for restriction.

- med:

  **\[experimental\]** If `TRUE`, use median of forecast draws instead
  of mean (default).

- warn:

  Give warning for stability of each coefficients record. By default,
  `FALSE`.

- use_sv:

  Use SV term

- x:

  Any object

- digits:

  digit option to print

## Value

`predbvhar` [class](https://rdrr.io/r/base/class.html) with the
following components:

- process:

  object\$process

- forecast:

  forecast matrix

- se:

  standard error matrix

- lower:

  lower confidence interval

- upper:

  upper confidence interval

- lower_joint:

  lower CI adjusted (Bonferroni)

- upper_joint:

  upper CI adjusted (Bonferroni)

- y:

  object\$y

## n-step ahead forecasting VAR(p)

See pp35 of Lütkepohl (2007). Consider h-step ahead forecasting (e.g.
n + 1, ... n + h).

Let \\y\_{(n)}^T = (y_n^T, ..., y\_{n - p + 1}^T, 1)\\. Then one-step
ahead (point) forecasting: \$\$\hat{y}\_{n + 1}^T = y\_{(n)}^T
\hat{B}\$\$

Recursively, let \\\hat{y}\_{(n + 1)}^T = (\hat{y}\_{n + 1}^T, y_n^T,
..., y\_{n - p + 2}^T, 1)\\. Then two-step ahead (point) forecasting:
\$\$\hat{y}\_{n + 2}^T = \hat{y}\_{(n + 1)}^T \hat{B}\$\$

Similarly, h-step ahead (point) forecasting: \$\$\hat{y}\_{n + h}^T =
\hat{y}\_{(n + h - 1)}^T \hat{B}\$\$

How about confident region? Confidence interval at h-period is
\$\$y\_{k,t}(h) \pm z\_(\alpha / 2) \sigma_k (h)\$\$

Joint forecast region of \\100(1-\alpha)\\% can be computed by \$\$\\
(y\_{k, 1}, y\_{k, h}) \mid y\_{k, n}(i) - z\_{(\alpha / 2h)}
\sigma_n(i) \le y\_{n, i} \le y\_{k, n}(i) + z\_{(\alpha / 2h)}
\sigma_k(i), i = 1, \ldots, h \\\$\$ See the pp41 of Lütkepohl (2007).

To compute covariance matrix, it needs VMA representation: \$\$Y\_{t}(h)
= c + \sum\_{i = h}^{\infty} W\_{i} \epsilon\_{t + h - i} = c + \sum\_{i
= 0}^{\infty} W\_{h + i} \epsilon\_{t - i}\$\$

Then

\$\$\Sigma_y(h) = MSE \[ y_t(h) \] = \sum\_{i = 0}^{h - 1} W_i
\Sigma\_{\epsilon} W_i^T = \Sigma_y(h - 1) + W\_{h - 1}
\Sigma\_{\epsilon} W\_{h - 1}^T\$\$

## n-step ahead forecasting VHAR

Let \\T\_{HAR}\\ is VHAR linear transformation matrix. Since VHAR is the
linearly transformed VAR(22), let \\y\_{(n)}^T = (y_n^T, y\_{n - 1}^T,
..., y\_{n - 21}^T, 1)\\.

Then one-step ahead (point) forecasting: \$\$\hat{y}\_{n + 1}^T =
y\_{(n)}^T T\_{HAR} \hat{\Phi}\$\$

Recursively, let \\\hat{y}\_{(n + 1)}^T = (\hat{y}\_{n + 1}^T, y_n^T,
..., y\_{n - 20}^T, 1)\\. Then two-step ahead (point) forecasting:
\$\$\hat{y}\_{n + 2}^T = \hat{y}\_{(n + 1)}^T T\_{HAR} \hat{\Phi}\$\$

and h-step ahead (point) forecasting: \$\$\hat{y}\_{n + h}^T =
\hat{y}\_{(n + h - 1)}^T T\_{HAR} \hat{\Phi}\$\$

## n-step ahead forecasting BVAR(p) with minnesota prior

Point forecasts are computed by posterior mean of the parameters. See
Section 3 of Bańbura et al. (2010).

Let \\\hat{B}\\ be the posterior MN mean and let \\\hat{V}\\ be the
posterior MN precision.

Then predictive posterior for each step

\$\$y\_{n + 1} \mid \Sigma_e, y \sim N( vec(y\_{(n)}^T A), \Sigma_e
\otimes (1 + y\_{(n)}^T \hat{V}^{-1} y\_{(n)}) )\$\$ \$\$y\_{n + 2} \mid
\Sigma_e, y \sim N( vec(\hat{y}\_{(n + 1)}^T A), \Sigma_e \otimes (1 +
\hat{y}\_{(n + 1)}^T \hat{V}^{-1} \hat{y}\_{(n + 1)}) )\$\$ and
recursively, \$\$y\_{n + h} \mid \Sigma_e, y \sim N( vec(\hat{y}\_{(n +
h - 1)}^T A), \Sigma_e \otimes (1 + \hat{y}\_{(n + h - 1)}^T
\hat{V}^{-1} \hat{y}\_{(n + h - 1)}) )\$\$

## n-step ahead forecasting BVHAR

Let \\\hat\Phi\\ be the posterior MN mean and let \\\hat\Psi\\ be the
posterior MN precision.

Then predictive posterior for each step

\$\$y\_{n + 1} \mid \Sigma_e, y \sim N( vec(y\_{(n)}^T \tilde{T}^T
\Phi), \Sigma_e \otimes (1 + y\_{(n)}^T \tilde{T} \hat\Psi^{-1}
\tilde{T} y\_{(n)}) )\$\$ \$\$y\_{n + 2} \mid \Sigma_e, y \sim N(
vec(y\_{(n + 1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y\_{(n +
1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y\_{(n + 1)}) )\$\$ and
recursively, \$\$y\_{n + h} \mid \Sigma_e, y \sim N( vec(y\_{(n + h -
1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y\_{(n + h - 1)}^T
\tilde{T} \hat\Psi^{-1} \tilde{T} y\_{(n + h - 1)}) )\$\$

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized
Volatility*. Journal of Financial Econometrics, 7(2), 174-196.

Baek, C. and Park, M. (2021). *Sparse vector heterogeneous
autoregressive modeling for realized volatility*. J. Korean Stat. Soc.
50, 495-510.

Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector
auto regressions*. Journal of Applied Econometrics, 25(1).

Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013).
*Bayesian data analysis*. Chapman and Hall/CRC.

Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector
Autoregression*. Handbook of Economic Forecasting, 2, 791-897.

Litterman, R. B. (1986). *Forecasting with Bayesian Vector
Autoregressions: Five Years of Experience*. Journal of Business &
Economic Statistics, 4(1), 25.

Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional
Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal
of the American Statistical Association, 114(526).

Korobilis, D. (2013). *VAR FORECASTING USING BAYESIAN VARIABLE
SELECTION*. Journal of Applied Econometrics, 28(2).

Korobilis, D. (2013). *VAR FORECASTING USING BAYESIAN VARIABLE
SELECTION*. Journal of Applied Econometrics, 28(2).

Huber, F., Koop, G., & Onorante, L. (2021). *Inducing Sparsity and
Shrinkage in Time-Varying Parameter Models*. Journal of Business &
Economic Statistics, 39(3), 669-683.

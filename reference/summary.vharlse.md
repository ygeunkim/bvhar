# Summarizing Vector HAR Model

`summary` method for `vharlse` class.

## Usage

``` r
# S3 method for class 'vharlse'
summary(object, ...)

# S3 method for class 'summary.vharlse'
print(x, digits = max(3L, getOption("digits") - 3L), signif_code = TRUE, ...)

# S3 method for class 'summary.vharlse'
knit_print(x, ...)
```

## Arguments

- object:

  A `vharlse` object

- ...:

  not used

- x:

  `summary.vharlse` object

- digits:

  digit option to print

- signif_code:

  Check significant rows (Default: `TRUE`)

## Value

`summary.vharlse` [class](https://rdrr.io/r/base/class.html)
additionally computes the following

- `names`:

  Variable names

- `totobs`:

  Total number of the observation

- `obs`:

  Sample size used when training = `totobs` - `p`

- `p`:

  3

- `week`:

  Order for weekly term

- `month`:

  Order for monthly term

- `coefficients`:

  Coefficient Matrix

- `call`:

  Matched call

- `process`:

  Process: VAR

- `covmat`:

  Covariance matrix of the residuals

- `corrmat`:

  Correlation matrix of the residuals

- `roots`:

  Roots of characteristic polynomials

- `is_stable`:

  Whether the process is stable or not based on `roots`

- `log_lik`:

  log-likelihood

- `ic`:

  Information criteria vector

&nbsp;

- `AIC` - AIC

- `BIC` - BIC

- `HQ` - HQ

- `FPE` - FPE

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized
Volatility*. Journal of Financial Econometrics, 7(2), 174-196.

Baek, C. and Park, M. (2021). *Sparse vector heterogeneous
autoregressive modeling for realized volatility*. J. Korean Stat. Soc.
50, 495-510.

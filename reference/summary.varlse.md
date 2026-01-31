# Summarizing Vector Autoregressive Model

`summary` method for `varlse` class.

## Usage

``` r
# S3 method for class 'varlse'
summary(object, ...)

# S3 method for class 'summary.varlse'
print(x, digits = max(3L, getOption("digits") - 3L), signif_code = TRUE, ...)

# S3 method for class 'summary.varlse'
knit_print(x, ...)
```

## Arguments

- object:

  A `varlse` object

- ...:

  not used

- x:

  `summary.varlse` object

- digits:

  digit option to print

- signif_code:

  Check significant rows (Default: `TRUE`)

## Value

`summary.varlse` [class](https://rdrr.io/r/base/class.html) additionally
computes the following

- `names`:

  Variable names

- `totobs`:

  Total number of the observation

- `obs`:

  Sample size used when training = `totobs` - `p`

- `p`:

  Lag of VAR

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

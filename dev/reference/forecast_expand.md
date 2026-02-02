# Out-of-sample Forecasting based on Expanding Window

This function conducts expanding window forecasting.

## Usage

``` r
forecast_expand(
  object,
  n_ahead,
  y_test,
  level = 0.05,
  newxreg = NULL,
  num_thread = 1,
  ...
)

# S3 method for class 'olsmod'
forecast_expand(
  object,
  n_ahead,
  y_test,
  level = 0.05,
  newxreg = NULL,
  num_thread = 1,
  ...
)

# S3 method for class 'normaliw'
forecast_expand(
  object,
  n_ahead,
  y_test,
  level = 0.05,
  newxreg = NULL,
  num_thread = 1,
  use_fit = TRUE,
  ...
)

# S3 method for class 'ldltmod'
forecast_expand(
  object,
  n_ahead,
  y_test,
  level = 0.05,
  newxreg = NULL,
  num_thread = 1,
  stable = FALSE,
  sparse = FALSE,
  med = FALSE,
  lpl = FALSE,
  mcmc = TRUE,
  use_fit = TRUE,
  verbose = FALSE,
  ...
)

# S3 method for class 'svmod'
forecast_expand(
  object,
  n_ahead,
  y_test,
  level = 0.05,
  newxreg = NULL,
  num_thread = 1,
  use_sv = TRUE,
  stable = FALSE,
  sparse = FALSE,
  med = FALSE,
  lpl = FALSE,
  mcmc = TRUE,
  use_fit = TRUE,
  verbose = FALSE,
  ...
)
```

## Arguments

- object:

  Model object

- n_ahead:

  Step to forecast in rolling window scheme

- y_test:

  Test data to be compared. Use [`divide_ts()`](divide_ts.md) if you
  don't have separate evaluation dataset. Use numeric, the length of
  test period, if `object` is the full sample result. Numeric `y_test`
  will automatically splits `object$y`. In this case, `use_fit` is
  forced to become `FALSE`.

- level:

  Specify alpha of confidence interval level 100(1 - alpha) percentage.
  By default, .05.

- newxreg:

  New values for exogenous variables. Should have the same row numbers
  as `y_test`.

- num_thread:

  **\[experimental\]** Number of threads

- ...:

  Additional arguments.

- use_fit:

  **\[experimental\]** Use `object` result for the first window. By
  default, `TRUE`.

- stable:

  **\[experimental\]** Filter only stable coefficient draws in MCMC
  records.

- sparse:

  **\[experimental\]** Apply restriction. By default, `FALSE`.

- med:

  **\[experimental\]** If `TRUE`, use median of forecast draws instead
  of mean (default).

- lpl:

  **\[experimental\]** Compute log-predictive likelihood (LPL). By
  default, `FALSE`.

- mcmc:

  **\[experimental\]** If `TRUE`, run new MCMC in new windows. By
  default, `TRUE`.

- verbose:

  Print the progress bar in the console. By default, `FALSE`.

- use_sv:

  Use SV term

## Value

`predbvhar_expand` [class](https://rdrr.io/r/base/class.html)

## Details

Expanding windows forecasting fixes the starting period. It moves the
window ahead and forecast h-ahead in `y_test` set.

## References

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles
and practice* (3rd ed.). OTEXTS. <https://otexts.com/fpp3/>

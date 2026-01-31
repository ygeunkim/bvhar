# Dynamic Spillover

This function gives connectedness table with h-step ahead normalized
spillover index (a.k.a. variance shares).

## Usage

``` r
dynamic_spillover(object, n_ahead = 10L, ...)

# S3 method for class 'bvhardynsp'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvhardynsp'
knit_print(x, ...)

# S3 method for class 'olsmod'
dynamic_spillover(object, n_ahead = 10L, window, num_thread = 1, ...)

# S3 method for class 'normaliw'
dynamic_spillover(
  object,
  n_ahead = 10L,
  window,
  num_iter = 1000L,
  num_burn = floor(num_iter/2),
  thinning = 1,
  num_thread = 1,
  ...
)

# S3 method for class 'ldltmod'
dynamic_spillover(
  object,
  n_ahead = 10L,
  window,
  level = 0.05,
  sparse = FALSE,
  num_thread = 1,
  ...
)

# S3 method for class 'svmod'
dynamic_spillover(
  object,
  n_ahead = 10L,
  level = 0.05,
  sparse = FALSE,
  num_thread = 1,
  ...
)
```

## Arguments

- object:

  Model object

- n_ahead:

  step to forecast. By default, 10.

- ...:

  not used

- x:

  `bvhardynsp` object

- digits:

  digit option to print

- window:

  Window size

- num_thread:

  **\[experimental\]** Number of threads

- num_iter:

  Number to sample MNIW distribution

- num_burn:

  Number of burn-in

- thinning:

  Thinning every thinning-th iteration

- level:

  Specify alpha of confidence interval level 100(1 - alpha) percentage.
  By default, .05.

- sparse:

  **\[experimental\]** Apply restriction. By default, `FALSE`.

## References

Diebold, F. X., & Yilmaz, K. (2012). *Better to give than to receive:
Predictive directional measurement of volatility spillovers*.
International Journal of forecasting, 28(1), 57-66.

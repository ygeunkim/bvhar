# h-step ahead Normalized Spillover

This function gives connectedness table with h-step ahead normalized
spillover index (a.k.a. variance shares).

## Usage

``` r
spillover(object, n_ahead = 10L, ...)

# S3 method for class 'bvharspillover'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'bvharspillover'
knit_print(x, ...)

# S3 method for class 'olsmod'
spillover(object, n_ahead = 10L, ...)

# S3 method for class 'normaliw'
spillover(
  object,
  n_ahead = 10L,
  num_iter = 5000L,
  num_burn = floor(num_iter/2),
  thinning = 1L,
  ...
)

# S3 method for class 'bvarldlt'
spillover(object, n_ahead = 10L, level = 0.05, sparse = FALSE, ...)

# S3 method for class 'bvharldlt'
spillover(object, n_ahead = 10L, level = 0.05, sparse = FALSE, ...)
```

## Arguments

- object:

  Model object

- n_ahead:

  step to forecast. By default, 10.

- ...:

  not used

- x:

  `bvharspillover` object

- digits:

  digit option to print

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

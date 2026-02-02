# Choose the Best VAR based on Information Criteria

This function computes AIC, FPE, BIC, and HQ up to p = `lag_max` of VAR
model.

## Usage

``` r
choose_var(y, lag_max = 5, include_mean = TRUE, parallel = FALSE)
```

## Arguments

- y:

  Time series data of which columns indicate the variables

- lag_max:

  Maximum Var lag to explore (default = 5)

- include_mean:

  Add constant term (Default: `TRUE`) or not (`FALSE`)

- parallel:

  Parallel computation using
  [`foreach::foreach()`](https://rdrr.io/pkg/foreach/man/foreach.html)?
  By default, `FALSE`.

## Value

Minimum order and information criteria values

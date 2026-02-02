# Coefficient Matrix of Multivariate Time Series Models

By defining [`stats::coef()`](https://rdrr.io/r/stats/coef.html) for
each model, this function returns coefficient matrix estimates.

## Usage

``` r
# S3 method for class 'varlse'
coef(object, ...)

# S3 method for class 'vharlse'
coef(object, ...)

# S3 method for class 'bvarmn'
coef(object, ...)

# S3 method for class 'bvarflat'
coef(object, ...)

# S3 method for class 'bvharmn'
coef(object, ...)

# S3 method for class 'bvharsp'
coef(object, ...)

# S3 method for class 'summary.bvharsp'
coef(object, ...)
```

## Arguments

- object:

  Model object

- ...:

  not used

## Value

[matrix](https://rdrr.io/r/base/matrix.html) object with appropriate
dimension.

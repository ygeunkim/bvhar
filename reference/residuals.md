# Residual Matrix from Multivariate Time Series Models

By defining
[`stats::residuals()`](https://rdrr.io/r/stats/residuals.html) for each
model, this function returns residual.

## Usage

``` r
# S3 method for class 'varlse'
residuals(object, ...)

# S3 method for class 'vharlse'
residuals(object, ...)

# S3 method for class 'bvarmn'
residuals(object, ...)

# S3 method for class 'bvarflat'
residuals(object, ...)

# S3 method for class 'bvharmn'
residuals(object, ...)
```

## Arguments

- object:

  Model object

- ...:

  not used

## Value

[matrix](https://rdrr.io/r/base/matrix.html) object.

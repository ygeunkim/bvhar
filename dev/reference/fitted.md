# Fitted Matrix from Multivariate Time Series Models

By defining
[`stats::fitted()`](https://rdrr.io/r/stats/fitted.values.html) for each
model, this function returns fitted matrix.

## Usage

``` r
# S3 method for class 'varlse'
fitted(object, ...)

# S3 method for class 'vharlse'
fitted(object, ...)

# S3 method for class 'bvarmn'
fitted(object, ...)

# S3 method for class 'bvarflat'
fitted(object, ...)

# S3 method for class 'bvharmn'
fitted(object, ...)
```

## Arguments

- object:

  Model object

- ...:

  not used

## Value

[matrix](https://rdrr.io/r/base/matrix.html) object.

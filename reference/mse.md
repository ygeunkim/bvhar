# Evaluate the Model Based on MSE (Mean Square Error)

This function computes MSE given prediction result versus evaluation
set.

## Usage

``` r
mse(x, y, ...)

# S3 method for class 'predbvhar'
mse(x, y, ...)

# S3 method for class 'bvharcv'
mse(x, y, ...)
```

## Arguments

- x:

  Forecasting object

- y:

  Test data to be compared. should be the same format with the train
  data.

- ...:

  not used

## Value

MSE vector corresponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. Then \$\$MSE = mean(e_t^2)\$\$ MSE is
the most used accuracy measure.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

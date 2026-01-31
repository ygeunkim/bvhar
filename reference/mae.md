# Evaluate the Model Based on MAE (Mean Absolute Error)

This function computes MAE given prediction result versus evaluation
set.

## Usage

``` r
mae(x, y, ...)

# S3 method for class 'predbvhar'
mae(x, y, ...)

# S3 method for class 'bvharcv'
mae(x, y, ...)
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

MAE vector corressponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. MAE is defined by

\$\$MSE = mean(\lvert e_t \rvert)\$\$

Some researchers prefer MAE to MSE because it is less sensitive to
outliers.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

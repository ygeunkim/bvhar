# Evaluate the Model Based on MAPE (Mean Absolute Percentage Error)

This function computes MAPE given prediction result versus evaluation
set.

## Usage

``` r
mape(x, y, ...)

# S3 method for class 'predbvhar'
mape(x, y, ...)

# S3 method for class 'bvharcv'
mape(x, y, ...)
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

MAPE vector corresponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. Percentage error is defined by \\p_t =
100 e_t / Y_t\\ (100 can be omitted since comparison is the focus).

\$\$MAPE = mean(\lvert p_t \rvert)\$\$

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

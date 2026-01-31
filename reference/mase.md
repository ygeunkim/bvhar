# Evaluate the Model Based on MASE (Mean Absolute Scaled Error)

This function computes MASE given prediction result versus evaluation
set.

## Usage

``` r
mase(x, y, ...)

# S3 method for class 'predbvhar'
mase(x, y, ...)

# S3 method for class 'bvharcv'
mase(x, y, ...)
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

MASE vector corresponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. Scaled error is defined by \$\$q_t =
\frac{e_t}{\sum\_{i = 2}^{n} \lvert Y_i - Y\_{i - 1} \rvert / (n -
1)}\$\$ so that the error can be free of the data scale. Then

\$\$MASE = mean(\lvert q_t \rvert)\$\$

Here, \\Y_i\\ are the points in the sample, i.e. errors are scaled by
the in-sample mean absolute error (\\mean(\lvert e_t \rvert)\\) from the
naive random walk forecasting.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

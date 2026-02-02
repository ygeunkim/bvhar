# Evaluate the Model Based on MRAE (Mean Relative Absolute Error)

This function computes MRAE given prediction result versus evaluation
set.

## Usage

``` r
mrae(x, pred_bench, y, ...)

# S3 method for class 'predbvhar'
mrae(x, pred_bench, y, ...)

# S3 method for class 'bvharcv'
mrae(x, pred_bench, y, ...)
```

## Arguments

- x:

  Forecasting object to use

- pred_bench:

  The same forecasting object from benchmark model

- y:

  Test data to be compared. should be the same format with the train
  data.

- ...:

  not used

## Value

MRAE vector corresponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. MRAE implements benchmark model as
scaling method. Relative error is defined by \$\$r_t =
\frac{e_t}{e_t^{\ast}}\$\$ where \\e_t^\ast\\ is the error from the
benchmark method. Then

\$\$MRAE = mean(\lvert r_t \rvert)\$\$

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

# Evaluate the Model Based on RMASE (Relative MASE)

This function computes RMASE given prediction result versus evaluation
set.

## Usage

``` r
rmase(x, pred_bench, y, ...)

# S3 method for class 'predbvhar'
rmase(x, pred_bench, y, ...)

# S3 method for class 'bvharcv'
rmase(x, pred_bench, y, ...)
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

RMASE vector corresponding to each variable.

## Details

RMASE is the ratio of MAPE of given model and the benchmark one. Let
\\MASE_b\\ be the MAPE of the benchmark model. Then

\$\$RMASE = \frac{mean(MASE)}{mean(MASE_b)}\$\$

where \\MASE\\ is the MASE of our model.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

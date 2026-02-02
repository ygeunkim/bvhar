# Evaluate the Model Based on RMAPE (Relative MAPE)

This function computes RMAPE given prediction result versus evaluation
set.

## Usage

``` r
rmape(x, pred_bench, y, ...)

# S3 method for class 'predbvhar'
rmape(x, pred_bench, y, ...)

# S3 method for class 'bvharcv'
rmape(x, pred_bench, y, ...)
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

RMAPE vector corresponding to each variable.

## Details

RMAPE is the ratio of MAPE of given model and the benchmark one. Let
\\MAPE_b\\ be the MAPE of the benchmark model. Then

\$\$RMAPE = \frac{mean(MAPE)}{mean(MAPE_b)}\$\$

where \\MAPE\\ is the MAPE of our model.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

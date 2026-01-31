# Evaluate the Model Based on RelMAE (Relative MAE)

This function computes RelMAE given prediction result versus evaluation
set.

## Usage

``` r
relmae(x, pred_bench, y, ...)

# S3 method for class 'predbvhar'
relmae(x, pred_bench, y, ...)

# S3 method for class 'bvharcv'
relmae(x, pred_bench, y, ...)
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

RelMAE vector corresponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. RelMAE implements MAE of benchmark model
as relative measures. Let \\MAE_b\\ be the MAE of the benchmark model.
Then

\$\$RelMAE = \frac{MAE}{MAE_b}\$\$

where \\MAE\\ is the MAE of our model.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

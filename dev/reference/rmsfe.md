# Evaluate the Model Based on RMSFE

This function computes RMSFE (Mean Squared Forecast Error Relative to
the Benchmark)

## Usage

``` r
rmsfe(x, pred_bench, y, ...)

# S3 method for class 'predbvhar'
rmsfe(x, pred_bench, y, ...)

# S3 method for class 'bvharcv'
rmsfe(x, pred_bench, y, ...)
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

RMSFE vector corresponding to each variable.

## Details

Let \\e_t = y_t - \hat{y}\_t\\. RMSFE is the ratio of L2 norm of \\e_t\\
from forecasting object and from benchmark model.

\$\$RMSFE = \frac{sum(\lVert e_t \rVert)}{sum(\lVert e_t^{(b)}
\rVert)}\$\$

where \\e_t^{(b)}\\ is the error from the benchmark model.

## References

Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of
forecast accuracy*. International Journal of Forecasting, 22(4),
679-688.

Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector
auto regressions*. Journal of Applied Econometrics, 25(1).

Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional
Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal
of the American Statistical Association, 114(526).

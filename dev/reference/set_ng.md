# Normal-Gamma Hyperparameter for Coefficients and Contemporaneous Coefficients

**\[experimental\]** Set NG hyperparameters for VAR or VHAR coefficient
and contemporaneous coefficient.

## Usage

``` r
set_ng(
  shape_sd = 0.01,
  group_shape = 0.01,
  group_scale = 0.01,
  global_shape = 0.01,
  global_scale = 0.01
)

# S3 method for class 'ngspec'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.ngspec(x)
```

## Arguments

- shape_sd:

  Standard deviation used in MH of Gamma shape

- group_shape:

  Inverse gamma prior shape for coefficient group shrinkage

- group_scale:

  Inverse gamma prior scale for coefficient group shrinkage

- global_shape:

  Inverse gamma prior shape for coefficient global shrinkage

- global_scale:

  Inverse gamma prior scale for coefficient global shrinkage

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Value

`ngspec` object

## References

Chan, J. C. C. (2021). *Minnesota-type adaptive hierarchical priors for
large Bayesian VARs*. International Journal of Forecasting, 37(3),
1212-1226.

Huber, F., & Feldkircher, M. (2019). *Adaptive Shrinkage in Bayesian
Vector Autoregressive Models*. Journal of Business & Economic
Statistics, 37(1), 27-39.

Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage
and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4),
230-354.

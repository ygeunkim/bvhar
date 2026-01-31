# Evaluate the Sparsity Estimation Based on Precision

This function computes precision for sparse element of the true
coefficients given threshold.

## Usage

``` r
conf_prec(x, y, ...)

# S3 method for class 'summary.bvharsp'
conf_prec(x, y, truth_thr = 0, ...)
```

## Arguments

- x:

  `summary.bvharsp` object.

- y:

  True inclusion variable.

- ...:

  not used

- truth_thr:

  Threshold value when using non-sparse true coefficient matrix. By
  default, `0` for sparse matrix.

## Value

Precision value in confusion table

## Details

If the element of the estimate \\\hat\Phi\\ is smaller than some
threshold, it is treated to be zero. Then the precision is computed by
\$\$precision = \frac{TP}{TP + FP}\$\$ where TP is true positive, and FP
is false positive.

## References

Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior
consistency under global-local shrinkage priors. Journal of Multivariate
Analysis, 167, 157-170.

## See also

[`confusion()`](confusion.md)

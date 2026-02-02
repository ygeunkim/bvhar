# Evaluate the Sparsity Estimation Based on FNR

This function computes false negative rate (FNR) for sparse element of
the true coefficients given threshold.

## Usage

``` r
conf_fnr(x, y, ...)

# S3 method for class 'summary.bvharsp'
conf_fnr(x, y, truth_thr = 0, ...)
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

FNR value in confusion table

## Details

False negative rate (FNR) is computed by \$\$FNR = \frac{FN}{TP +
FN}\$\$ where TP is true positive, and FN is false negative.

## References

Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior
consistency under global-local shrinkage priors. Journal of Multivariate
Analysis, 167, 157-170.

## See also

[`confusion()`](confusion.md)

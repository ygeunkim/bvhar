# Evaluate the Sparsity Estimation Based on FDR

This function computes false discovery rate (FDR) for sparse element of
the true coefficients given threshold.

## Usage

``` r
conf_fdr(x, y, ...)

# S3 method for class 'summary.bvharsp'
conf_fdr(x, y, truth_thr = 0, ...)
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

FDR value in confusion table

## Details

When using this function, the true coefficient matrix \\\Phi\\ should be
sparse. False discovery rate (FDR) is computed by \$\$FDR =
\frac{FP}{TP + FP}\$\$ where TP is true positive, and FP is false
positive.

## References

Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior
consistency under global-local shrinkage priors. Journal of Multivariate
Analysis, 167, 157-170.

## See also

[`confusion()`](confusion.md)

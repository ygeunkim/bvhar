# Evaluate the Sparsity Estimation Based on Recall

This function computes recall for sparse element of the true
coefficients given threshold.

## Usage

``` r
conf_recall(x, y, ...)

# S3 method for class 'summary.bvharsp'
conf_recall(x, y, truth_thr = 0L, ...)
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

Recall value in confusion table

## Details

Precision is computed by \$\$recall = \frac{TP}{TP + FN}\$\$ where TP is
true positive, and FN is false negative.

## References

Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior
consistency under global-local shrinkage priors. Journal of Multivariate
Analysis, 167, 157-170.

## See also

[`confusion()`](confusion.md)

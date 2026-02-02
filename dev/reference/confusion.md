# Evaluate the Sparsity Estimation Based on Confusion Matrix

This function computes FDR (false discovery rate) and FNR (false
negative rate) for sparse element of the true coefficients given
threshold.

## Usage

``` r
confusion(x, y, ...)

# S3 method for class 'summary.bvharsp'
confusion(x, y, truth_thr = 0, ...)
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

Confusion table as following.

|               |              |              |
|---------------|--------------|--------------|
| True-estimate | Positive (0) | Negative (1) |
| Positive (0)  | TP           | FN           |
| Negative (1)  | FP           | TN           |

## Details

When using this function, the true coefficient matrix \\\Phi\\ should be
sparse.

In this confusion matrix, positive (0) means sparsity. FP is false
positive, and TP is true positive. FN is false negative, and FN is false
negative.

## References

Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior
consistency under global-local shrinkage priors. Journal of Multivariate
Analysis, 167, 157-170.

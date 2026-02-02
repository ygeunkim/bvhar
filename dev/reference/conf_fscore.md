# Evaluate the Sparsity Estimation Based on F1 Score

This function computes F1 score for sparse element of the true
coefficients given threshold.

## Usage

``` r
conf_fscore(x, y, ...)

# S3 method for class 'summary.bvharsp'
conf_fscore(x, y, truth_thr = 0, ...)
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

F1 score in confusion table

## Details

The F1 score is computed by \$\$F_1 = \frac{2 precision \times
recall}{precision + recall}\$\$

## See also

[`confusion()`](confusion.md)

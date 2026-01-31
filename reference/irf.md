# Impulse Response Analysis

Computes responses to impulses or orthogonal impulses

## Usage

``` r
# S3 method for class 'varlse'
irf(object, lag_max = 10, orthogonal = TRUE, impulse_var, response_var, ...)

# S3 method for class 'vharlse'
irf(object, lag_max = 10, orthogonal = TRUE, impulse_var, response_var, ...)

# S3 method for class 'bvharirf'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

irf(object, lag_max, orthogonal, impulse_var, response_var, ...)

is.bvharirf(x)

# S3 method for class 'bvharirf'
knit_print(x, ...)
```

## Arguments

- object:

  Model object

- lag_max:

  Maximum lag to investigate the impulse responses (By default, `10`)

- orthogonal:

  Orthogonal impulses (`TRUE`) or just impulses (`FALSE`)

- impulse_var:

  Impulse variables character vector. If not specified, use every
  variable.

- response_var:

  Response variables character vector. If not specified, use every
  variable.

- ...:

  not used

- x:

  Any object

- digits:

  digit option to print

## Value

`bvharirf` [class](https://rdrr.io/r/base/class.html)

## Responses to forecast errors

If `orthogonal = FALSE`, the function gives \\W_j\\ VMA representation
of the process such that \$\$Y_t = \sum\_{j = 0}^\infty W_j
\epsilon\_{t - j}\$\$

## Responses to orthogonal impulses

If `orthogonal = TRUE`, it gives orthogonalized VMA representation
\$\$\Theta\$\$. Based on variance decomposition (Cholesky decomposition)
\$\$\Sigma = P P^T\$\$ where \\P\\ is lower triangular matrix, impulse
response analysis if performed under MA representation \$\$y_t =
\sum\_{i = 0}^\infty \Theta_i v\_{t - i}\$\$ Here, \$\$\Theta_i = W_i
P\$\$ and \\v_t = P^{-1} \epsilon_t\\ are orthogonal.

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

## See also

[`VARtoVMA()`](VARtoVMA.md)

[`VHARtoVMA()`](VHARtoVMA.md)

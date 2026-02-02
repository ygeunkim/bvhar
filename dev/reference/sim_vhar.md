# Generate Multivariate Time Series Process Following VAR(p)

This function generates multivariate time series dataset that follows
VAR(p).

## Usage

``` r
sim_vhar(
  num_sim,
  num_burn,
  vhar_coef,
  week = 5L,
  month = 22L,
  sig_error = diag(ncol(vhar_coef)),
  init = matrix(0L, nrow = month, ncol = ncol(vhar_coef)),
  method = c("eigen", "chol"),
  process = c("gaussian", "student"),
  t_param = 5
)
```

## Arguments

- num_sim:

  Number to generated process

- num_burn:

  Number of burn-in

- vhar_coef:

  VAR coefficient. The format should be the same as the output of
  [`coef()`](coef.md) from [`var_lm()`](var_lm.md)

- week:

  Weekly order of VHAR. By default, `5`.

- month:

  Weekly order of VHAR. By default, `22`.

- sig_error:

  Variance matrix of the error term. By default, `diag(dim)`.

- init:

  Initial y1, ..., yp matrix to simulate VAR model. Try
  `matrix(0L, nrow = month, ncol = dim)`.

- method:

  Method to compute \\\Sigma^{1/2}\\. Choose between `eigen` (spectral
  decomposition) and `chol` (cholesky decomposition). By default,
  `eigen`.

- process:

  Process to generate error term. `gaussian`: Normal distribution
  (default) or `student`: Multivariate t-distribution.

- t_param:

  **\[experimental\]** argument for MVT, e.g. DF: 5.

## Value

T x k matrix

## Details

Let \\M\\ be the month order, e.g. \\M = 22\\.

1.  Generate \\\epsilon_1, \epsilon_n \sim N(0, \Sigma)\\

2.  For i = 1, ... n, \$\$y\_{M + i} = (y\_{M + i - 1}^T, \ldots, y_i^T,
    1)^T C\_{HAR}^T \Phi + \epsilon_i\$\$

3.  Then the output is \\(y\_{M + 1}, \ldots, y\_{n + M})^T\\

4.  For i = 1, ... n, \$\$y\_{p + i} = (y\_{p + i - 1}^T, \ldots, y_i^T,
    1)^T B + \epsilon_i\$\$

5.  Then the output is \\(y\_{p + 1}, \ldots, y\_{n + p})^T\\

Initial values might be set to be zero vector or \\(I_m - A_1 - \cdots -
A_p)^{-1} c\\.

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

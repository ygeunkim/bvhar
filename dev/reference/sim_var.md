# Generate Multivariate Time Series Process Following VAR(p)

This function generates multivariate time series dataset that follows
VAR(p).

## Usage

``` r
sim_var(
  num_sim,
  num_burn,
  var_coef,
  var_lag,
  sig_error = diag(ncol(var_coef)),
  init = matrix(0L, nrow = var_lag, ncol = ncol(var_coef)),
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

- var_coef:

  VAR coefficient. The format should be the same as the output of
  [`coef()`](coef.md) from [`var_lm()`](var_lm.md)

- var_lag:

  Lag of VAR

- sig_error:

  Variance matrix of the error term. By default, `diag(dim)`.

- init:

  Initial y1, ..., yp matrix to simulate VAR model. Try
  `matrix(0L, nrow = var_lag, ncol = dim)`.

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

1.  Generate \\\epsilon_1, \epsilon_n \sim N(0, \Sigma)\\

2.  For i = 1, ... n, \$\$y\_{p + i} = (y\_{p + i - 1}^T, \ldots, y_i^T,
    1)^T B + \epsilon_i\$\$

3.  Then the output is \\(y\_{p + 1}, \ldots, y\_{n + p})^T\\

Initial values might be set to be zero vector or \\(I_m - A_1 - \cdots -
A_p)^{-1} c\\.

## References

Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
Analysis*. Springer Publishing.

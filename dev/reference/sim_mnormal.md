# Generate Multivariate Normal Random Vector

This function samples n x muti-dimensional normal random matrix.

## Usage

``` r
sim_mnormal(
  num_sim,
  mu = rep(0, 5),
  sig = diag(5),
  method = c("eigen", "chol")
)
```

## Arguments

- num_sim:

  Number to generate process

- mu:

  Mean vector

- sig:

  Variance matrix

- method:

  Method to compute \\\Sigma^{1/2}\\. Choose between `eigen` (spectral
  decomposition) and `chol` (cholesky decomposition). By default,
  `eigen`.

## Value

T x k matrix

## Details

Consider \\x_1, \ldots, x_n \sim N_m (\mu, \Sigma)\\.

1.  Lower triangular Cholesky decomposition: \\\Sigma = L L^T\\

2.  Standard normal generation: \\Z\_{i1}, Z\_{in} \stackrel{iid}{\sim}
    N(0, 1)\\

3.  \\Z_i = (Z\_{i1}, \ldots, Z\_{in})^T\\

4.  \\X_i = L Z_i + \mu\\

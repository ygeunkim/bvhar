# Generate Normal-IW Random Family

This function samples normal inverse-wishart matrices.

## Usage

``` r
sim_mniw(num_sim, mat_mean, mat_scale_u, mat_scale, shape, u_prec = FALSE)
```

## Arguments

- num_sim:

  Number to generate

- mat_mean:

  Mean matrix of MN

- mat_scale_u:

  First scale matrix of MN

- mat_scale:

  Scale matrix of IW

- shape:

  Shape of IW

- u_prec:

  If `TRUE`, use `mat_scale_u` as its inverse. By default, `FALSE`.

## Details

Consider \\(Y_i, \Sigma_i) \sim MIW(M, U, \Psi, \nu)\\.

1.  Generate upper triangular factor of \\\Sigma_i = C_i C_i^T\\ in the
    upper triangular Bartlett decomposition.

2.  Standard normal generation: n x k matrix \\Z_i = \[z\_{ij} \sim N(0,
    1)\]\\ in row-wise direction.

3.  Lower triangular Cholesky decomposition: \\U = P P^T\\

4.  \\A_i = M + P Z_i C_i^T\\

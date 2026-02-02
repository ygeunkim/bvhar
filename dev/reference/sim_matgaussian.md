# Generate Matrix Normal Random Matrix

This function samples one matrix gaussian matrix.

## Usage

``` r
sim_matgaussian(mat_mean, mat_scale_u, mat_scale_v, u_prec)
```

## Arguments

- mat_mean:

  Mean matrix

- mat_scale_u:

  First scale matrix

- mat_scale_v:

  Second scale matrix

- u_prec:

  If `TRUE`, use `mat_scale_u` as its inverse.

## Value

One n x k matrix following MN distribution.

## Details

Consider n x k matrix \\Y_1, \ldots, Y_n \sim MN(M, U, V)\\ where M is n
x k, U is n x n, and V is k x k.

1.  Lower triangular Cholesky decomposition: \\U = P P^T\\ and \\V = L
    L^T\\

2.  Standard normal generation: s x m matrix \\Z_i = \[z\_{ij} \sim N(0,
    1)\]\\ in row-wise direction.

3.  \\Y_i = M + P Z_i L^T\\

This function only generates one matrix, i.e. \\Y_1\\.

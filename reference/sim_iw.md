# Generate Inverse-Wishart Random Matrix

This function samples one matrix IW matrix.

## Usage

``` r
sim_iw(mat_scale, shape)
```

## Arguments

- mat_scale:

  Scale matrix

- shape:

  Shape

## Value

One k x k matrix following IW distribution

## Details

Consider \\\Sigma \sim IW(\Psi, \nu)\\.

1.  Upper triangular Bartlett decomposition: k x k matrix \\Q =
    \[q\_{ij}\]\\ upper triangular with

    1.  \\q\_{ii}^2 \chi\_{\nu - i + 1}^2\\

    2.  \\q\_{ij} \sim N(0, 1)\\ with i \< j (upper triangular)

2.  Lower triangular Cholesky decomposition: \\\Psi = L L^T\\

3.  \\A = L (Q^{-1})^T\\

4.  \\\Sigma = A A^T \sim IW(\Psi, \nu)\\

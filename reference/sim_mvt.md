# Generate Multivariate t Random Vector

This function samples n x multi-dimensional t-random matrix.

## Usage

``` r
sim_mvt(num_sim, df, mu, sig, method = c("eigen", "chol"))
```

## Arguments

- num_sim:

  Number to generate process.

- df:

  Degrees of freedom.

- mu:

  Location vector

- sig:

  Scale matrix.

- method:

  Method to compute \\\Sigma^{1/2}\\. Choose between `eigen` (spectral
  decomposition) and `chol` (cholesky decomposition). By default,
  `eigen`.

## Value

T x k matrix

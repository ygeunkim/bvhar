# Evaluate the Estimation Based on Spectral Norm Error

This function computes estimation error given estimated model and true
coefficient.

## Usage

``` r
spne(x, y, ...)

# S3 method for class 'bvharsp'
spne(x, y, ...)
```

## Arguments

- x:

  Estimated model.

- y:

  Coefficient matrix to be compared.

- ...:

  not used

## Value

Spectral norm value

## Details

Let \\\lVert \cdot \rVert_2\\ be the spectral norm of a matrix, let
\\\hat{\Phi}\\ be the estimates, and let \\\Phi\\ be the true
coefficients matrix. Then the function computes estimation error by
\$\$\lVert \hat{\Phi} - \Phi \rVert_2\$\$

## References

Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional
Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal
of the American Statistical Association, 114(526).

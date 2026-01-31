# Evaluate the Estimation Based on Frobenius Norm

This function computes estimation error given estimated model and true
coefficient.

## Usage

``` r
fromse(x, y, ...)

# S3 method for class 'bvharsp'
fromse(x, y, ...)
```

## Arguments

- x:

  Estimated model.

- y:

  Coefficient matrix to be compared.

- ...:

  not used

## Value

Frobenius norm value

## Details

Consider the Frobenius Norm \\\lVert \cdot \rVert_F\\. let
\\\hat{\Phi}\\ be nrow x k the estimates, and let \\\Phi\\ be the true
coefficients matrix. Then the function computes estimation error by
\$\$MSE = 100 \frac{\lVert \hat{\Phi} - \Phi \rVert_F}{nrow \times
k}\$\$

## References

Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior
consistency under global-local shrinkage priors. Journal of Multivariate
Analysis, 167, 157-170.

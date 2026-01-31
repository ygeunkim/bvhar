# Dirichlet-Laplace Hyperparameter for Coefficients and Contemporaneous Coefficients

**\[experimental\]** Set DL hyperparameters for VAR or VHAR coefficient
and contemporaneous coefficient.

## Usage

``` r
set_dl(dir_grid = 100L, shape = 0.01, scale = 0.01)

# S3 method for class 'dlspec'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

is.dlspec(x)
```

## Arguments

- dir_grid:

  Griddy gibbs grid size for Dirichlet hyperparameter

- shape:

  Inverse Gamma shape

- scale:

  Inverse Gamma scale

- x:

  Any object

- digits:

  digit option to print

- ...:

  not used

## Value

`dlspec` object

## References

Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015).
*Dirichlet-Laplace Priors for Optimal Shrinkage*. Journal of the
American Statistical Association, 110(512), 1479-1490.

Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage
and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4),
230-354.

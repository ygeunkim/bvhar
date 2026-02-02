# Generalized Double Pareto Shrinkage Hyperparameters for Coefficients and Contemporaneous Coefficients

**\[experimental\]** Set GDP hyperparameters for VAR or VHAR coefficient
and contemporaneous coefficient.

## Usage

``` r
set_gdp(shape_grid = 100L, rate_grid = 100L)

is.gdpspec(x)
```

## Arguments

- shape_grid:

  Griddy gibbs grid size for Gamma shape hyperparameter

- rate_grid:

  Griddy gibbs grid size for Gamma rate hyperparameter

- x:

  Any object

## Value

`gdpspec` object

## References

Armagan, A., Dunson, D. B., & Lee, J. (2013). *GENERALIZED DOUBLE PARETO
SHRINKAGE*. Statistica Sinica, 23(1), 119–143.

Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage
and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4),
230-354.

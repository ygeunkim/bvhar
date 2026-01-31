# Summarizing BVAR and BVHAR with Shrinkage Priors

Conduct variable selection.

## Usage

``` r
# S3 method for class 'summary.bvharsp'
print(x, digits = max(3L, getOption("digits") - 3L), ...)

# S3 method for class 'summary.bvharsp'
knit_print(x, ...)

# S3 method for class 'ssvsmod'
summary(object, method = c("pip", "ci"), threshold = 0.5, level = 0.05, ...)

# S3 method for class 'hsmod'
summary(object, method = c("pip", "ci"), threshold = 0.5, level = 0.05, ...)

# S3 method for class 'ngmod'
summary(object, level = 0.05, ...)
```

## Arguments

- x:

  `summary.bvharsp` object

- digits:

  digit option to print

- ...:

  not used

- object:

  Model fit

- method:

  Use PIP (`pip`) or credible interval (`ci`).

- threshold:

  Threshold for posterior inclusion probability

- level:

  Specify alpha of credible interval level 100(1 - alpha) percentage. By
  default, `.05`.

## Value

`summary.ssvsmod` object

`hsmod` object

`ngmod` object

## References

George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs
Sampling*. Journal of the American Statistical Association, 88(423),
881-889.

George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for
VAR model restrictions*. Journal of Econometrics, 142(1), 553-580.

Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series
Methods for Empirical Macroeconomics*. Foundations and Trends® in
Econometrics, 3(4), 267-358.

O’Hara, R. B., & Sillanpää, M. J. (2009). *A review of Bayesian variable
selection methods: what, how and which*. Bayesian Analysis, 4(1),
85-117.

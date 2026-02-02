# Generate Minnesota BVAR Parameters

This function generates parameters of BVAR with Minnesota prior.

## Usage

``` r
sim_mncoef(p, bayes_spec = set_bvar(), full = TRUE)
```

## Arguments

- p:

  VAR lag

- bayes_spec:

  A BVAR model specification by [`set_bvar()`](set_bvar.md).

- full:

  Generate variance matrix from IW (default: `TRUE`) or not (`FALSE`)?

## Value

List with the following component.

- coefficients:

  BVAR coefficient (MN)

- covmat:

  BVAR variance (IW or diagonal matrix of `sigma` of `bayes_spec`)

## Details

Implementing dummy observation constructions, Bańbura et al. (2010) sets
Normal-IW prior. \$\$A \mid \Sigma_e \sim MN(A_0, \Omega_0,
\Sigma_e)\$\$ \$\$\Sigma_e \sim IW(S_0, \alpha_0)\$\$ If `full = FALSE`,
the result of \\\Sigma_e\\ is the same as input (`diag(sigma)`).

## References

Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector
auto regressions*. Journal of Applied Econometrics, 25(1).

Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector
Autoregression*. Handbook of Economic Forecasting, 2, 791-897.

Litterman, R. B. (1986). *Forecasting with Bayesian Vector
Autoregressions: Five Years of Experience*. Journal of Business &
Economic Statistics, 4(1), 25.

## See also

- [`set_bvar()`](set_bvar.md) to specify the hyperparameters of
  Minnesota prior.

## Examples

``` r
# Generate (A, Sigma)
# BVAR(p = 2)
# sigma: 1, 1, 1
# lambda: .1
# delta: .1, .1, .1
# epsilon: 1e-04
set.seed(1)
sim_mncoef(
  p = 2,
  bayes_spec = set_bvar(
    sigma = rep(1, 3),
    lambda = .1,
    delta = rep(.1, 3),
    eps = 1e-04
  ),
  full = TRUE
)
#> $coefficients
#>               [,1]         [,2]        [,3]
#> [1,]  0.1760184710 -0.016532893 -0.06514725
#> [2,]  0.1556541344  0.036163539 -0.16746242
#> [3,] -0.2280271782  0.157961996  0.34810835
#> [4,] -0.0008334809  0.018352564  0.05185982
#> [5,]  0.0305742650  0.002010504  0.02197815
#> [6,]  0.0038386336 -0.039734185 -0.02366305
#> 
#> $covmat
#>            [,1]       [,2]       [,3]
#> [1,]  1.0600918 -0.5356457 -0.9412103
#> [2,] -0.5356457  0.4150287  0.6410763
#> [3,] -0.9412103  0.6410763  1.5794713
#> 
```

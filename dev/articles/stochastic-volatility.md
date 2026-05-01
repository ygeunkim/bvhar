# Stochastic Volatility Models

``` r

library(bvhar)
```

``` r

etf <- etf_vix[1:55, 1:3]
# Split-------------------------------
h <- 5
etf_eval <- divide_ts(etf, h)
etf_train <- etf_eval$train
etf_test <- etf_eval$test
```

## Models with Stochastic Volatilities

By specifying `cov_spec = set_sv()`,
[`var_bayes()`](../reference/var_bayes.md) and
[`vhar_bayes()`](../reference/vhar_bayes.md) fits VAR-SV and VHAR-SV
with shrinkage priors, respectively.

- Three different prior for innovation covariance, and specify through
  `coef_spec`
  - Minneosta prior
    - BVAR: [`set_bvar()`](../reference/set_bvar.md)
    - BVHAR: [`set_bvhar()`](../reference/set_bvar.md) and
      [`set_weight_bvhar()`](../reference/set_bvar.md)
  - SSVS prior: [`set_ssvs()`](../reference/set_ssvs.md)
  - Horseshoe prior: [`set_horseshoe()`](../reference/set_horseshoe.md)
  - NG prior: [`set_ng()`](../reference/set_ng.md)
  - DL prior: [`set_dl()`](../reference/set_dl.md)
- `sv_spec`: prior settings for SV,
  [`set_sv()`](../reference/set_ldlt.md)
- `intercept`: prior for constant term,
  [`set_intercept()`](../reference/set_intercept.md)

``` r

set_sv()
#> Model Specification for SV with Cholesky Prior
#> 
#> Parameters: Contemporaneous coefficients, State variance, Initial state
#> Prior: Cholesky
#> ========================================================
#> Setting for 'shape':
#> [1]  rep(3, dim)
#> 
#> Setting for 'scale':
#> [1]  rep(0.01, dim)
#> 
#> Setting for 'initial_mean':
#> [1]  rep(1, dim)
#> 
#> Setting for 'initial_prec':
#> [1]  0.1 * diag(dim)
```

### SSVS

``` r

(fit_ssvs <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_ssvs(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_ssvs(), 
#>     cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with Stochastic Volatility
#> Fitted by Gibbs sampling
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 177 variables
#>      phi[1]   phi[2]    phi[3]  phi[4]     phi[5]   phi[6]  phi[7]   phi[8]
#> 1    0.0719   0.2090   0.10809  -0.980  -0.000526   0.3464   0.135  -0.0685
#> 2   -0.0233  -0.3991  -0.10453   0.414   0.075749   1.2320  -0.152  -0.1220
#> 3    0.0784  -0.1957  -0.04816   0.281  -0.027756   0.7372   0.145  -0.2823
#> 4   -0.1395  -0.1106  -0.06796  -0.135  -0.054614   0.8069   0.359  -0.0863
#> 5   -0.1870  -0.0175   0.03601  -0.102  -0.153686   0.7294   0.615  -0.3071
#> 6    0.4732  -0.0425   0.43854  -0.533  -0.045269   0.0988   1.189  -0.3845
#> 7    0.1115  -0.0724   0.07539  -0.499   0.042667  -0.0100   1.323  -0.2860
#> 8   -0.1291  -0.2984   0.00408   0.473   0.017545  -0.1782   1.586  -0.1215
#> 9    0.1372  -0.2387  -0.10112   0.204  -0.023939   0.6411  -0.158   0.0301
#> 10   0.0254  -0.2727  -0.05475   0.134  -0.003172   0.4632   0.218   0.0289
#> # ... with 10 more draws, and 169 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

### Horseshoe

``` r

(fit_hs <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_horseshoe(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_horseshoe(), 
#>     cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with Stochastic Volatility
#> Fitted by Gibbs sampling
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 211 variables
#>      phi[1]   phi[2]     phi[3]   phi[4]  phi[5]   phi[6]     phi[7]    phi[8]
#> 1    0.2305  -0.0596  -1.63e-03   0.1492  0.2418  -0.0172   0.445102   0.12523
#> 2    0.0807  -0.1439   8.46e-03  -0.0660  0.1012  -0.0764   0.760116   0.10467
#> 3    0.1378  -0.0970  -7.61e-03  -0.1445  0.4949   0.1310   0.114081   0.00523
#> 4    0.0641  -0.1928   5.79e-03   0.0624  0.2726  -0.0120  -0.327455  -0.04149
#> 5   -0.0749  -0.1716  -2.67e-03  -0.0628  0.1615   0.1056   0.060760  -0.06067
#> 6   -0.1128  -0.2914   8.26e-04   0.0410  0.2038   0.2036   0.017924  -0.03508
#> 7    0.1986  -0.1563   2.04e-03   0.0953  0.0288   0.4556  -0.029937   0.40808
#> 8    0.1010  -0.1697  -2.58e-04   0.0569  0.6293   0.6354   0.014622  -0.24877
#> 9    0.0250  -0.1645   9.27e-05   0.0751  0.2343   0.4192   0.007443  -0.08209
#> 10   0.0108  -0.1848  -1.46e-04   0.1697  0.5062   0.6513  -0.000994  -0.20476
#> # ... with 10 more draws, and 203 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

### Normal-Gamma prior

``` r

(fit_ng <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_ng(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_ng(), 
#>     cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with Stochastic Volatility
#> Fitted by Metropolis-within-Gibbs
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 184 variables
#>       phi[1]   phi[2]   phi[3]     phi[4]   phi[5]  phi[6]   phi[7]     phi[8]
#> 1    0.25518  -0.9770   0.4259   0.417954  -0.1846   1.283   0.3877  -0.273345
#> 2    0.48881  -0.2540  -0.8135   0.656938  -0.0854   1.172   0.4056   0.000304
#> 3   -0.21958   0.7115   0.7059  -0.015675   0.4809   0.977   1.0520  -0.063890
#> 4    0.23791  -0.1875   0.1521   0.010566   0.2176   0.892   0.3509  -0.220596
#> 5   -0.11051   0.1282   0.2192   0.002291   0.2853   1.138   0.0922  -0.186967
#> 6   -0.24014  -0.1462   0.1873  -0.002420   0.0432   0.711  -0.2923  -0.056603
#> 7    0.12877  -0.0630  -0.0936  -0.006839   0.4895   1.106   0.8267   0.109545
#> 8    0.00938  -0.3370   0.1838   0.021547  -0.2261   1.386   0.3885   0.111162
#> 9    0.00284  -0.0438   0.0134   0.004785   0.9719   1.899  -0.0484  -0.078218
#> 10  -0.01474  -0.0781  -0.0557   0.000692   0.8656   1.593  -0.4511   0.007074
#> # ... with 10 more draws, and 176 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

### Dirichlet-Laplace prior

``` r

(fit_dl <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_dl(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_dl(), 
#>     cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with Stochastic Volatility
#> Fitted by Gibbs sampling
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 178 variables
#>       phi[1]     phi[2]    phi[3]     phi[4]    phi[5]  phi[6]  phi[7]
#> 1    0.01572   2.43e-02   0.09985   0.452805   0.00257   0.966  -0.385
#> 2    0.01141   1.05e-02   0.09363   0.376326   0.00332   0.762  -0.857
#> 3   -0.02085  -4.18e-03   0.32863   0.420273  -0.01554   0.796  -0.569
#> 4   -0.05024   1.34e-03   0.45740   0.001272  -0.02070   0.867  -1.440
#> 5    0.02476  -6.36e-04   0.49315   0.001427   0.01904   0.939  -1.162
#> 6    0.02065  -4.09e-04   0.06321   0.000834   0.53316   0.899  -1.165
#> 7   -0.02309  -1.71e-03  -0.00409  -0.001295   0.77605   0.843  -1.231
#> 8   -0.00636  -1.91e-04  -0.01217  -0.000551   0.62815   0.862  -1.257
#> 9   -0.02534   2.71e-05   0.00684   0.000970   0.55055   0.871  -0.925
#> 10  -0.02044  -7.04e-05  -0.01042  -0.006080   0.71595   0.871  -0.738
#>       phi[8]
#> 1   -0.00553
#> 2    0.00288
#> 3    0.02929
#> 4   -0.39449
#> 5   -0.18117
#> 6   -0.20311
#> 7   -0.21344
#> 8   -0.34973
#> 9   -0.09570
#> 10  -0.00752
#> # ... with 10 more draws, and 170 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

### Bayesian visualization

[`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
also provides Bayesian visualization. `type = "trace"` gives MCMC trace
plot.

``` r

autoplot(fit_hs, type = "trace", regex_pars = "tau")
```

![](stochastic-volatility_files/figure-html/unnamed-chunk-1-1.png)

`type = "dens"` draws MCMC density plot.

``` r

autoplot(fit_hs, type = "dens", regex_pars = "tau")
```

![](stochastic-volatility_files/figure-html/denshs-1.png)

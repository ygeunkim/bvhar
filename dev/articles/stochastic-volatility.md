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
#> # A draws_df: 10 iterations, 2 chains, and 210 variables
#>      phi[1]   phi[2]    phi[3]   phi[4]   phi[5]     phi[6]   phi[7]   phi[8]
#> 1    0.8684  -0.2146  -0.14044   0.1378  -0.0308  -0.000384  -0.4801  -0.3490
#> 2   -0.4049  -0.0846  -0.00810   0.8647   0.4126  -0.151308  -0.1753   0.1049
#> 3    0.0770   0.2000   0.03086   0.3788  -0.0445  -0.266596  -0.0309   0.2883
#> 4    0.1682  -0.0637  -0.24831   0.7883   0.3193  -0.005273  -0.0455   0.2271
#> 5   -0.3598   0.1974   0.02541   0.6589   0.0442  -0.270894   0.0414  -0.0972
#> 6    0.1275  -0.2775   0.28456  -0.0302   0.1573  -0.245175   0.1869  -0.2225
#> 7    0.0789  -0.1542   0.18568  -0.0426  -0.0396  -0.311265   0.1153  -0.1044
#> 8    0.2895   0.0598  -0.03109   0.3774   0.2059  -0.144213   0.1770  -0.1146
#> 9    0.1676  -0.1985   0.04614  -0.3004   0.3456  -0.358534   1.1948   0.1381
#> 10   0.1313  -0.0969   0.00871  -0.6039  -0.0129  -0.373374   1.2069   0.4315
#> # ... with 10 more draws, and 202 more variables
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
#> # A draws_df: 10 iterations, 2 chains, and 215 variables
#>      phi[1]   phi[2]   phi[3]   phi[4]     phi[5]    phi[6]   phi[7]   phi[8]
#> 1    0.2362   0.1463  -0.0517  -0.0742  -1.66e-03  -0.06187   0.1833   0.1611
#> 2    0.1831   0.1819  -0.1090   0.1610   8.89e-03  -0.00887   0.0232   0.1321
#> 3    0.2455   0.1286  -0.0703   0.1222  -8.73e-03  -0.05949  -0.1382   0.1039
#> 4    0.0963   0.1087  -0.1558   0.1166   5.65e-03   0.02191   0.1070   0.1130
#> 5    0.0372   0.1497  -0.1538   0.1946  -2.67e-03  -0.02359  -0.0237   0.2995
#> 6   -0.0573   0.1079  -0.2566   0.5879   7.73e-04   0.01944   0.0156   0.0960
#> 7    0.1053   0.0943  -0.1837   0.2004   1.80e-03   0.10304   0.0697   0.2996
#> 8    0.0232  -0.0412  -0.1501   0.3045  -2.18e-04  -0.13582   0.0191  -0.0158
#> 9    0.0504   0.1025  -0.1617   0.2406   9.16e-05   0.00157  -0.0346   0.2380
#> 10   0.0223   0.0831  -0.1558   0.1374  -1.42e-04  -0.01194   0.0992   0.1018
#> # ... with 10 more draws, and 207 more variables
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
#> # A draws_df: 10 iterations, 2 chains, and 188 variables
#>       phi[1]    phi[2]    phi[3]   phi[4]   phi[5]   phi[6]   phi[7]  phi[8]
#> 1   -0.12129   0.46645   0.00633  -0.2542   0.0983  -0.1233   0.8472   1.195
#> 2   -0.06070  -0.03576   0.26003   0.3627  -0.0493   0.1058  -0.6588  -0.233
#> 3   -0.12030   0.09924   0.06952   0.6217   0.2413   0.2023  -0.3341   0.896
#> 4   -0.05398  -0.83427   0.32929  -0.7151   0.5682  -0.7710   1.8176   0.311
#> 5   -0.08097  -0.00469  -0.01688   1.0261  -0.0247   0.2056  -0.7039   0.113
#> 6   -0.09683   0.00503   0.00561   0.9218  -0.3035  -0.0282  -0.8611   0.264
#> 7   -0.04153   0.00752  -0.03169   0.8208   0.4102  -0.0718   0.0765   0.490
#> 8    0.00496  -0.18928   0.07010   0.4604   0.3786   0.0708   0.0404   0.923
#> 9   -0.02271   0.11403   0.26097   0.2908  -0.1860  -0.5323  -0.0621   0.049
#> 10  -0.01412   0.02235   0.01678  -0.0771  -0.3385  -0.2547  -0.0895   0.377
#> # ... with 10 more draws, and 180 more variables
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
#> # A draws_df: 10 iterations, 2 chains, and 182 variables
#>       phi[1]    phi[2]     phi[3]    phi[4]    phi[5]    phi[6]   phi[7]
#> 1    0.16969  -0.24454  -0.126475  -0.06212  -0.00613   0.10133   0.2047
#> 2    0.13486  -0.04642  -0.148850   0.29445  -0.00510   0.03525   0.1231
#> 3    0.09879  -0.00417   0.012937   0.08343  -0.00992  -0.03586   0.0190
#> 4   -0.05609  -0.02515  -0.009510   0.27036  -0.04458  -0.00890  -0.0160
#> 5    0.00416   0.02970   0.007852  -0.08386  -0.01198  -0.13536   0.0647
#> 6   -0.00753  -0.13360  -0.053984   0.06215   0.01830   0.02951  -0.1006
#> 7    0.03670   0.06412  -0.129178  -0.00837   0.00700   0.06118   0.1877
#> 8    0.04795   0.01647  -0.091950   0.01753   0.02061   0.03345   0.4122
#> 9    0.00318   0.01126   0.000964   0.01404  -0.00825  -0.01743   0.3645
#> 10   0.04624  -0.09924   0.000799  -0.00658   0.02443   0.00132   0.3980
#>        phi[8]
#> 1    8.58e-05
#> 2   -8.17e-04
#> 3   -7.97e-04
#> 4    2.09e-03
#> 5    5.46e-03
#> 6    1.57e-02
#> 7    3.77e-01
#> 8    5.26e-01
#> 9    4.34e-01
#> 10   3.80e-01
#> # ... with 10 more draws, and 174 more variables
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

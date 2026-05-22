# Bayesian VAR and VHAR Models

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

## Bayesian VAR and VHAR

[`var_bayes()`](../reference/var_bayes.md) and
[`vhar_bayes()`](../reference/vhar_bayes.md) fit BVAR and BVHAR each
with various priors.

- `y`: Multivariate time series data. It should be data frame or matrix,
  which means that every column is numeric. Each column indicates
  variable, i.e. it sould be wide format.
- `p` or `har`: VAR lag, or order of VHAR
- `num_chains`: Number of chains
  - If OpenMP is enabled, parallel loop will be run.
- `num_iter`: Total number of iterations
- `num_burn`: Number of burn-in
- `thinning`: Thinning
- `coef_spec`: Coefficient prior specification.
  - Minneosta prior
    - BVAR: [`set_bvar()`](../reference/set_bvar.md)
    - BVHAR: [`set_bvhar()`](../reference/set_bvar.md) and
      [`set_weight_bvhar()`](../reference/set_bvar.md)
    - Can induce prior on $`\lambda`$ using `lambda = set_lambda()`
  - SSVS prior: [`set_ssvs()`](../reference/set_ssvs.md)
  - Horseshoe prior: [`set_horseshoe()`](../reference/set_horseshoe.md)
  - NG prior: [`set_ng()`](../reference/set_ng.md)
  - DL prior: [`set_dl()`](../reference/set_dl.md)
- `contem_spec`: Contemporaneous prior specification.
- `cov_spec`: Covariance prior specification. Use
  [`set_ldlt()`](../reference/set_ldlt.md) for homoskedastic model.
- `include_mean = TRUE`: By default, you include the constant term in
  the model.
- `minnesota = c("no", "short", "longrun")`: Minnesota-type shrinkage.
- `verbose = FALSE`: Progress bar
- `num_thread`: Number of thread for OpenMP
  - Used in parallel multi-chain loop
  - This option is valid only when OpenMP in user’s machine.

### Stochastic Search Variable Selection (SSVS) Prior

``` r

(fit_ssvs <- vhar_bayes(etf_train, num_chains = 1, num_iter = 20, coef_spec = set_ssvs(), contem_spec = set_ssvs(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 1, num_iter = 20, coef_spec = set_ssvs(), 
#>     contem_spec = set_ssvs(), cov_spec = set_ldlt(), include_mean = FALSE, 
#>     minnesota = "longrun")
#> 
#> BVHAR with SSVS prior + SSVS prior
#> Fitted by Gibbs sampling
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 1 chains, and 90 variables
#>       phi[1]    phi[2]   phi[3]    phi[4]   phi[5]   phi[6]    phi[7]   phi[8]
#> 1   -0.12653   0.33809  -0.4063   0.83450  -0.0837   0.0786  -0.06567   0.1567
#> 2    0.14062   0.20715  -0.4043   0.06268   0.1470   0.0134   0.66528  -0.0353
#> 3    0.50002   0.21007  -0.3089   0.03100   0.0705   0.0383   0.20525  -0.0334
#> 4    0.19881  -0.07308  -0.0889  -0.00346  -0.2732  -0.0646   0.00358  -0.0982
#> 5    0.01917   0.00323  -0.1040   0.05931  -0.1828  -0.1766  -0.11164   0.2980
#> 6    0.05360   0.09245  -0.0783  -0.14020  -0.2887  -0.1908  -0.19322  -0.0847
#> 7    0.04501   0.09108  -0.0817  -0.97082  -0.0987  -0.2435   1.25380   1.0925
#> 8    0.24950   0.53442  -0.2132  -2.09965  -0.6910  -0.2955   2.07689   2.0892
#> 9    0.03687  -0.02904   0.0204  -1.24057  -0.2383  -0.4127   1.08957   0.9673
#> 10  -0.00362  -0.07775   0.0124  -1.39975   0.1540  -0.5541   1.77939   1.6711
#> # ... with 82 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

[`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
for the fit (`bvharsp` object) provides coefficients heatmap. There is
`type` argument, and the default `type = "coef"` draws the heatmap.

``` r

autoplot(fit_ssvs)
#> Warning: `label` cannot be a <ggplot2::element_blank> object.
#> `label` cannot be a <ggplot2::element_blank> object.
#> `label` cannot be a <ggplot2::element_blank> object.
```

![](shrinkage_files/figure-html/heatssvs-1.png)

### Horseshoe Prior

`coef_spec` is the initial specification by
[`set_horseshoe()`](../reference/set_horseshoe.md). Others are the same.

``` r

(fit_hs <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_horseshoe(), contem_spec = set_horseshoe(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_horseshoe(), 
#>     contem_spec = set_horseshoe(), cov_spec = set_ldlt(), include_mean = FALSE, 
#>     minnesota = "longrun")
#> 
#> BVHAR with Horseshoe prior + Horseshoe prior
#> Fitted by Gibbs sampling
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 124 variables
#>      phi[1]    phi[2]     phi[3]    phi[4]    phi[5]  phi[6]    phi[7]   phi[8]
#> 1    0.3966  -0.04871  -0.073796  -0.11783   0.00293   1.078   0.01430  -0.0790
#> 2   -0.0702  -0.02313   0.102949   0.23774  -0.01323   0.919   0.01125  -0.1105
#> 3    0.0616  -0.16276  -0.022509   0.34429   0.04521   1.022   0.01897   0.0413
#> 4   -0.0212  -0.20882  -0.080171   0.28627  -0.07634   0.978  -0.04182  -0.0055
#> 5    0.0240  -0.21210   0.000561   0.19611  -0.00800   1.055   0.04668  -0.0653
#> 6    0.1203  -0.00324   0.020411   0.00470   0.04876   0.965   0.00531  -0.0131
#> 7    0.0786  -0.06914   0.035252   0.02279   0.02359   1.020   0.00335   0.0411
#> 8    0.0973   0.10813  -0.126038   0.06279  -0.01706   0.887   0.03039  -0.0348
#> 9    0.1122  -0.09498   0.123461  -0.02566   0.00284   0.747  -0.05953   0.0396
#> 10   0.1360   0.00468   0.312362   0.00279  -0.03629   0.854  -0.00382  -0.0222
#> # ... with 10 more draws, and 116 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

``` r

autoplot(fit_hs)
#> Warning: `label` cannot be a <ggplot2::element_blank> object.
#> `label` cannot be a <ggplot2::element_blank> object.
#> `label` cannot be a <ggplot2::element_blank> object.
```

![](shrinkage_files/figure-html/heaths-1.png)

### Minnesota Prior

``` r

(fit_mn <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_bvhar(lambda = set_lambda()), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_bvhar(lambda = set_lambda()), 
#>     cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with MN_Hierarchical prior + MN_Hierarchical prior
#> Fitted by Gibbs sampling
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 63 variables
#>     phi[1]   phi[2]   phi[3]   phi[4]   phi[5]  phi[6]   phi[7]   phi[8]
#> 1   0.4138  -0.0515   0.3069  -0.1077   0.4402   1.101  -0.1993  -0.0312
#> 2   0.3537  -0.2305  -0.0559  -0.1199   0.2445   0.917   0.0434   0.1963
#> 3   0.2705  -0.0465   0.0360   0.0420   0.1887   0.874   0.1031  -0.1930
#> 4   0.3118  -0.1969  -0.0404   0.0694   0.0781   0.823   0.1961   0.0728
#> 5   0.4405   0.0575   0.0506   0.0897   0.0893   0.801   0.1058   0.2704
#> 6   0.0848  -0.2301   0.1407   0.0722   0.1061   1.128   0.2398   0.2817
#> 7   0.1809  -0.0362   0.1832   0.1701   0.0527   1.200   0.3462   0.1840
#> 8   0.1078  -0.2035   0.2829   0.3574  -0.0320   1.036   0.1089  -0.0697
#> 9   0.1737  -0.0848   0.3014   0.2573  -0.0514   0.785   0.0675   0.1181
#> 10  0.2752  -0.1289   0.3415  -0.1696   0.0768   0.940  -0.4074  -0.0673
#> # ... with 10 more draws, and 55 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

### Normal-Gamma prior

``` r

(fit_ng <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_ng(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_ng(), 
#>     cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with NG prior + NG prior
#> Fitted by Metropolis-within-Gibbs
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 97 variables
#>       phi[1]    phi[2]    phi[3]  phi[4]  phi[5]  phi[6]   phi[7]   phi[8]
#> 1   -0.00173  -0.24368  -0.29826   0.102   0.207   0.781   0.1525  -0.0774
#> 2   -0.09968  -0.03622   0.38895   0.540  -0.591   1.112  -0.0653   0.0068
#> 3    0.35748  -0.30801  -0.10203   0.391   0.210   0.908  -0.0314   0.0480
#> 4   -0.04341  -0.31232  -0.05314   0.215   0.342   0.623  -0.3212  -0.4890
#> 5    0.02756  -0.21996  -0.00452   0.399  -0.353   0.896   0.4851   0.1693
#> 6    0.04017  -0.23864   0.04099   0.242  -0.395   0.767   0.1866  -0.1876
#> 7    0.01861  -0.02194  -0.01980   0.275  -0.101   0.824  -0.1321  -0.0942
#> 8    0.03002   0.01648   0.06374   0.120  -0.211   0.731   0.2664  -0.2778
#> 9    0.01933   0.00377  -0.03888   0.237  -0.293   0.955   0.7503   0.0754
#> 10  -0.05574  -0.26308  -0.69345  -0.511   0.607   0.774   0.6656  -0.0587
#> # ... with 10 more draws, and 89 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

### Dirichlet-Laplace prior

``` r

(fit_dl <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, coef_spec = set_dl(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))
#> Call:
#> vhar_bayes(y = etf_train, num_chains = 2, num_iter = 20, coef_spec = set_dl(), 
#>     cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun")
#> 
#> BVHAR with DL prior + DL prior
#> Fitted by Gibbs sampling
#> Number of chains: 2
#> Total number of iteration: 20
#> Number of burn-in: 10
#> ====================================================
#> 
#> Parameter Record:
#> # A draws_df: 10 iterations, 2 chains, and 91 variables
#>       phi[1]    phi[2]     phi[3]     phi[4]    phi[5]  phi[6]    phi[7]
#> 1   -0.00239   0.02170   0.000518  -0.495052   0.24836   1.052  -0.06795
#> 2   -0.11489  -0.09058   0.001962   0.288790   0.15949   0.841   0.10399
#> 3    0.09675  -0.05739  -0.007962   0.023629   0.14821   0.794  -0.07386
#> 4    0.10078  -0.04879   0.002286  -0.009170  -0.06958   1.039   0.10305
#> 5   -0.03324  -0.03554   0.002868   0.020514  -0.02233   1.000   0.34392
#> 6   -0.02685   0.00793   0.012527   0.008080   0.40298   0.787  -0.18903
#> 7    0.06241   0.00953   0.020831  -0.001462   0.16422   0.801  -0.00514
#> 8    0.17010   0.04481  -0.054743  -0.000354   0.80419   0.861  -0.00458
#> 9    0.02462  -0.07490   0.088378   0.000552   0.01114   0.986  -0.00428
#> 10   0.05811  -0.04678  -0.030620  -0.000847  -0.00922   0.858  -0.00381
#>        phi[8]
#> 1   -4.29e-02
#> 2   -1.29e-02
#> 3   -1.73e-02
#> 4    1.04e-03
#> 5   -4.67e-04
#> 6   -3.63e-05
#> 7    2.92e-05
#> 8    3.67e-06
#> 9    1.14e-06
#> 10  -6.19e-04
#> # ... with 10 more draws, and 83 more variables
#> # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

## Bayesian visualization

[`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
also provides Bayesian visualization. `type = "trace"` gives MCMC trace
plot.

``` r

autoplot(fit_hs, type = "trace", regex_pars = "tau")
```

![](shrinkage_files/figure-html/unnamed-chunk-1-1.png)

`type = "dens"` draws MCMC density plot. If specifying additional
argument `facet_args = list(dir = "v")` of `bayesplot`, you can see plot
as the same format with coefficient matrix.

``` r

autoplot(fit_hs, type = "dens", regex_pars = "kappa", facet_args = list(dir = "v", nrow = nrow(fit_hs$coefficients)))
```

![](shrinkage_files/figure-html/denshs-1.png)

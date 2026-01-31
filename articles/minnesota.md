# Minnesota Prior

``` r
library(bvhar)
```

## Normal-inverse-Wishart Matrix

We provide functions to generate matrix-variate Normal and
inverse-Wishart.

- `sim_mnormal(num_sim, mu, sig)`: `num_sim` of
  $\mathbf{X}_{i}\overset{iid}{\sim}N({\mathbf{μ}},\Sigma)$.
- `sim_matgaussian(mat_mean, mat_scale_u, mat_scale_v)`: One
  $X_{m \times n} \sim MN\left( M_{m \times n},U_{m \times m},V_{n \times n} \right)$
  which means that $vec(X) \sim N\left( vec(M),V \otimes U \right)$.
- `sim_iw(mat_scale, shape)`: One $\Sigma \sim IW(\Psi,\nu)$.
- `sim_mniw(num_sim, mat_mean, mat_scale_u, mat_scale, shape)`:
  `num_sim` of
  $\left( X_{i},\Sigma_{i} \right)\overset{iid}{\sim}MNIW(M,U,V,\nu)$.

Multivariate Normal generation gives `num_sim` x dim matrix. For
example, generating 3 vector from
Normal(${\mathbf{μ}} = \mathbf{0}_{2}$,
$\Sigma = diag\left( \mathbf{1}_{2} \right)$):

``` r
sim_mnormal(3, rep(0, 2), diag(2))
#>        [,1]   [,2]
#> [1,] -0.626  0.184
#> [2,] -0.836  1.595
#> [3,]  0.330 -0.820
```

The output of [`sim_matgaussian()`](../reference/sim_matgaussian.md) is
a matrix.

``` r
sim_matgaussian(matrix(1:20, nrow = 4), diag(4), diag(5), FALSE)
#>      [,1] [,2]  [,3] [,4] [,5]
#> [1,] 1.49 5.74  9.58 12.7 18.5
#> [2,] 2.39 5.38  7.79 15.1 18.0
#> [3,] 2.98 7.94 11.82 15.6 19.9
#> [4,] 4.78 8.07 10.01 16.6 19.9
```

When generating IW, violating $\nu > dim - 1$ gives error. But we ignore
$\nu > dim + 1$ (condition for mean existence) in this function.
Nonetheless, we recommend you to keep $\nu > dim + 1$ condition. As
mentioned, it guarantees the existence of the mean.

``` r
sim_iw(diag(5), 7)
#>        [,1]   [,2]   [,3]   [,4]   [,5]
#> [1,]  0.588  0.428  0.219  0.364 -0.562
#> [2,]  0.428  0.632  0.262  0.512 -0.517
#> [3,]  0.219  0.262  0.463  0.289 -0.551
#> [4,]  0.364  0.512  0.289  0.693 -0.565
#> [5,] -0.562 -0.517 -0.551 -0.565  1.084
```

In case of [`sim_mniw()`](../reference/sim_mniw.md), it returns list
with `mn` (stacked MN matrices) and `iw` (stacked IW matrices). Each
`mn` and `iw` has draw lists.

``` r
sim_mniw(2, matrix(1:20, nrow = 4), diag(4), diag(5), 7, FALSE)
#> $mn
#> $mn[[1]]
#>      [,1] [,2]  [,3] [,4] [,5]
#> [1,] 2.33 5.71  8.22 13.1 16.1
#> [2,] 2.10 4.59 11.37 13.2 18.9
#> [3,] 3.26 6.57 11.36 14.0 16.9
#> [4,] 4.16 7.73 12.02 15.9 19.5
#> 
#> $mn[[2]]
#>       [,1] [,2]  [,3] [,4] [,5]
#> [1,] 0.434 4.50  9.01 12.7 16.8
#> [2,] 1.100 6.54  9.89 14.7 18.5
#> [3,] 3.899 6.80 11.44 14.8 18.6
#> [4,] 5.415 6.89 11.75 16.3 19.5
#> 
#> 
#> $iw
#> $iw[[1]]
#>         [,1]   [,2]    [,3]    [,4]    [,5]
#> [1,]  0.3060  0.171 -0.2698  0.0521 -0.0726
#> [2,]  0.1711  0.755 -0.4490  0.3805  0.1666
#> [3,] -0.2698 -0.449  0.6684 -0.2751 -0.0972
#> [4,]  0.0521  0.381 -0.2751  0.6968  0.8058
#> [5,] -0.0726  0.167 -0.0972  0.8058  1.4994
#> 
#> $iw[[2]]
#>        [,1]    [,2]    [,3]    [,4]    [,5]
#> [1,]  0.976 -0.4780 -0.1529  0.2778 -0.1651
#> [2,] -0.478  0.6363  0.0262 -0.2094  0.2551
#> [3,] -0.153  0.0262  0.1619 -0.0537 -0.0292
#> [4,]  0.278 -0.2094 -0.0537  0.3866 -0.0893
#> [5,] -0.165  0.2551 -0.0292 -0.0893  0.3105
```

This function has been defined for the next simulation functions.

## Minnesota Prior

### BVAR

Consider BVAR Minnesota prior setting,

$$A \sim MN\left( A_{0},\Omega_{0},\Sigma_{e} \right)$$

$$\Sigma_{e} \sim IW\left( S_{0},\alpha_{0} \right)$$

- From Litterman (1986) and Bańbura et al. (2010)
- Each $A_{0},\Omega_{0},S_{0},\alpha_{0}$ is defined by adding dummy
  observations
  - `build_xdummy()`
  - `build_ydummy()`
- `sigma`: Vector $\sigma_{1},\ldots,\sigma_{m}$
  - $\Sigma_{e} = diag\left( \sigma_{1}^{2},\ldots,\sigma_{m}^{2} \right)$
  - $\sigma_{i}^{2}/\sigma_{j}^{2}$: different scale and variability of
    the data
- `lambda`
  - Controls the overall tightness of the prior distribution around the
    RW or WN
  - Governs the relative importance of the prior beliefs w.r.t. the
    information contained in the data
    - If $\lambda = 0$, then posterior = prior and the data do not
      influence the estimates.
    - If $\lambda = \infty$, then posterior expectations = OLS.
  - Choose in relation to the size of the system (Bańbura et al. (2010))
    - As `m` increases, $\lambda$ should be smaller to avoid overfitting
      (De Mol et al. (2008))
- `delta`: Persistence
  - Litterman (1986) originally sets high persistence $\delta_{i} = 1$
  - For Non-stationary variables: random walk prior $\delta_{i} = 1$
  - For stationary variables: white noise prior $\delta_{i} = 0$
- `eps`: Very small number to make matrix invertible

``` r
bvar_lag <- 5
(spec_to_sim <- set_bvar(
  sigma = c(3.25, 11.1, 2.2, 6.8), # sigma vector
  lambda = .2, # lambda
  delta = rep(1, 4), # 4-dim delta vector
  eps = 1e-04 # very small number
))
#> Model Specification for BVAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: Minnesota
#> ========================================================
#> 
#> Setting for 'sigma':
#> [1]   3.25  11.10   2.20   6.80
#> 
#> Setting for 'lambda':
#> [1]  0.2
#> 
#> Setting for 'delta':
#> [1]  1  1  1  1
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'hierarchical':
#> [1]  FALSE
```

- `sim_mncoef(p, bayes_spec, full = TRUE)` can generate both $A$ and
  $\Sigma$ matrices.
- In `bayes_spec`, only [`set_bvar()`](../reference/set_bvar.md) works.
- If `full = FALSE`, $\Sigma$ is not random. It is same as `diag(sigma)`
  from the `bayes_spec`.
- `full = TRUE` is the default.

``` r
(sim_mncoef(bvar_lag, spec_to_sim))
#> $coefficients
#>           [,1]     [,2]      [,3]    [,4]
#>  [1,]  0.94670  0.04018 -0.015784  0.2745
#>  [2,] -0.07307  1.05811  0.004270  0.1084
#>  [3,] -0.12827 -0.36258  1.047951 -0.3413
#>  [4,]  0.09243 -0.35626 -0.068964  0.8931
#>  [5,] -0.06170  0.45019  0.040800 -0.3748
#>  [6,] -0.03948  0.03861  0.019841  0.0436
#>  [7,] -0.11283 -0.42310 -0.012247  0.7918
#>  [8,] -0.02440 -0.13014  0.041177  0.1533
#>  [9,] -0.01307  0.14998  0.031109 -0.1510
#> [10,]  0.03539 -0.02044 -0.027690 -0.0599
#> [11,]  0.01680  0.46625  0.006200 -0.2890
#> [12,] -0.00202 -0.02148  0.000093  0.0578
#> [13,]  0.08527  0.08206 -0.017698 -0.3457
#> [14,]  0.01184  0.00329 -0.013434 -0.0112
#> [15,] -0.00964  0.22695 -0.010620 -0.1717
#> [16,] -0.01819 -0.00357  0.012219  0.0124
#> [17,]  0.02730 -0.10795 -0.030850  0.1132
#> [18,] -0.00978  0.01280  0.003423  0.0204
#> [19,]  0.08201  0.17083 -0.042995 -0.4951
#> [20,]  0.03924  0.01520 -0.014961 -0.0968
#> 
#> $covmat
#>        [,1]   [,2]  [,3]   [,4]
#> [1,]   7.13  -2.04 -3.51 -13.83
#> [2,]  -2.04  45.95  2.03 -27.17
#> [3,]  -3.51   2.03  3.13   5.27
#> [4,] -13.83 -27.17  5.27  73.48
```

### BVHAR

`sim_mnvhar_coef(bayes_spec, full = TRUE)` generates BVHAR model
setting:

$$\Phi \mid \Sigma_{e} \sim MN\left( M_{0},\Omega_{0},\Sigma_{e} \right)$$

$$\Sigma_{e} \sim IW\left( \Psi_{0},\nu_{0} \right)$$

- similar to BVAR, `bayes_spec` option wants `bvharspec`. But
  - [`set_bvhar()`](../reference/set_bvar.md)
  - [`set_weight_bvhar()`](../reference/set_bvar.md)
- There is `full = TRUE`, too.

#### BVHAR-S

``` r
(bvhar_var_spec <- set_bvhar(
  sigma = c(1.2, 2.3), # sigma vector
  lambda = .2, # lambda
  delta = c(.3, 1), # 2-dim delta vector
  eps = 1e-04 # very small number
))
#> Model Specification for BVHAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_VAR
#> ========================================================
#> 
#> Setting for 'sigma':
#> [1]  1.2  2.3
#> 
#> Setting for 'lambda':
#> [1]  0.2
#> 
#> Setting for 'delta':
#> [1]  0.3  1.0
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'hierarchical':
#> [1]  FALSE
```

``` r
(sim_mnvhar_coef(bvhar_var_spec))
#> $coefficients
#>         [,1]      [,2]
#> [1,]  0.2549 -0.265804
#> [2,]  0.0581  1.158779
#> [3,] -0.0174 -0.121060
#> [4,]  0.0189  0.000728
#> [5,] -0.0651 -0.009972
#> [6,] -0.0124 -0.013118
#> 
#> $covmat
#>       [,1]  [,2]
#> [1,] 0.457 0.071
#> [2,] 0.071 1.295
```

#### BVHAR-L

``` r
(bvhar_vhar_spec <- set_weight_bvhar(
  sigma = c(1.2, 2.3), # sigma vector
  lambda = .2, # lambda
  eps = 1e-04, # very small number
  daily = c(.5, 1), # 2-dim daily weight vector
  weekly = c(.2, .3), # 2-dim weekly weight vector
  monthly = c(.1, .1) # 2-dim monthly weight vector
))
#> Model Specification for BVHAR
#> 
#> Parameters: Coefficent matrice and Covariance matrix
#> Prior: MN_VHAR
#> ========================================================
#> 
#> Setting for 'sigma':
#> [1]  1.2  2.3
#> 
#> Setting for 'lambda':
#> [1]  0.2
#> 
#> Setting for 'eps':
#> [1]  1e-04
#> 
#> Setting for 'daily':
#> [1]  0.5  1.0
#> 
#> Setting for 'weekly':
#> [1]  0.2  0.3
#> 
#> Setting for 'monthly':
#> [1]  0.1  0.1
#> 
#> Setting for 'hierarchical':
#> [1]  FALSE
```

``` r
(sim_mnvhar_coef(bvhar_vhar_spec))
#> $coefficients
#>          [,1]    [,2]
#> [1,]  0.56613  0.1127
#> [2,] -0.17248  0.4087
#> [3,] -0.00739  0.0752
#> [4,] -0.00522  0.1394
#> [5,]  0.09602 -0.1269
#> [6,]  0.06397  0.0498
#> 
#> $covmat
#>      [,1] [,2]
#> [1,] 4.05 1.06
#> [2,] 1.06 5.00
```
